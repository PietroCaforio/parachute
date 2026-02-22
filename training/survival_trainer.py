"""Survival trainers and auxiliary modules for PaRaChute."""

import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional


import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import json

sys.path.insert(0, "./")

from torch.utils.data import DataLoader  # noqa E402
from .trainer import BaseTrainer
from .metrics import cindex


class GCSController(nn.Module):
    """Gradient curvature steering controller."""
    def __init__(self, use_missing_flag=False):
        """
        Initialize the controller.

        :param use_missing_flag: Whether to include missing-modality flag.
        :type use_missing_flag: bool
        :return: None.
        :rtype: None
        """
        super().__init__()

        # Input dimension is 2 (curvature and grad norm) + 1 if missing_flag is used
        in_dim = 2 + int(use_missing_flag)

        # Linear layer that maps input features to a single scalar score
        self.linear = nn.Linear(in_dim, 1)

        # Sigmoid to put the output into [0, 1] range (interpreted as controller weight)
        self.sigmoid = nn.Sigmoid()

        # Store whether the controller should consider missing modality info
        self.use_missing_flag = use_missing_flag

    def forward(self, curvature_estimate, grad_norm, missing_flag=None):
        """
        Compute controller weights from curvature and gradient norm.

        :param curvature_estimate: Curvature estimate per sample.
        :type curvature_estimate: torch.Tensor
        :param grad_norm: Gradient norm per sample.
        :type grad_norm: torch.Tensor
        :param missing_flag: Optional missing-modality indicator per sample.
        :type missing_flag: torch.Tensor or None
        :return: Controller weights in ``[0, 1]`` of shape ``[B, 1]``.
        :rtype: torch.Tensor
        """
        # Unsqueeze to make curvature and gradient norm tensors shape [B, 1]
        x = [curvature_estimate.unsqueeze(1), grad_norm.unsqueeze(1)]

        # If enabled, include the missing modality flag in the input
        if self.use_missing_flag:
            assert missing_flag is not None  # Ensure it's provided
            x.append(missing_flag.unsqueeze(1))  # Shape [B, 1]

        # Concatenate all inputs to form a [B, in_dim] tensor
        x = torch.cat(x, dim=1)

        # Pass through the linear layer + sigmoid to get controller values in [0, 1]
        return self.sigmoid(self.linear(x))  # Output shape: [B, 1]


class SurvivalTrainer(BaseTrainer):
    """Trainer for multimodal CT-WSI survival learning."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the survival trainer.

        :param args: Positional arguments for :class:`BaseTrainer`.
        :type args: tuple
        :param kwargs: Keyword arguments for :class:`BaseTrainer`.
        :type kwargs: dict
        :return: None.
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        # Add default metric functions if not provided
        if not self.metric_functions:
            self.metric_functions = {
                "cindex": cindex,
            }
        self.gcs_controller = GCSController(use_missing_flag=True).to(self.device)

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to device and standardize keys.

        :param batch: Raw batch from the dataset.
        :type batch: Dict[str, torch.Tensor]
        :return: Processed batch dictionary.
        :rtype: Dict[str, torch.Tensor]
        """
        return {
            "ct_feat": batch["ct_feature"].float().to(self.device),
            "wsi_feat": batch["wsi_feature"].float().to(self.device),
            "survtimes": batch["survtime"].to(self.device),
            "censors": batch["censor"].to(self.device),
            "modality_mask": batch["modality_mask"].to(self.device),
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        :return: Training metrics for the epoch.
        :rtype: Dict[str, float]
        """
        self.model.train()
        metrics = defaultdict(float)
        # metrics = {
        #    "train_loss": 0.0,
        #    "train_accuracy": 0.0,
        #    "G1_TrainAcc": 0.0,
        #    "G2_TrainAcc": 0.0,
        #    "G3_TrainAcc": 0.0,
        # }
        num_batches = len(self.train_loader)
        total_outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        total_survtimes = torch.tensor([], dtype=torch.long, device=self.device)
        total_censors = torch.tensor([], dtype=torch.long, device=self.device)

        for batch_idx, batch in enumerate(self.train_loader):
            # Process batch
            batch_data = self.process_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                batch_data["ct_feat"],
                batch_data["wsi_feat"],
                modality_flag=batch_data["modality_mask"],
                output_layers=[
                    "hazard",
                    "fused_features",
                    "adapted_rad",
                    "adapted_histo",
                ],
            )
            # Compute loss
            loss = self.criterion(
                outputs["hazard"], batch_data["survtimes"], batch_data["censors"]
            )

            # ====== GRADIENT STEERING =====

            # Compute gradient manually wrt to f_att
            grad_f_att = torch.autograd.grad(
                outputs=loss,
                inputs=outputs["fused_features"],
                create_graph=True,
                retain_graph=True,
            )[0]

            grad_norm = grad_f_att.flatten(1).norm(p=2, dim=1)  # [B]
            # Estimate local curvature using Hutchinson's trick
            random_vec = grad_f_att.detach().clone().sign()
            grad_dot_random = torch.sum(grad_f_att * random_vec)
            hvp = torch.autograd.grad(
                outputs=grad_dot_random,
                inputs=outputs["fused_features"],
                retain_graph=False,
            )[0]
            curvature_estimate = torch.sum(
                hvp * random_vec, dim=list(range(1, grad_f_att.ndim))
            )  # [B]
            # self.logger.info(f"curvature_estimate {curvature_estimate}")
            # Missing modality flag (1 if any missing)
            # print("modality_mask", batch_data["modality_mask"])
            rad_mask = batch_data["modality_mask"][:, 0].bool()
            histo_mask = batch_data["modality_mask"][:, 1].bool()

            # self.logger.info(f"histo_mask {histo_mask}")
            missing_flag = (~rad_mask | ~histo_mask).float()  # [B]

            curvature_missingmod = curvature_estimate[missing_flag.bool()]
            curvature_fullmod = curvature_estimate[~(missing_flag.bool())]

            # Gate value
            controller = self.gcs_controller(
                curvature_estimate.detach(), grad_norm.detach(), missing_flag.detach()
            )  # [B, 1]
            controller = controller.view(
                -1, *[1] * (grad_f_att.ndim - 1)
            )  # reshape for broadcasting

            gamma = self.model.gamma  # Learnable parameter inside the model

            steered_grad = (1 - controller) * grad_f_att + controller * (
                grad_f_att
                / (
                    1.0
                    + gamma * curvature_estimate.view(-1, *[1] * (grad_f_att.ndim - 1))
                )
            )

            # Custom backward for f_att
            outputs["fused_features"].backward(gradient=steered_grad, retain_graph=False)

            
             # Recompute hazard prediction with detached features
            fused_features_detached = outputs["fused_features"].detach()
            hazard = self.model.hazard_net(fused_features_detached.unsqueeze(1))
            # Standard Backward pass for the rest of the model
            # Final loss (detached forward) and backward
            final_loss = self.criterion(
                hazard,
                batch_data["survtimes"],
                batch_data["censors"]
            )
            final_loss.backward()

            self.optimizer.step()
            

            # Update metrics
            metrics["train_loss"] += loss.item()
            metrics["curvature"] += torch.sum(curvature_estimate)
            metrics["curvature_missingmod"] += torch.sum(curvature_missingmod)
            metrics["curvature_fullmod"] += torch.sum(curvature_fullmod)
            metrics["gradnorm"] += torch.sum(grad_norm)
            metrics["gradnorm_missingmod"] += torch.sum(grad_norm[missing_flag.bool()])
            metrics["gradnorm_fullmod"] += torch.sum(grad_norm[~(missing_flag.bool())])

            total_outputs = torch.cat((total_outputs, outputs["hazard"]), dim=0)
            total_survtimes = torch.cat(
                (total_survtimes, batch_data["survtimes"]), dim=0
            )
            total_censors = torch.cat((total_censors, batch_data["censors"]))
            # accuracy_metrics = self.metric_functions["accuracy"](
            #     outputs, batch_data["labels"]
            # )
            # class_metrics = self.metric_functions["per_class_accuracy"](
            #     outputs, batch_data["labels"]
            # )
            #
            # metrics["train_accuracy"] += accuracy_metrics["accuracy"]
            # for key, value in class_metrics.items():
            # metrics[f"{key.replace('Acc', 'TrainAcc')}"] += value

            # Log batch progress
            if (batch_idx + 1) % self.config["training"]["log_interval"] == 0:
                self.logger.info(
                    f"Epoch [{self.current_epoch+1}/{self.config['training']['num_epochs']}], "
                    f"Batch [{batch_idx+1}/{num_batches}], "
                    f"Loss: {loss.item():.4f}"
                )
        if torch.isnan(total_outputs).any():
            return None
        # Compute additional metrics
        with torch.no_grad():
            for k, v in self.metric_functions.items():
                mtrc = v(total_outputs, total_survtimes, 1 - total_censors)
                for kk, vv in mtrc.items():
                    metrics["train_" + kk] = vv
        metrics = dict(metrics)
        # Compute averages
        # for key in metrics:
        #    metrics[key] /= num_batches
        metrics["train_loss"] /= num_batches
        metrics["curvature"] /= num_batches
        metrics["curvature_missingmod"] /= num_batches
        metrics["curvature_fullmod"] /= num_batches
        metrics["gradnorm"] /= num_batches
        metrics["gradnorm_missingmod"] /= num_batches
        metrics["gradnorm_fullmod"] /= num_batches

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the current validation loader.

        :return: Validation metrics for the epoch.
        :rtype: Dict[str, float]
        """
        self.model.eval()
        metrics = defaultdict(float)
        # metrics = {
        #    "val_loss": 0.0,
        #    "val_accuracy": 0.0,
        #    "G1_ValAcc": 0.0,
        #    "G2_ValAcc": 0.0,
        #    "G3_ValAcc": 0.0,
        # }
        num_batches = len(self.val_loader)
        total_outputs = torch.tensor([], dtype=torch.long, device=self.device)
        total_survtimes = torch.tensor([], dtype=torch.long, device=self.device)
        total_censors = torch.tensor([], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for batch in self.val_loader:
                # Process batch
                batch_data = self.process_batch(batch)

                # Forward pass
                outputs = self.model(
                    batch_data["ct_feat"],
                    batch_data["wsi_feat"],
                    modality_flag=batch_data["modality_mask"],
                )["hazard"]

                # Compute loss
                loss = self.criterion(
                    outputs, batch_data["survtimes"], batch_data["censors"]
                )

                # Update metrics
                metrics["val_loss"] += loss.item()

                total_outputs = torch.cat((total_outputs, outputs), dim=0)
                total_survtimes = torch.cat(
                    (total_survtimes, batch_data["survtimes"]), dim=0
                )
                total_censors = torch.cat((total_censors, batch_data["censors"]))

                # accuracy_metrics = self.metric_functions["accuracy"](
                #     outputs, batch_data["labels"]
                # )
                # class_metrics = self.metric_functions["per_class_accuracy"](
                #     outputs, batch_data["labels"]
                # )
        #
        # metrics["val_accuracy"] += accuracy_metrics["accuracy"]
        # for key, value in class_metrics.items():
        #     metrics[f"{key.replace('Acc', 'ValAcc')}"] += value
        # Compute additional metrics
        for k, v in self.metric_functions.items():
            mtrc = v(total_outputs, total_survtimes, 1 - total_censors)

            for kk, vv in mtrc.items():

                metrics["val_" + kk] = vv
        metrics = dict(metrics)

        metrics["val_loss"] /= num_batches
        # Compute averages
        # for key in metrics:
        #  metrics[key] /= num_batches
        return metrics



class GradientSteerer(nn.Module):
    """Steer gradients based on confidence to reduce variance."""
    def __init__(self, init_blend=0.3):
        """
        Initialize the gradient steerer.

        :param init_blend: Initial blend ratio in ``[0, 1]``.
        :type init_blend: float
        :return: None.
        :rtype: None
        """
        super().__init__()
        # Learnable parameter: logit so it's constrained in [0, 1] via sigmoid
        self.logit_blend = nn.Parameter(torch.logit(torch.tensor(init_blend)))

    def forward(self, g, confidence_scores, beta=1.0, eps=1e-8):
        """
        Steer gradients using a confidence-weighted blend.

        :param g: Gradient tensor of shape ``[B, C, ...]``.
        :type g: torch.Tensor
        :param confidence_scores: Confidence per sample (lower = more uncertain).
        :type confidence_scores: torch.Tensor
        :param beta: Strength of magnitude modulation.
        :type beta: float
        :param eps: Small constant for numerical stability.
        :type eps: float
        :return: Steered gradients with the same shape as ``g``.
        :rtype: torch.Tensor
        """
        B = g.shape[0]
        if B == 1:
            return g  # can't homogenize with just 1 sample

        # Convert logit to actual blend ratio [0, 1]
        dir_blend = torch.sigmoid(self.logit_blend)

        # Flatten
        g_flat = g.flatten(1)  # [B, N]

        # Normalize direction
        g_unit = g_flat / (g_flat.norm(p=2, dim=1, keepdim=True) + eps)  # [B, N]
        g_mean = g_unit.mean(dim=0, keepdim=True)  # [1, N]
        g_blend = (1 - dir_blend) * g_unit + dir_blend * g_mean  # [B, N]

        # Confidence-based magnitude modulation
        conf_softmax = torch.softmax(confidence_scores, dim=0)  # [B]
        tau = 1.0 - beta * (conf_softmax - conf_softmax.mean())  # [B]

        # Restore magnitude
        g_mag = g_flat.norm(p=2, dim=1, keepdim=False)  # [B]
        g_weighted_mag = g_mag * tau  # [B]
        g_steered = g_blend * g_weighted_mag.view(-1, 1)  # [B, N]

        return g_steered.view_as(g)  # [B, C, D, H, W]


class SurvivalTrainerMultival(SurvivalTrainer):
    """
    Trainer that supports multiple validation loaders.

    After each epoch, it runs validation on each loader, tracks a separate
    best model per split, and writes split-specific checkpoints and metrics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loaders: List[DataLoader],
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: torch.device,
        experiment_name: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping: Optional[Dict] = None,
        n_validations: Optional[int] = None,
        val_loader_names: Optional[List[str]] = None,
    ):
        """
        Initialize a multi-validation trainer.

        :param model: Model to train.
        :type model: torch.nn.Module
        :param train_loader: Training data loader.
        :type train_loader: torch.utils.data.DataLoader
        :param val_loaders: List of validation data loaders.
        :type val_loaders: List[torch.utils.data.DataLoader]
        :param criterion: Loss function.
        :type criterion: torch.nn.Module
        :param optimizer: Optimizer instance.
        :type optimizer: torch.optim.Optimizer
        :param config: Configuration dictionary.
        :type config: Dict
        :param device: Device to run on.
        :type device: torch.device
        :param experiment_name: Experiment identifier for logs/checkpoints.
        :type experiment_name: str
        :param scheduler: Optional learning-rate scheduler.
        :type scheduler: torch.optim.lr_scheduler._LRScheduler or None
        :param early_stopping: Optional early stopping configuration.
        :type early_stopping: Dict or None
        :param n_validations: Number of validation runs to average.
        :type n_validations: int or None
        :param val_loader_names: Optional names for validation loaders.
        :type val_loader_names: List[str] or None
        :return: None.
        :rtype: None
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loaders[0],
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            experiment_name=experiment_name,
            scheduler=scheduler,
            early_stopping=early_stopping,
            n_validations=n_validations,
        )

        # Override the single val_loader with the full list
        self.val_loaders = val_loaders

        # Assign or generate names for each validation split
        if val_loader_names is None:
            self.val_loader_names = [f"val_{i}" for i in range(len(val_loaders))]
        else:
            if len(val_loader_names) != len(val_loaders):
                raise ValueError("val_loader_names must match the length of val_loaders.")
            self.val_loader_names = val_loader_names

        # Determine which metric to monitor and whether lower or higher is better
        self.monitor_metric = config["training"]["monitor_metric"]
        if "monitor_mode" in config["training"]:
            self.monitor_mode = config["training"]["monitor_mode"]
        else:
            # By convention, assume "loss" metrics are to be minimized, others maximized
            self.monitor_mode = "min" if "loss" in self.monitor_metric else "max"

        # Initialize per-split best values and metric-storage
        self.best_monitor_values: List[float] = []
        self.best_metrics_per_loader: List[Dict[str, float]] = []
        for _ in self.val_loaders:
            if self.monitor_mode == "min":
                self.best_monitor_values.append(float("inf"))
            else:
                self.best_monitor_values.append(float("-inf"))
            self.best_metrics_per_loader.append({})

    def validate_loader(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run validation on a single DataLoader.

        :param loader: Validation data loader.
        :type loader: torch.utils.data.DataLoader
        :return: Validation metrics (e.g., ``val_loss`` and ``val_cindex``).
        :rtype: Dict[str, float]
        """
        self.model.eval()
        metrics = defaultdict(float)
        num_batches = len(loader)

        total_outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        total_survtimes = torch.tensor([], dtype=torch.long, device=self.device)
        total_censors = torch.tensor([], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for batch in loader:
                batch_data = self.process_batch(batch)
                # Forward pass: get hazard predictions
                outputs = self.model(
                    batch_data["ct_feat"],
                    batch_data["wsi_feat"],
                    modality_flag=batch_data["modality_mask"],
                )["hazard"]

                # Compute loss
                loss = self.criterion(
                    outputs, batch_data["survtimes"], batch_data["censors"]
                )
                metrics["val_loss"] += loss.item()

                total_outputs = torch.cat((total_outputs, outputs), dim=0)
                total_survtimes = torch.cat((total_survtimes, batch_data["survtimes"]), dim=0)
                total_censors = torch.cat((total_censors, batch_data["censors"]), dim=0)

            # Compute any additional metrics (e.g., c-index) once per loader
            for metric_name, metric_fn in self.metric_functions.items():
                result_dict = metric_fn(total_outputs, total_survtimes, 1 - total_censors)
                for k, v in result_dict.items():
                    metrics[f"val_{k}"] = v

        # Average the accumulated loss over all batches
        metrics["val_loss"] /= num_batches
        return dict(metrics)

    def _save_split_checkpoint(self, loader_idx: int):
        """
        Save a checkpoint of the current model state for the split at index loader_idx.
        Filename: {checkpoint_dir}/{experiment_name}_best_{loader_name}.pth

        :param loader_idx: Index of the validation loader/split.
        :type loader_idx: int
        :return: None.
        :rtype: None
        """
        loader_name = self.val_loader_names[loader_idx]
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metrics": self.best_metrics_per_loader[loader_idx],
            "config": self.config,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
        filepath = self.checkpoint_dir / f"{self.experiment_name}_best_{loader_name}.pth"
        torch.save(ckpt, filepath)
        self.logger.info(f"Saved new best checkpoint for split '{loader_name}' to {filepath}")

    def train(self):
        """
        Main training loop: for each epoch, run one train_epoch, then validate on all splits.
        Save separate best-model checkpoints for each split whenever its monitor metric improves.
        At the end, write a JSON file per split with that split's best metrics.

        :return: Summary of best metrics per split.
        :rtype: dict
        """
        # Loop over epochs
        num_epochs = self.config["training"]["num_epochs"]
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # ======== TRAINING PHASE ========
            train_metrics = self.train_epoch()
            if train_metrics is None:
                self.logger.info("Detected NaN values during training. Stopping early.")
                break

            # ======== VALIDATION PHASE ========
            all_val_metrics: List[Dict[str, float]] = []
            for idx, loader in enumerate(self.val_loaders):
                # Perform validation n_validations times if requested, otherwise just once
                if self.n_validations is None:
                    metrics_i = self.validate_loader(loader)
                else:
                    # Collect multiple runs
                    metrics_list = []
                    for _ in range(self.n_validations):
                        metrics_list.append(self.validate_loader(loader))
                    # Average each metric across the n_validations runs
                    averaged = {}
                    for key in metrics_list[0].keys():
                        averaged[key] = sum(d[key] for d in metrics_list) / len(metrics_list)
                    metrics_i = averaged

                all_val_metrics.append(metrics_i)

                
                # Check if this split's monitor metric improved
                mon_val = metrics_i.get(self.monitor_metric)
                if mon_val is None:
                    raise KeyError(
                        f"Monitor metric '{self.monitor_metric}' not found in validation metrics for split '{self.val_loader_names[idx]}'."
                    )

                best_so_far = self.best_monitor_values[idx]
                improved = (
                    (self.monitor_mode == "min" and mon_val < best_so_far)
                    or (self.monitor_mode == "max" and mon_val > best_so_far)
                )
                if improved:
                    self.best_monitor_values[idx] = mon_val
                    # Store the entire metrics dict for this split
                    self.best_metrics_per_loader[idx] = metrics_i.copy()
                    # Save a checkpoint for this split
                    self._save_split_checkpoint(idx)

           
            log_dict = {}
            # Include all training metrics as-is
            log_dict.update(train_metrics)
            # Prefix each split's validation metrics with its name
            for idx, metrics_i in enumerate(all_val_metrics):
                split_name = self.val_loader_names[idx]
                for k, v in metrics_i.items():
                    log_dict[f"{split_name}_{k}"] = v
            
            # Update learning rate scheduler based on split 0's monitor metric
            first_mon = all_val_metrics[0][self.monitor_metric]
            self.update_scheduler(first_mon)

            
            torch_lr = self.optimizer.param_groups[0]["lr"]
            log_dict["learning_rate"] = torch_lr
    

            # Print a concise epoch summary
            monitor_strs = []
            for idx in range(len(self.val_loaders)):
                name = self.val_loader_names[idx]
                val_m = all_val_metrics[idx][self.monitor_metric]
                monitor_strs.append(f"{name}_{self.monitor_metric}={val_m:.4f}")
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"train_loss={train_metrics.get('train_loss', float('nan')):.4f}, "
                + ", ".join(monitor_strs)
            )

        # ======== AFTER ALL EPOCHS: SAVE BEST-METRICS JSON PER SPLIT ========
        for idx, loader_name in enumerate(self.val_loader_names):
            out_dict = {
                "split": loader_name,
                "best_monitor_metric": self.best_monitor_values[idx],
                "best_metrics": self.best_metrics_per_loader[idx],
            }
            json_path = self.checkpoint_dir / f"{self.experiment_name}_{loader_name}_best_metrics.json"
            with open(json_path, "w") as f:
                json.dump(out_dict, f, indent=4)
            self.logger.info(f"Wrote best-metrics JSON for '{loader_name}' -> {json_path}")

        # Return a summary so the caller can inspect best values/metrics
        return {
            "best_monitor_values": {
                self.val_loader_names[i]: self.best_monitor_values[i]
                for i in range(len(self.val_loaders))
            },
            "best_metrics_per_split": {
                self.val_loader_names[i]: self.best_metrics_per_loader[i]
                for i in range(len(self.val_loaders))
            },
        }
        
 
