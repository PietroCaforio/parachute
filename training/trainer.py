"""Base trainer implementation for model training loops."""

import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional
from typing import List

import torch
from torch.utils.data import DataLoader



sys.path.insert(0, "./")

from torch.utils.data import DataLoader  # noqa E402

from .metrics import (
    per_class_accuracy,
    precision_per_class,
    recall_per_class,
    f1_per_class,
    per_class_accuracy_binary,
    precision_per_class_binary,
    recall_per_class_binary,
    f1_per_class_binary,
)


class BaseTrainer:
    """Flexible base trainer with configurable training and validation loops."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: torch.device,
        experiment_name: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping: Optional[Dict] = None,
        metric_functions: Optional[Dict[str, Callable]] = None,
        n_validations: int = None,
    ):
        """
        Initialize the base trainer.

        :param model: Model to train.
        :type model: torch.nn.Module
        :param train_loader: Training data loader.
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: Validation data loader.
        :type val_loader: torch.utils.data.DataLoader
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
        :param metric_functions: Optional metric functions mapping.
        :type metric_functions: Dict[str, Callable] or None
        :param n_validations: Number of validation runs to average.
        :type n_validations: int or None
        :return: None.
        :rtype: None
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.experiment_name = experiment_name.replace("/", "_")
        self.scheduler = scheduler
        self.n_validations = n_validations
        # Early stopping configuration
        self.early_stopping_config = early_stopping or {}
        self.early_stopping_counter = 0
        self.best_monitor_metric = (
            float("inf")
            if self.early_stopping_config.get("mode") == "min"
            else float("-inf")
        )

        # Metric functions for tracking
        self.metric_functions = metric_functions or {}

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_metrics = {}
        

        # Setup logging and checkpoints
        self.setup_logging()

    def setup_logging(self):
        """
        Initialize logging and create the checkpoint directory.

        :return: None.
        :rtype: None
        """
        self.checkpoint_dir = Path(
            self.config["training"]["checkpoint_dir"] + self.experiment_name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    self.checkpoint_dir / f"{self.experiment_name}.log"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def compute_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor, phase: str
    ) -> Dict[str, float]:
        """
        Compute all registered metrics for a phase.

        :param outputs: Model outputs.
        :type outputs: torch.Tensor
        :param targets: Ground-truth targets.
        :type targets: torch.Tensor
        :param phase: Phase name prefix (e.g., ``"train"`` or ``"val"``).
        :type phase: str
        :return: Dictionary of metric values.
        :rtype: Dict[str, float]
        """
        metrics = {}
        for metric_name, metric_fn in self.metric_functions.items():
            try:
                metric_value = metric_fn(outputs, targets)
                metrics[f"{phase}_{metric_name}"] = metric_value
            except Exception as e:
                self.logger.warning(f"Failed to compute {metric_name}: {str(e)}")
        return metrics

    def check_early_stopping(self, monitor_value: float) -> bool:
        """
        Check whether early stopping criteria are met.

        :param monitor_value: Current monitored metric value.
        :type monitor_value: float
        :return: True if training should stop early.
        :rtype: bool
        """
        if not self.early_stopping_config:
            return False

        patience = self.early_stopping_config.get("patience", 0)
        mode = self.early_stopping_config.get("mode", "min")
        min_delta = self.early_stopping_config.get("min_delta", 0.0)

        improved = (
            mode == "min" and monitor_value < self.best_monitor_metric - min_delta
        ) or (mode == "max" and monitor_value > self.best_monitor_metric + min_delta)

        if improved:
            self.best_monitor_metric = monitor_value
            self.early_stopping_counter = 0
            return False

        self.early_stopping_counter += 1
        if self.early_stopping_counter >= patience:
            self.logger.info(
                f"Early stopping triggered after {patience} epochs without improvement"
            )
            return True
        return False

    def update_scheduler(self, monitor_value: Optional[float] = None):
        """
        Update the learning-rate scheduler.

        :param monitor_value: Metric value for schedulers like ReduceLROnPlateau.
        :type monitor_value: float or None
        :return: None.
        :rtype: None
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(monitor_value)
        else:
            self.scheduler.step()

    def save_checkpoint(self, is_best: bool = False, val_loss=True):
        """
        Save a training checkpoint to disk.

        :param is_best: Whether the checkpoint is the best by monitored metric.
        :type is_best: bool
        :param val_loss: If True, tag the best checkpoint by validation loss.
        :type val_loss: bool
        :return: None.
        :rtype: None
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_val_loss": self.best_val_loss,
            "best_monitor_metric": self.best_monitor_metric,
            "config": self.config,
            "training_metrics": self.training_metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"
        torch.save(checkpoint, latest_path)

        if is_best and val_loss == False:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model checkpoint to {best_path}")
        elif is_best:
            best_path = (
                self.checkpoint_dir / f"{self.experiment_name}_best_val_loss.pth"
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Load a training checkpoint from disk.

        :param checkpoint_path: Optional explicit checkpoint path.
        :type checkpoint_path: str or None
        :return: True if a checkpoint was loaded.
        :rtype: bool
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"

        if not Path(checkpoint_path).exists():
            self.logger.info(f"No checkpoint found at {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_monitor_metric = checkpoint.get(
            "best_monitor_metric", self.best_monitor_metric
        )
        self.training_metrics = checkpoint.get("training_metrics", {})

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return True

    def train_epoch(self):
        """
        Train for one epoch (to be implemented by subclasses).

        :return: Training metrics for the epoch.
        :rtype: dict
        """
        self.model.train()
        epoch_metrics = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Implement in child class
            raise NotImplementedError

        return epoch_metrics

    def validate(self):
        """
        Validate the model (to be implemented by subclasses).

        :return: Validation metrics for the epoch.
        :rtype: dict
        """
        self.model.eval()
        epoch_metrics = {}

        with torch.no_grad():
            for batch in self.val_loader:
                # Implement in child class
                raise NotImplementedError

        return epoch_metrics

    def train(self):
        """
        Run the main training loop.

        :return: Summary of best metrics.
        :rtype: dict
        """

        for epoch in range(self.current_epoch, self.config["training"]["num_epochs"]):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self.train_epoch()
            if train_metrics is None:
                self.logger.info("Detected NaN values, training stopped.")
                break
            # Validation phase
            if self.n_validations == None:
                val_metrics = self.validate()
            else:
                val_metrics_list = []
                for i in range(self.n_validations):
                    val_metrics_list.append(self.validate())

                # Calculate the average for each metric
                val_metrics = {}
                for metric_name in val_metrics_list[0].keys():
                    val_metrics[metric_name] = sum(
                        d[metric_name] for d in val_metrics_list
                    ) / len(val_metrics_list)

            # Combine metrics and log
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_metrics[epoch] = epoch_metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_metrics["learning_rate"] = current_lr
          

            # Update learning rate scheduler
            monitor_metric = val_metrics.get(
                self.config["training"].get("monitor_metric", "val_loss")
            )
            val_loss = val_metrics.get("val_loss")
            self.update_scheduler(monitor_metric)
            # Save checkpoint
            is_best = (
                monitor_metric < self.best_monitor_metric
                if self.early_stopping_config.get("mode") == "min"
                else monitor_metric > self.best_monitor_metric
            )
            # Check early stopping
            if self.check_early_stopping(monitor_metric):
                break
            is_best_loss = val_loss < self.best_val_loss
            if is_best_loss:
                self.best_val_loss = val_loss
            self.save_checkpoint(is_best_loss)
            self.save_checkpoint(is_best, val_loss=False)

            self.logger.info(f"Best monitor metric: {self.best_monitor_metric}")
            self.logger.info(f"Best monitor metric: {self.best_monitor_metric}")

            # Log epoch summary
            self.logger.info(
                f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]} - Metrics: '
                + ", ".join([f"{k}: {v:.8f}" for k, v in epoch_metrics.items()])
            )
        return {
            self.config["training"]["monitor_metric"]: self.best_monitor_metric,
            "val_loss": self.best_val_loss,
        }
