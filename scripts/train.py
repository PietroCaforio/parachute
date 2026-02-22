"""Train PaRaChute with standard cross-validation."""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


sys.path.insert(0, "./")  
from data.multimodal import MultimodalCTWSIDatasetSurv 
from models.parachute.parachute import PaRaChuteModel
from training.survival_trainer import SurvivalTrainer  
from training.losses import CoxLoss

SEED = 0


def set_global_seed(seed=SEED):
    """
    Set global seeds for reproducibility.

    :param seed: Seed value to use.
    :type seed: int
    :return: None.
    :rtype: None
    """
    # Python's built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure PyTorch to make computations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Train multimodal model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load a JSON configuration file.

    :param config_path: Path to a JSON config file.
    :type config_path: str
    :return: Configuration dictionary.
    :rtype: dict
    """
    with open(config_path) as f:
        config = json.load(f)
    return config


def main():
    """
    Entry point for training.

    :return: None.
    :rtype: None
    """

    set_global_seed(seed=SEED)
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "gpu" in config["training"].keys():
        torch.cuda.set_device(config["training"]["gpu"])
        gpu_id = config["training"]["gpu"]
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    experiment_dir = Path(config["training"]["checkpoint_dir"]) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    results = {"fold_results": [], "mean_metrics": {}, "std_metrics": {}}

    for fold in range(config["n_folds"]):
        print(f"Running Cross-Validation - Fold {fold}")
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(args.config, fold_dir / f"config_fold_{fold}.json")
        fold_config = config.copy()
        fold_config["training"]["checkpoint_dir"] = str(fold_dir) + "/"

        train_dataset = MultimodalCTWSIDatasetSurv(
            fold=fold, split="train", **config["data_training"]
        )

        # Copy training config and remove 'n_patches' key if present for validation
        validation_args = config["data_training"].copy()
        validation_args.pop("n_patches", None)

        test_dataset = MultimodalCTWSIDatasetSurv(
            fold=fold, split="test", **validation_args
        )
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data_loader"]["num_workers"],
            pin_memory=True,
            # collate_fn=MultimodalCTWSIDataset.collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data_loader"]["num_workers"],
            pin_memory=True,
            # collate_fn=MultimodalCTWSIDataset.collate_fn
        )

        # Initialize model
        model = PaRaChuteModel(
            rad_input_dim=config["model"]["rad_input_dim"],
            histo_input_dim=config["model"]["histo_input_dim"],
            inter_dim=config["model"]["inter_dim"],
            token_dim=config["model"]["token_dim"],
        )
        model.to(device)
        # Initialize loss and optimizer
        criterion = CoxLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=0.01,
        )
        if config["training"]["scheduler"] == None:
            scheduler = None
        elif config["training"]["scheduler"]["type"] == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config["training"]["scheduler"]["mode"],
                factor=config["training"]["scheduler"]["factor"],
                patience=config["training"]["scheduler"]["patience"],
                min_lr=config["training"]["scheduler"]["min_lr"],
            )
        elif config["training"]["scheduler"]["type"] == "cosine_annealing_lr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["training"]["scheduler"]["T_max"],
                eta_min=config["training"]["scheduler"]["eta_min"],
                last_epoch=config["training"]["scheduler"]["last_epoch"],
            )
        elif (
            config["training"]["scheduler"]["type"] == "cosine_annealing_warm_restarts"
        ):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config["training"]["scheduler"]["T_0"],
                T_mult=config["training"]["scheduler"]["T_mult"],
                eta_min=config["training"]["scheduler"]["eta_min"],
                last_epoch=config["training"]["scheduler"]["last_epoch"],
            )

        trainer = SurvivalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            experiment_name=args.experiment_name + f"_fold_{fold}",
            scheduler=scheduler,
            early_stopping=config["training"]["early_stopping"],
            n_validations=config["training"]["n_validations"],
        )

        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        # Start training
        best_metrics = trainer.train()
        results["fold_results"].append(
            {
                "fold": fold,
                "num_train_samples": len(train_dataset),
                "num_test_samples": len(test_dataset),
                "best_metrics": best_metrics,
            }
        )

    time.sleep(10)

    if results["fold_results"]:
        # Aggregate results over folds
        metric_names = results["fold_results"][0]["best_metrics"].keys()
        for metric_name in metric_names:
            metric_values = [
                fold_results["best_metrics"][metric_name]
                for fold_results in results["fold_results"]
            ]
            results["mean_metrics"][metric_name] = np.mean(metric_values)
            results["std_metrics"][metric_name] = np.std(metric_values)

    # Save results
    results_path = experiment_dir / "cv_results.json"
    print("Saving results...")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
