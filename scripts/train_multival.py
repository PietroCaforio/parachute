"""Train PaRaChute with multi-validation and missing-modality splits."""

import argparse
import json
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
from training.losses import CoxLoss
from training.survival_trainer import SurvivalTrainerMultival  


SEED = 0


def set_global_seed(seed=SEED):
    """
    Set global seeds for reproducibility.

    :param seed: Seed value to use.
    :type seed: int
    :return: None.
    :rtype: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train multimodal model with multiple validation splits"
    )
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
    Entry point for multival training.

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

    results = {"fold_results": []}


    for fold in range(config["n_folds"]):
        print(f"Running Cross-Validation - Fold {fold}")

        # Create a directory for this fold and copy config
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, fold_dir / f"config_fold_{fold}.json")

        # Modify config for this fold's checkpoint directory
        fold_config = config.copy()
        fold_config["training"]["checkpoint_dir"] = str(fold_dir) + "/"

        # -----------------------
        # Construct datasets
        # -----------------------

        train_dataset = MultimodalCTWSIDatasetSurv(
            fold=fold, split="train", **config["data_training"]
        )

        common_test_args = {
            "ct_path": config["data_training"]["ct_path"],
            "wsi_path": config["data_training"]["wsi_path"],
            "labels_splits_path": config["data_training"]["labels_splits_path"],
            "missing_modality_prob": config["data_training"]["missing_modality_prob"],
            "require_both_modalities": True,
            "pairing_mode": "one_to_one",
            "allow_repeats": True,
            "pairs_per_patient": None,
            "rad_dim": config["data_training"]["rad_dim"],
            "histo_dim": config["data_training"]["histo_dim"],
            "n_patches": config["data_training"]["n_patches"]
        }

        test_ct = MultimodalCTWSIDatasetSurv(
            fold=fold,
            split="test",
            missing_modality="ct",
            **common_test_args,
        )
        test_histo = MultimodalCTWSIDatasetSurv(
            fold=fold,
            split="test",
            missing_modality="wsi",
            **common_test_args,
        )
        test_mixed = MultimodalCTWSIDatasetSurv(
            fold=fold,
            split="test",
            missing_modality="both",
            **common_test_args,
        )


        # -----------------------
        # Create dataloaders
        # -----------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data_loader"]["num_workers"],
            pin_memory=True,
        )

        test_loaders = [
            DataLoader(
                test_ct,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["data_loader"]["num_workers"],
                pin_memory=True,
            ),
            DataLoader(
                test_histo,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["data_loader"]["num_workers"],
                pin_memory=True,
            ),
            DataLoader(
                test_mixed,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["data_loader"]["num_workers"],
                pin_memory=True,
            ),
        ]

        # -----------------------
        # Initialize model, loss, optimizer, scheduler
        # -----------------------

        model = PaRaChuteModel(
            rad_input_dim=config["model"]["rad_input_dim"],
            histo_input_dim=config["model"]["histo_input_dim"],
            inter_dim=config["model"]["inter_dim"],
            token_dim=config["model"]["token_dim"],
        )
        model.to(device)

        criterion = CoxLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=0.01,
        )

        if config["training"]["scheduler"] is None:
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
        else:
            scheduler = None

        # -----------------------
        # Instantiate the multi-validation trainer
        # -----------------------
        val_loader_names = [
            "ct_missing",
            "histo_missing",
            "mixed_missing",
        ]

        trainer = SurvivalTrainerMultival(
            model=model,
            train_loader=train_loader,
            val_loaders=test_loaders,
            criterion=criterion,
            optimizer=optimizer,
            config=fold_config,
            device=device,
            experiment_name=args.experiment_name + f"_fold_{fold}",
            scheduler=scheduler,
            early_stopping=None,  # Early stopping is disabled
            n_validations=config["training"]["n_validations"],
            val_loader_names=val_loader_names,
        )

        # Optionally resume from a checkpoint (applies to all splits simultaneously)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        # Run training. This will:
        #  - Train for `num_epochs`,
        #  - After each epoch, validate on each of the 9 splits,
        #  - Save a "best" checkpoint per split when its monitor metric improves,
        #  - At the end, write out a JSON per split containing its best metrics.
        best_info = trainer.train()

        # -----------------------
        # Collect fold-level results
        # -----------------------
        fold_entry = {
            "fold": fold,
            "num_train_samples": len(train_dataset),
            "best_monitor_values": best_info["best_monitor_values"],
            "best_metrics_per_split": best_info["best_metrics_per_split"],
        }
        results["fold_results"].append(fold_entry)

    # Sleep briefly to ensure all checkpoints have been flushed
    time.sleep(10)

    # Save the overall cross-validation results (per-fold, per-split) to a single JSON
    results_path = experiment_dir / "cv_results_multival.json"
    print(f"Saving CV results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
