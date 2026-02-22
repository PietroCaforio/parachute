"""Evaluate trained PaRaChute checkpoints."""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, "./")

from data.multimodal import MultimodalCTWSIDatasetSurv
from models.parachute.parachute import PaRaChuteModel
from training.losses import CoxLoss
from training.survival_trainer import SurvivalTrainer, SurvivalTrainerMultival


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
        description="Evaluate checkpoints on the test split (full or multival)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to an experiment directory, fold directory, or run directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "multival"],
        required=True,
        help="Evaluation mode: full or multival.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config JSON to use instead of the checkpoint-embedded config.",
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default="best",
        help="Which checkpoint to load (best, best_val_loss, latest).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the evaluation JSON.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./models/test_logs/",
        help=(
            "Base directory for test logs. Logs are written to "
            "<log-dir>/test_fold_X/."
        ),
    )
    return parser.parse_args()


def _load_config(config_path: str) -> Dict:
    """
    Load a JSON configuration file.

    :param config_path: Path to a JSON config file.
    :type config_path: str
    :return: Configuration dictionary.
    :rtype: Dict
    """
    with open(config_path) as f:
        return json.load(f)


def _normalize_log_dir(log_dir: Optional[str]) -> Optional[str]:
    """
    Normalize a log directory path to ensure a trailing slash.

    :param log_dir: Log directory path or None.
    :type log_dir: str or None
    :return: Normalized log directory path.
    :rtype: str or None
    """
    if log_dir is None:
        return None
    log_dir = str(Path(log_dir))
    return log_dir if log_dir.endswith("/") else log_dir + "/"


def _override_checkpoint_dir(config: Dict, log_dir: Optional[str]) -> Dict:
    """
    Override ``training.checkpoint_dir`` in a config for logging outputs.

    :param config: Original configuration dictionary.
    :type config: Dict
    :param log_dir: Log directory to set, or None to leave unchanged.
    :type log_dir: str or None
    :return: Updated configuration dictionary.
    :rtype: Dict
    """
    if not log_dir:
        return config
    updated = dict(config)
    updated["training"] = dict(config["training"])
    updated["training"]["checkpoint_dir"] = _normalize_log_dir(log_dir)
    return updated


def _infer_fold_idx(path: Path) -> int:
    """
    Infer fold index from a path containing ``fold_<k>``.

    :param path: Path to parse.
    :type path: pathlib.Path
    :return: Fold index (defaults to 0 if not found).
    :rtype: int
    """
    match = re.search(r"fold_(\d+)", str(path))
    return int(match.group(1)) if match else 0


def _select_checkpoint_by_patterns(root: Path, patterns: List[str]) -> Optional[Path]:
    """
    Select the first checkpoint that matches any of the glob patterns.

    :param root: Root directory to search.
    :type root: pathlib.Path
    :param patterns: List of glob patterns.
    :type patterns: List[str]
    :return: Path to the first matching checkpoint, if any.
    :rtype: pathlib.Path or None
    """
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _collect_fold_dirs(root: Path) -> List[Path]:
    """
    Collect fold directories from a checkpoint root.

    :param root: Root experiment or fold directory.
    :type root: pathlib.Path
    :return: List of fold directories or a singleton list with ``root``.
    :rtype: List[pathlib.Path]
    """
    fold_dirs = sorted([p for p in root.glob("fold_*") if p.is_dir()])
    if fold_dirs:
        return fold_dirs

    run_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and "_fold_" in p.name]
    )
    if run_dirs:
        return run_dirs

    return [root]


def _get_device(config: Dict) -> torch.device:
    """
    Resolve the torch device from config settings.

    :param config: Configuration dictionary.
    :type config: Dict
    :return: Torch device to use.
    :rtype: torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "gpu" in config["training"].keys():
        torch.cuda.set_device(config["training"]["gpu"])
        gpu_id = config["training"]["gpu"]
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return device


def _build_model(config: Dict) -> PaRaChuteModel:
    """
    Build a PaRaChute model instance from configuration.

    :param config: Configuration dictionary.
    :type config: Dict
    :return: Initialized model.
    :rtype: PaRaChuteModel
    """
    return PaRaChuteModel(
        rad_input_dim=config["model"]["rad_input_dim"],
        histo_input_dim=config["model"]["histo_input_dim"],
        inter_dim=config["model"]["inter_dim"],
        token_dim=config["model"]["token_dim"],
    )


def _resolve_data_training(config: Dict) -> Dict:
    """
    Resolve training data section from config.

    :param config: Configuration dictionary.
    :type config: Dict
    :return: Training data configuration.
    :rtype: Dict
    :raises KeyError: If no data section is present.
    """
    if "data_training" in config:
        return config["data_training"]
    if "data" in config:
        return config["data"]
    raise KeyError("Config must contain 'data_training' or legacy 'data' section.")


def _resolve_data_splits(config: Dict) -> Tuple[Dict, Dict]:
    """
    Resolve training and validation data sections from config.

    :param config: Configuration dictionary.
    :type config: Dict
    :return: Tuple of (data_training, data_validation) configs.
    :rtype: Tuple[Dict, Dict]
    """
    data_training = _resolve_data_training(config)
    data_validation = data_training.copy()
    data_validation.update(config.get("data_validation") or {})
    return data_training, data_validation


def _get_n_validations(config: Dict) -> Optional[int]:
    """
    Return the number of validation repetitions, if configured.

    :param config: Configuration dictionary.
    :type config: Dict
    :return: Number of validation repetitions or None.
    :rtype: int or None
    """
    return config.get("training", {}).get("n_validations")


def _build_full_loaders(config: Dict, fold: int):
    """
    Build train and test loaders for full evaluation.

    :param config: Configuration dictionary.
    :type config: Dict
    :param fold: Fold index.
    :type fold: int
    :return: Tuple of datasets and loaders.
    :rtype: tuple
    """
    data_training, data_validation = _resolve_data_splits(config)
    train_dataset = MultimodalCTWSIDatasetSurv(
        fold=fold, split="train", **data_training
    )

    validation_args = data_validation.copy()
    validation_args.pop("n_patches", None)
    test_dataset = MultimodalCTWSIDatasetSurv(
        fold=fold, split="test", **validation_args
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data_loader"]["num_workers"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data_loader"]["num_workers"],
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_loader, test_loader


def _build_multival_loaders(config: Dict, fold: int):
    """
    Build train and test loaders for multival evaluation.

    :param config: Configuration dictionary.
    :type config: Dict
    :param fold: Fold index.
    :type fold: int
    :return: Tuple of datasets, loaders, and split names.
    :rtype: tuple
    """
    data_training, data_validation = _resolve_data_splits(config)
    train_dataset = MultimodalCTWSIDatasetSurv(
        fold=fold, split="train", **data_training
    )

    common_test_args = {
        "ct_path": data_validation["ct_path"],
        "wsi_path": data_validation["wsi_path"],
        "labels_splits_path": data_validation["labels_splits_path"],
        "missing_modality_prob": data_validation.get("missing_modality_prob", 0.0),
        "require_both_modalities": True,
        "pairing_mode": "one_to_one",
        "allow_repeats": True,
        "pairs_per_patient": None,
    }
    if "rad_dim" in data_validation or "rad_dim" in data_training:
        common_test_args["rad_dim"] = data_validation.get(
            "rad_dim", data_training.get("rad_dim")
        )
    if "histo_dim" in data_validation or "histo_dim" in data_training:
        common_test_args["histo_dim"] = data_validation.get(
            "histo_dim", data_training.get("histo_dim")
        )
    if "n_patches" in data_validation or "n_patches" in data_training:
        common_test_args["n_patches"] = data_validation.get(
            "n_patches", data_training.get("n_patches")
        )

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

    val_loader_names = [
        "ct_missing",
        "histo_missing",
        "mixed_missing",
    ]

    return train_dataset, [test_ct, test_histo, test_mixed], train_loader, test_loaders, val_loader_names


def _average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average metrics across repeated evaluations.

    :param metrics_list: List of metric dictionaries.
    :type metrics_list: List[Dict[str, float]]
    :return: Averaged metrics.
    :rtype: Dict[str, float]
    """
    averaged = {}
    for key in metrics_list[0].keys():
        averaged[key] = sum(d[key] for d in metrics_list) / len(metrics_list)
    return averaged


def _evaluate_full_checkpoint(
    ckpt_path: Path,
    fold: int,
    checkpoint_tag: str,
    config_override: Optional[Dict] = None,
    log_dir: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
    """
    Evaluate a single checkpoint in full mode.

    :param ckpt_path: Path to checkpoint.
    :type ckpt_path: pathlib.Path
    :param fold: Fold index.
    :type fold: int
    :param checkpoint_tag: Checkpoint tag requested (best/latest/etc.).
    :type checkpoint_tag: str
    :param config_override: Optional config override dictionary.
    :type config_override: Dict or None
    :param log_dir: Optional log directory override.
    :type log_dir: str or None
    :return: Tuple of (metrics, counts, config).
    :rtype: Tuple[Dict[str, float], Dict[str, int], Dict]
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = config_override or ckpt.get("config")
    if config is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain a config and no --config was provided."
        )
    config = _override_checkpoint_dir(config, log_dir)

    device = _get_device(config)
    model = _build_model(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    criterion = CoxLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=0.01,
    )

    train_dataset, test_dataset, train_loader, test_loader = _build_full_loaders(
        config, fold
    )

    trainer = SurvivalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device,
        experiment_name=f"test_fold_{fold}",
        scheduler=None,
        early_stopping=None,
        n_validations=_get_n_validations(config),
    )

    n_validations = _get_n_validations(config)
    if n_validations is None:
        metrics = trainer.validate()
    else:
        metrics_list = []
        for _ in range(n_validations):
            metrics_list.append(trainer.validate())
        metrics = _average_metrics(metrics_list)

    counts = {
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
    }
    return metrics, counts, config


def _evaluate_multival_checkpoint(
    ckpt_path: Path,
    fold: int,
    split_name: str,
    config_override: Optional[Dict] = None,
    log_dir: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
    """
    Evaluate a single checkpoint in multival mode for one split.

    :param ckpt_path: Path to checkpoint.
    :type ckpt_path: pathlib.Path
    :param fold: Fold index.
    :type fold: int
    :param split_name: Split name (e.g., ``ct_missing``).
    :type split_name: str
    :param config_override: Optional config override dictionary.
    :type config_override: Dict or None
    :param log_dir: Optional log directory override.
    :type log_dir: str or None
    :return: Tuple of (metrics, counts, config).
    :rtype: Tuple[Dict[str, float], Dict[str, int], Dict]
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = config_override or ckpt.get("config")
    if config is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain a config and no --config was provided."
        )
    config = _override_checkpoint_dir(config, log_dir)

    device = _get_device(config)
    model = _build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    criterion = CoxLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=0.01,
    )

    train_dataset, test_datasets, train_loader, test_loaders, val_loader_names = (
        _build_multival_loaders(config, fold)
    )

    trainer = SurvivalTrainerMultival(
        model=model,
        train_loader=train_loader,
        val_loaders=test_loaders,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device,
        experiment_name=f"test_fold_{fold}",
        scheduler=None,
        early_stopping=None,
        n_validations=_get_n_validations(config),
        val_loader_names=val_loader_names,
    )

    split_idx = val_loader_names.index(split_name)
    loader = test_loaders[split_idx]

    n_validations = _get_n_validations(config)
    if n_validations is None:
        metrics = trainer.validate_loader(loader)
    else:
        metrics_list = []
        for _ in range(n_validations):
            metrics_list.append(trainer.validate_loader(loader))
        metrics = _average_metrics(metrics_list)

    counts = {
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_datasets[split_idx]),
    }
    return metrics, counts, config


def _select_full_checkpoint(fold_dir: Path, checkpoint_tag: str) -> Path:
    """
    Select a checkpoint in full mode given a tag.

    :param fold_dir: Fold directory to search.
    :type fold_dir: pathlib.Path
    :param checkpoint_tag: Tag indicating which checkpoint to use.
    :type checkpoint_tag: str
    :return: Path to selected checkpoint.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If no checkpoint is found.
    """
    if checkpoint_tag == "best":
        patterns = ["*_best.pth", "*_best_val_loss.pth", "*_latest.pth", "*.pth"]
    elif checkpoint_tag == "best_val_loss":
        patterns = ["*_best_val_loss.pth", "*_best.pth", "*_latest.pth", "*.pth"]
    elif checkpoint_tag == "latest":
        patterns = ["*_latest.pth", "*_best.pth", "*_best_val_loss.pth", "*.pth"]
    else:
        patterns = [checkpoint_tag]

    ckpt = _select_checkpoint_by_patterns(fold_dir, patterns)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under {fold_dir}")
    return ckpt


def _select_multival_checkpoints(fold_dir: Path, checkpoint_tag: str) -> Dict[str, Path]:
    """
    Select split-specific checkpoints in multival mode.

    :param fold_dir: Fold directory to search.
    :type fold_dir: pathlib.Path
    :param checkpoint_tag: Tag indicating which checkpoint to use.
    :type checkpoint_tag: str
    :return: Mapping from split name to checkpoint path.
    :rtype: Dict[str, pathlib.Path]
    :raises FileNotFoundError: If no checkpoint is found for a split.
    """
    split_names = ["ct_missing", "histo_missing", "mixed_missing"]
    checkpoints = {}

    for split in split_names:
        if checkpoint_tag == "best":
            patterns = [
                f"*_best_{split}.pth",
                "*_best.pth",
                "*_latest.pth",
                "*_best_val_loss.pth",
                "*.pth",
            ]
        elif checkpoint_tag == "best_val_loss":
            patterns = [
                "*_best_val_loss.pth",
                "*_best.pth",
                "*_latest.pth",
                "*.pth",
            ]
        elif checkpoint_tag == "latest":
            patterns = [
                "*_latest.pth",
                "*_best.pth",
                "*_best_val_loss.pth",
                "*.pth",
            ]
        else:
            patterns = [checkpoint_tag]

        ckpt = _select_checkpoint_by_patterns(fold_dir, patterns)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found for split {split} under {fold_dir}")
        checkpoints[split] = ckpt

    return checkpoints


def main():
    """
    Entry point for evaluation.

    :return: None.
    :rtype: None
    """
    set_global_seed(SEED)
    args = parse_args()

    checkpoint_root = Path(args.checkpoint_dir)
    fold_dirs = _collect_fold_dirs(checkpoint_root)
    config_override = _load_config(args.config) if args.config else None

    if args.mode == "full":
        results = {"fold_results": [], "mean_metrics": {}, "std_metrics": {}}
        for fold_dir in fold_dirs:
            fold_idx = _infer_fold_idx(fold_dir)
            ckpt_path = _select_full_checkpoint(fold_dir, args.checkpoint_tag)
            metrics, counts, _ = _evaluate_full_checkpoint(
                ckpt_path,
                fold_idx,
                args.checkpoint_tag,
                config_override=config_override,
                log_dir=args.log_dir,
            )
            results["fold_results"].append(
                {
                    "fold": fold_idx,
                    "checkpoint": str(ckpt_path),
                    **counts,
                    "metrics": metrics,
                }
            )

        if results["fold_results"]:
            metric_names = results["fold_results"][0]["metrics"].keys()
            for metric_name in metric_names:
                metric_values = [
                    fr["metrics"][metric_name] for fr in results["fold_results"]
                ]
                results["mean_metrics"][metric_name] = float(np.mean(metric_values))
                results["std_metrics"][metric_name] = float(np.std(metric_values))

        output_path = (
            Path(args.output)
            if args.output
            else (checkpoint_root / "eval_results.json")
        )
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved evaluation results to {output_path}")

    else:
        results = {"fold_results": [], "mean_metrics": {}, "std_metrics": {}}
        for fold_dir in fold_dirs:
            fold_idx = _infer_fold_idx(fold_dir)
            split_ckpts = _select_multival_checkpoints(fold_dir, args.checkpoint_tag)
            fold_entry = {
                "fold": fold_idx,
                "checkpoints": {},
                "num_train_samples": None,
                "num_test_samples": {},
                "metrics_per_split": {},
            }

            for split_name, ckpt_path in split_ckpts.items():
                metrics, counts, _ = _evaluate_multival_checkpoint(
                    ckpt_path,
                    fold_idx,
                    split_name,
                    config_override=config_override,
                    log_dir=args.log_dir,
                )
                fold_entry["checkpoints"][split_name] = str(ckpt_path)
                fold_entry["metrics_per_split"][split_name] = metrics
                fold_entry["num_test_samples"][split_name] = counts["num_test_samples"]
                fold_entry["num_train_samples"] = counts["num_train_samples"]

            results["fold_results"].append(fold_entry)

        if results["fold_results"]:
            metric_keys = results["fold_results"][0]["metrics_per_split"].keys()
            for split_name in metric_keys:
                split_metrics = results["fold_results"][0]["metrics_per_split"][
                    split_name
                ].keys()
                for metric_name in split_metrics:
                    key = f"{split_name}_{metric_name}"
                    metric_values = [
                        fr["metrics_per_split"][split_name][metric_name]
                        for fr in results["fold_results"]
                    ]
                    results["mean_metrics"][key] = float(np.mean(metric_values))
                    results["std_metrics"][key] = float(np.std(metric_values))

        output_path = (
            Path(args.output)
            if args.output
            else (checkpoint_root / "eval_results_multival.json")
        )
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
