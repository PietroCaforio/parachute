"""Metric functions for classification and survival tasks."""

from typing import Dict

import numpy as np
import torch
from lifelines.utils import concordance_index


def per_class_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute per-class accuracy for a 3-class classification task.

    :param outputs: Model outputs of shape ``[N, C]``.
    :type outputs: torch.Tensor
    :param targets: Ground-truth labels of shape ``[N]``.
    :type targets: torch.Tensor
    :return: Mapping with per-class and average accuracy.
    :rtype: Dict[str, float]
    """
    _, predicted = torch.max(outputs, 1)
    accuracies = {}
    for i in range(3):  # G1, G2, G3
        mask = targets == i
        if mask.sum() > 0:
            # correct per class i / number of sampler per class i
            acc = ((predicted == i) & mask).float().sum() / mask.float().sum()
            accuracies[f"G{i+1}_Acc"] = acc.item()
        else:
            accuracies[f"G{i+1}_Acc"] = 0.0
    accuracies["avg_Acc"] = sum(v for k, v in accuracies.items()) / 3.0
    return accuracies


def precision_per_class(
    outputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute per-class precision for a 3-class classification task.

    :param outputs: Model outputs of shape ``[N, C]``.
    :type outputs: torch.Tensor
    :param targets: Ground-truth labels of shape ``[N]``.
    :type targets: torch.Tensor
    :return: Mapping with per-class and average precision.
    :rtype: Dict[str, float]
    """
    num_classes = 3
    tp = torch.zeros(num_classes, device=targets.device)
    fp = torch.zeros(num_classes, device=targets.device)
    fn = torch.zeros(num_classes, device=targets.device)

    # Calculate TP, FP, FN
    _, outputs = torch.max(outputs, 1)
    for c in range(num_classes):
        tp[c] = ((targets == c) & (outputs == c)).sum().float()
        fp[c] = ((targets != c) & (outputs == c)).sum().float()
        fn[c] = ((targets == c) & (outputs != c)).sum().float()
    precision_per_class = tp / (tp + fp + 1e-8)

    return {
        "precisionG1": precision_per_class[0].item(),
        "precisionG2": precision_per_class[1].item(),
        "precisionG3": precision_per_class[2].item(),
        "avg_precision": sum(precision_per_class) / 3.0,
    }


def recall_per_class(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute per-class recall for a 3-class classification task.

    :param outputs: Model outputs of shape ``[N, C]``.
    :type outputs: torch.Tensor
    :param targets: Ground-truth labels of shape ``[N]``.
    :type targets: torch.Tensor
    :return: Mapping with per-class and average recall.
    :rtype: Dict[str, float]
    """
    num_classes = 3
    tp = torch.zeros(num_classes, device=targets.device)
    fp = torch.zeros(num_classes, device=targets.device)
    fn = torch.zeros(num_classes, device=targets.device)
    _, predicted = torch.max(outputs, 1)
    # Calculate TP, FP, FN
    for c in range(num_classes):
        tp[c] = ((targets == c) & (predicted == c)).sum().float()
        fp[c] = ((targets != c) & (predicted == c)).sum().float()
        fn[c] = ((targets == c) & (predicted != c)).sum().float()
    recall_per_class = tp / (tp + fn + 1e-8)
    return {
        "recallG1": recall_per_class[0].item(),
        "recallG2": recall_per_class[1].item(),
        "recallG3": recall_per_class[2].item(),
        "avg_recall": sum(recall_per_class) / 3.0,
    }


def f1_per_class(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute per-class F1 score for a 3-class classification task.

    :param outputs: Model outputs of shape ``[N, C]``.
    :type outputs: torch.Tensor
    :param targets: Ground-truth labels of shape ``[N]``.
    :type targets: torch.Tensor
    :return: Mapping with per-class and average F1 score.
    :rtype: Dict[str, float]
    """
    num_classes = 3
    _, predicted = torch.max(outputs, 1)
    tp = torch.zeros(num_classes, device=targets.device)
    fp = torch.zeros(num_classes, device=targets.device)
    fn = torch.zeros(num_classes, device=targets.device)

    # Calculate TP, FP, FN

    for c in range(num_classes):
        tp[c] = ((targets == c) & (predicted == c)).sum().float()
        fp[c] = ((targets != c) & (predicted == c)).sum().float()
        fn[c] = ((targets == c) & (predicted != c)).sum().float()

    # Calculate precision and recall directly
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "F1scoreG1": f1_per_class[0].item(),
        "F1scoreG2": f1_per_class[1].item(),
        "F1scoreG3": f1_per_class[2].item(),
        "avg_F1score": sum(f1_per_class) / 3.0,
    }


def per_class_accuracy_binary(
    outputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute per-class accuracy for a 2-class classification task.

    :param outputs: Model outputs of shape ``[N, C]``.
    :type outputs: torch.Tensor
    :param targets: Ground-truth labels of shape ``[N]``.
    :type targets: torch.Tensor
    :return: Mapping with per-class and average accuracy.
    :rtype: Dict[str, float]
    """
    _, predicted = torch.max(outputs, 1)
    accuracies = {}
    for i in range(2):  # G2, G3
        mask = targets == i
        if mask.sum() > 0:
            # correct per class i / number of sampler per class i
            acc = ((predicted == i) & mask).float().sum() / mask.float().sum()
            accuracies[f"G{i+2}_Acc"] = acc.item()
        else:
            accuracies[f"G{i+2}_Acc"] = 0.0
    accuracies["avg_Acc"] = sum(v for k, v in accuracies.items()) / 2.0
    return accuracies


def precision_per_class_binary(
    outputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute per-class precision for a 2-class classification task.

    :param outputs: Model outputs of shape ``[N, C]``.
    :type outputs: torch.Tensor
    :param targets: Ground-truth labels of shape ``[N]``.
    :type targets: torch.Tensor
    :return: Mapping with per-class and average precision.
    :rtype: Dict[str, float]
    """
    num_classes = 2
    tp = torch.zeros(num_classes, device=targets.device)
    fp = torch.zeros(num_classes, device=targets.device)
    fn = torch.zeros(num_classes, device=targets.device)

    # Calculate TP, FP, FN
    _, outputs = torch.max(outputs, 1)
    for c in range(num_classes):
        tp[c] = ((targets == c) & (outputs == c)).sum().float()
        fp[c] = ((targets != c) & (outputs == c)).sum().float()
        fn[c] = ((targets == c) & (outputs != c)).sum().float()
    precision_per_class = tp / (tp + fp + 1e-8)

    return {
        "precisionG2": precision_per_class[0].item(),
        "precisionG3": precision_per_class[1].item(),
        "avg_precision": sum(precision_per_class) / 2.0,
    }


def recall_per_class_binary(
    outputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute per-class recall for a 2-class classification task.

    :param outputs: Model outputs of shape ``[N, C]``.
    :type outputs: torch.Tensor
    :param targets: Ground-truth labels of shape ``[N]``.
    :type targets: torch.Tensor
    :return: Mapping with per-class and average recall.
    :rtype: Dict[str, float]
    """
    num_classes = 2
    tp = torch.zeros(num_classes, device=targets.device)
    fp = torch.zeros(num_classes, device=targets.device)
    fn = torch.zeros(num_classes, device=targets.device)
    _, predicted = torch.max(outputs, 1)
    # Calculate TP, FP, FN
    for c in range(num_classes):
        tp[c] = ((targets == c) & (predicted == c)).sum().float()
        fp[c] = ((targets != c) & (predicted == c)).sum().float()
        fn[c] = ((targets == c) & (predicted != c)).sum().float()
    recall_per_class = tp / (tp + fn + 1e-8)
    return {
        "recallG2": recall_per_class[0].item(),
        "recallG3": recall_per_class[1].item(),
        "avg_recall": sum(recall_per_class) / 2.0,
    }


def f1_per_class_binary(
    outputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute per-class F1 score for a 2-class classification task.

    :param outputs: Model outputs of shape ``[N, C]``.
    :type outputs: torch.Tensor
    :param targets: Ground-truth labels of shape ``[N]``.
    :type targets: torch.Tensor
    :return: Mapping with per-class and average F1 score.
    :rtype: Dict[str, float]
    """
    num_classes = 2
    _, predicted = torch.max(outputs, 1)
    tp = torch.zeros(num_classes, device=targets.device)
    fp = torch.zeros(num_classes, device=targets.device)
    fn = torch.zeros(num_classes, device=targets.device)

    # Calculate TP, FP, FN

    for c in range(num_classes):
        tp[c] = ((targets == c) & (predicted == c)).sum().float()
        fp[c] = ((targets != c) & (predicted == c)).sum().float()
        fn[c] = ((targets == c) & (predicted != c)).sum().float()

    # Calculate precision and recall directly
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "F1scoreG2": f1_per_class[0].item(),
        "F1scoreG3": f1_per_class[1].item(),
        "avg_F1score": sum(f1_per_class) / 2.0,
    }


def cindex(preds_hazard: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
    """
    Compute concordance index for survival predictions.

    :param preds_hazard: Predicted hazard scores.
    :type preds_hazard: torch.Tensor
    :param time: Event or censoring times.
    :type time: torch.Tensor
    :param event: Event indicator (1 if event observed, 0 if censored).
    :type event: torch.Tensor
    :return: Dictionary with ``cindex`` key.
    :rtype: Dict[str, float]
    """
    # preds_hazard = preds_hazard.cpu().detach().numpy()
    # time = time.cpu().detach().numpy()
    # event = event.cpu().detach().numpy()
    ci = concordance_index_torch(time, preds_hazard, event)
    return {"cindex": ci}


def concordance_index_torch(
    predicted_scores: torch.Tensor,
    event_times: torch.Tensor,
    event_observed: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute the concordance index (C-index) in a vectorized manner.

    :param predicted_scores: Predicted risk scores (higher means higher risk).
    :type predicted_scores: torch.Tensor
    :param event_times: Observed event or censoring times.
    :type event_times: torch.Tensor
    :param event_observed: Event indicator (1 if event occurred, 0 if censored).
    :type event_observed: torch.Tensor
    :param eps: Small value to avoid division by zero.
    :type eps: float
    :return: Concordance index in ``[0, 1]``.
    :rtype: float
    """
    # Ensure inputs are 1D tensors
    predicted_scores = predicted_scores.flatten()
    event_times = event_times.flatten()
    event_observed = event_observed.flatten()

    # Create pairwise matrices
    diff_event_times = event_times.unsqueeze(0) - event_times.unsqueeze(
        1
    )  # shape (N, N)
    diff_pred_scores = predicted_scores.unsqueeze(0) - predicted_scores.unsqueeze(
        1
    )  # shape (N, N)

    # Valid comparable pairs: i's event time < j's event time AND i is observed
    is_valid = (diff_event_times < 0) & (event_observed.unsqueeze(0) == 1)

    # Concordant pairs: if predicted score of i > predicted score of j
    concordant = (diff_pred_scores > 0) & is_valid

    # Tied predicted scores are counted as 0.5 concordant
    tied = (diff_pred_scores.abs() < eps) & is_valid

    # Count
    concordant_sum = concordant.sum(dtype=torch.float32)
    tied_sum = tied.sum(dtype=torch.float32)
    valid_pairs = is_valid.sum(dtype=torch.float32)

    c_index = (concordant_sum + 0.5 * tied_sum) / (valid_pairs + eps)

    return c_index.item()
