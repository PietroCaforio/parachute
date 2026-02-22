"""Loss functions for survival training."""

import torch
from torch import nn
from torch.nn.modules.loss import _WeightedLoss


class CoxLoss(_WeightedLoss):
    """Cox proportional hazards loss."""
    def forward(
        self,
        hazard_pred: torch.Tensor,
        survtime: torch.Tensor,
        censor: torch.Tensor,
    ):
        """
        Compute Cox partial likelihood loss.

        :param hazard_pred: Predicted hazard scores.
        :type hazard_pred: torch.Tensor
        :param survtime: Survival/censoring times.
        :type survtime: torch.Tensor
        :param censor: Censor indicator (1 if event observed, 0 if censored).
        :type censor: torch.Tensor
        :return: Loss scalar.
        :rtype: torch.Tensor
        """
        censor = censor.float()
        current_batch_len = len(survtime)
        # modified for speed
        R_mat = survtime.reshape((1, current_batch_len)) >= survtime.reshape(
            (current_batch_len, 1)
        )
        # epsilon = 1e-7 # To prevent log(0)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean(
            (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor
        )
        return loss_cox
