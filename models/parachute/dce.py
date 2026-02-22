"""Dynamic contextual embedding module for cross-modal fusion."""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "./") 
import torch.nn.functional as F


class DynamicContextualEmbedding(nn.Module):
    """Compute dynamic contextual embeddings from cross-modal correlations."""
    def __init__(self, in_channels, out_channels, topk_pos=16, topk_neg=16):
        """
        Initialize the embedding module.

        :param in_channels: Input feature dimension.
        :type in_channels: int
        :param out_channels: Output embedding dimension.
        :type out_channels: int
        :param topk_pos: Number of top positive correlations to average.
        :type topk_pos: int
        :param topk_neg: Number of top negative correlations to average.
        :type topk_neg: int
        :return: None.
        :rtype: None
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels * 2, out_channels))
        # Learnable token for missing modality
        self.missing_modality_token = nn.Parameter(torch.zeros(1, out_channels))
        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

    def forward(self, f_rad, f_histo, rad_mask, histo_mask):
        """
        Compute dynamic positional embeddings from CT and WSI features.

        :param f_rad: Radiology features of shape ``[B, D]``.
        :type f_rad: torch.Tensor
        :param f_histo: Histopathology features of shape ``[B, D]``.
        :type f_histo: torch.Tensor
        :param rad_mask: Radiology availability mask of shape ``[B]``.
        :type rad_mask: torch.Tensor
        :param histo_mask: Histopathology availability mask of shape ``[B]``.
        :type histo_mask: torch.Tensor
        :return: Positional embeddings of shape ``[B, out_channels]``.
        :rtype: torch.Tensor
        """

        pred_pos, pred_neg = self._correlation(f_rad, f_histo)
        # concatenate maps
        class_layers = torch.cat([pred_pos, pred_neg], dim=1)

        modality_flags = ~torch.min(
            rad_mask, histo_mask
        )  # 1 when sample contains missing modality

        # Flatten input for MLP
        pos_input_flat = class_layers.view(class_layers.size(0), -1)
        pos_embedding = self.mlp(pos_input_flat)
        # Adjust positional embedding with the missing modality token
        adjusted_embedding = (
            pos_embedding + self.missing_modality_token * modality_flags.unsqueeze(1)
        )

        # Reshape back to spatial dimensions
        return adjusted_embedding

    # correlation operation
    def _correlation(self, f_rad, f_histo):
        """
        Compute positive and negative correlation maps between modalities.

        :param f_rad: Radiology features of shape ``[B, D]``.
        :type f_rad: torch.Tensor
        :param f_histo: Histopathology features of shape ``[B, D]``.
        :type f_histo: torch.Tensor
        :return: Tuple of (positive_map, negative_map).
        :rtype: tuple
        """

        # first normalize train and test features to have L2 norm 1
        # cosine similarity and reshape last two dimensions into one

        # klv - mnw is the pair of coordinates of the volume sections
        # on which the similarity is calculated

        sim = torch.einsum(
            "bi,bj->bij",
            F.normalize(f_rad, p=2, dim=1),
            F.normalize(f_histo, p=2, dim=1),
        )

        # TODO: Patho fusion

        sim = sim.unsqueeze(1)
        # sim_resh = sim.view(sim.shape[0], sim.shape[1], sim.shape[2]*sim.shape[3])
        # print(sim_resh.shape)

        pos_map = torch.mean(torch.topk(sim, self.topk_pos, dim=-1).values, dim=-1)
        neg_map = torch.mean(torch.topk(-sim, self.topk_neg, dim=-1).values, dim=-1)
        return pos_map, neg_map
