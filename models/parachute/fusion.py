"""Attention-based fusion layers for cross-modal features."""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "./") 



class fusion_layer(nn.Module):
    """Multi-head attention fusion block for two modalities."""
    def __init__(self, d_model=64, dim_hider=256, nhead=2, dropout=0.1):
        """
        Initialize the fusion layer.

        :param d_model: Token dimension for attention.
        :type d_model: int
        :param dim_hider: Hidden dimension (reserved for compatibility).
        :type dim_hider: int
        :param nhead: Number of attention heads.
        :type nhead: int
        :param dropout: Dropout probability in attention.
        :type dropout: float
        :return: None.
        :rtype: None
        """
        super().__init__()
        self.cross_att1 = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.cross_att2 = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.cross_att3 = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )

    def forward(self, f1, f2, pos):
        """
        Fuse two modality token sequences with cross-attention.

        :param f1: First modality tokens of shape ``[B, D]`` or ``[T, B, D]``.
        :type f1: torch.Tensor
        :param f2: Second modality tokens of shape ``[B, D]`` or ``[T, B, D]``.
        :type f2: torch.Tensor
        :param pos: Positional encoding to add to ``f2``.
        :type pos: torch.Tensor
        :return: Fused token representation.
        :rtype: torch.Tensor
        """
        fv, fv_weights = self.cross_att1(f1, f2, f2 + pos)
        fk, fk_weights = self.cross_att1(f1, f2, f2)
        fq, fq_weights = self.cross_att2(f2, f1, f1)
        f22, f22_weights = self.cross_att3(fq, fk, fv)

        return f22
