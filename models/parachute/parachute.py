"""PaRaChute model definitions and adapters."""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "./") 

from collections import OrderedDict
from models.parachute.fusion import fusion_layer
from models.parachute.dce import DynamicContextualEmbedding

class HistoAdapter(nn.Module):
    """Adapter MLP for histopathology feature projection."""
    def __init__(self, input_dim, inter_dim, token_dim):
        """
        Initialize the histopathology adapter.

        :param input_dim: Input feature dimension.
        :type input_dim: int
        :param inter_dim: Intermediate hidden dimension.
        :type inter_dim: int
        :param token_dim: Output token dimension.
        :type token_dim: int
        :return: None.
        :rtype: None
        """
        super().__init__()
        self.fc_in = nn.Linear(input_dim, inter_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(inter_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, inter_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(inter_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, inter_dim),
        )
        self.block3 = nn.Sequential(
            nn.LayerNorm(inter_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, token_dim),
        )
        self.block4 = nn.Sequential(
            nn.LayerNorm(token_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim, token_dim),
        )
        self.final_norm = nn.LayerNorm(token_dim, eps=1e-5)

    def forward(self, x):
        """
        Project histopathology features to token space.

        :param x: Input features of shape ``[B, input_dim]``.
        :type x: torch.Tensor
        :return: Adapted features of shape ``[B, token_dim]``.
        :rtype: torch.Tensor
        """
        x = self.fc_in(x)
        x = x + self.block1(x)  # Residual connection
        x = x + self.block2(x)

        x = self.block3(x)  # Projection to token_dim
        x = x + self.block4(x)  # Residual connection
        x = self.final_norm(x)
        return x


class PaRaChuteModel(nn.Module): 
    """PaRaChute multimodal fusion model for survival prediction."""

    def __init__(
        self,
        rad_input_dim=1024,
        histo_input_dim=512,
        inter_dim=512,
        token_dim=256,
        dim_hider=256,  # For the attention fusion
        num_classes=3,
    ):
        """
        Initialize the PaRaChute model.

        :param rad_input_dim: Radiology input feature dimension.
        :type rad_input_dim: int
        :param histo_input_dim: Histopathology input feature dimension.
        :type histo_input_dim: int
        :param inter_dim: Intermediate hidden dimension.
        :type inter_dim: int
        :param token_dim: Token dimension used for fusion.
        :type token_dim: int
        :param dim_hider: Hidden dimension for the hazard head.
        :type dim_hider: int
        :param num_classes: Number of output classes (kept for compatibility).
        :type num_classes: int
        :return: None.
        :rtype: None
        """
        super().__init__()
        self.rad_input_dim = rad_input_dim
        self.histo_input_dim = histo_input_dim
        self.inter_dim = inter_dim
        self.token_dim = token_dim
        self.num_classes = num_classes
        self.dim_hider = dim_hider

        self.rad_adapter = nn.Sequential(
            nn.Conv1d(
                in_channels=self.rad_input_dim,
                out_channels=self.inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.Conv1d(
                in_channels=self.inter_dim,
                out_channels=self.inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.AdaptiveAvgPool1d(output_size=1),  #
            nn.Flatten(start_dim=1),
        )

        self.histo_adapter = HistoAdapter(
            self.histo_input_dim, self.inter_dim, self.inter_dim
        )


        self.missing_rad_token = nn.Parameter(
            torch.randn(
                1,
                self.inter_dim,
            ),
            requires_grad=True,
        )
        self.missing_histo_token = nn.Parameter(
            torch.randn(
                1,
                self.inter_dim,
            ),
            requires_grad=True,
        )

        self.missing_rad_token_fusion = nn.Parameter(
            torch.randn(
                1,
                self.token_dim,
            ),
            requires_grad=True,
        )
        self.missing_histo_token_fusion = nn.Parameter(
            torch.randn(
                1,
                self.token_dim,
            ),
            requires_grad=True,
        )

        self.token_adapt_rad = nn.Sequential(
            nn.Conv1d(
                in_channels=self.rad_input_dim,
                out_channels=self.inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.Conv1d(
                in_channels=self.inter_dim,
                out_channels=self.inter_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(self.inter_dim),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.inter_dim, self.inter_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.inter_dim, self.token_dim),
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        self.token_adapt_histo = HistoAdapter(
            self.histo_input_dim, self.inter_dim, self.token_dim
        )


        self.token_adapt_rad_pe = nn.Sequential(
            nn.Linear(self.inter_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        self.token_adapt_histo_pe = nn.Sequential(
            nn.Linear(self.inter_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim),
            nn.LayerNorm(self.token_dim, eps=1e-5),
        )

        self.dpe = DynamicContextualEmbedding(
            in_channels=self.token_dim, out_channels=self.token_dim
        )

        self.fusion = fusion_layer(
            d_model=self.token_dim, dim_hider=self.dim_hider, nhead=4, dropout=0.25
        )

        self.hazard_net = nn.Sequential(
            nn.Linear(self.token_dim, self.dim_hider),  # First hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout for regularization
            nn.Linear(self.dim_hider, self.token_dim),  # Second hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, 1),  # Output layer
        )

        self.norm_pe = nn.LayerNorm(self.token_dim, eps=1e-5)
        self.norm_att = nn.LayerNorm(self.token_dim, eps=1e-5)
        # self.act = nn.Sigmoid() #
        # self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False) #
        # self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False) #
        self.gamma = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(
        self, rad_feature, histo_feature, modality_flag=None, output_layers=["hazard"]
    ):
        """
        Run a forward pass through the model.

        :param rad_feature: Radiology features of shape ``[B, S, D_r]``.
        :type rad_feature: torch.Tensor
        :param histo_feature: Histopathology features of shape ``[B, D_h]``.
        :type histo_feature: torch.Tensor
        :param modality_flag: Binary modality mask of shape ``[B, 2]``.
        :type modality_flag: torch.Tensor or None
        :param output_layers: Output keys to return from intermediate layers.
        :type output_layers: list
        :return: Ordered dictionary of requested outputs.
        :rtype: collections.OrderedDict
        """
        outputs = OrderedDict()
        # if self._add_output_and_check("features",torch.cat([rad_feature,histo_feature], dim = 1), outputs, output_layers):
        #    return outputs

        rad_mask = modality_flag[:, 0].bool().to(rad_feature.device)
        histo_mask = modality_flag[:, 1].bool().to(histo_feature.device)
        batch_size = rad_mask.shape[0]

        # Adapt rad_features to (,512) (adapt only the available modality)
        adapted_rad = self.rad_adapter(
            rad_feature[rad_mask].permute(0, 2, 1)
        )  # (.,512)

        # Adapt histo_features to (,512) (only the available modality)
        adapted_histo = self.histo_adapter(histo_feature[histo_mask])  # (.,512)
        # Return the concatenated adapted features if chosen

        f_adapted_rad = torch.empty(batch_size, adapted_rad.shape[1]).to(
            self.missing_rad_token.device
        )

        f_adapted_histo = torch.empty(
            batch_size,
            adapted_histo.shape[1],
        ).to(self.missing_rad_token.device)

        f_adapted_rad[rad_mask] = adapted_rad
        f_adapted_histo[histo_mask] = adapted_histo

        f_adapted_rad[~rad_mask] = self.missing_rad_token.repeat((~rad_mask).sum(), 1)
        f_adapted_histo[~histo_mask] = self.missing_histo_token.repeat(
            (~histo_mask).sum(), 1
        )

        modality_flags = ~torch.min(rad_mask, histo_mask)

        # Return the concatenated adapted features if chosen
        if self._add_output_and_check(
            "adapted_features",
            torch.cat([f_adapted_rad, f_adapted_histo], dim=1),
            outputs,
            output_layers,
        ):
            return outputs
        if self._add_output_and_check(
            "adapted_histo", f_adapted_histo, outputs, output_layers
        ):
            return outputs
        if self._add_output_and_check(
            "adapted_rad", f_adapted_rad, outputs, output_layers
        ):
            return outputs

        # Calculate Positional Embedding
        pe = self.dpe(
            self.token_adapt_rad_pe(f_adapted_rad),  # (B,64)
            self.token_adapt_histo_pe(f_adapted_histo),  # (B,64)
            rad_mask,
            histo_mask,
        )  # (B, 64)
        # Return the positional embeddings if chosen
        if self._add_output_and_check(
            "positional_embeddings", pe, outputs, output_layers
        ):
            return outputs

        # Adapt for tokenization and inject missing tokens for missing modalities
        rad_tokens_pre = self.token_adapt_rad(
            rad_feature[rad_mask].permute(0, 2, 1)
        )  # (, 64)
        histo_tokens_pre = self.token_adapt_histo(histo_feature[histo_mask])  # (,64)
        rad_tokens = torch.empty(
            batch_size,
            rad_tokens_pre.shape[1],
        ).to(self.missing_rad_token.device)

        histo_tokens = torch.empty(
            batch_size,
            histo_tokens_pre.shape[1],
        ).to(self.missing_histo_token.device)

        rad_tokens[rad_mask] = rad_tokens_pre
        histo_tokens[histo_mask] = histo_tokens_pre

        rad_tokens[~rad_mask] = self.missing_rad_token_fusion.repeat(
            (~rad_mask).sum(), 1
        )
        histo_tokens[~histo_mask] = self.missing_histo_token_fusion.repeat(
            (~histo_mask).sum(), 1
        )

        # Attention-based fusion
        f_att = self.fusion(
            rad_tokens,
            histo_tokens,
            pe.sigmoid(),
        )

        # Skip connection
        pe_norm = self.norm_pe(pe)
        f_att_norm = self.norm_att(f_att)
        out = f_att_norm + pe_norm

        # Return the fused features if chosen
        if self._add_output_and_check("fused_features", out, outputs, output_layers):
            return outputs

        out = self.hazard_net(out)

        # hazard = self.act(out) #
        # outputs["hazard"] = self.output_range * hazard + self.output_shift #
        # outputs["hazard"] = 3.0 * torch.tanh(out) #
        outputs["hazard"] = out
        return outputs

    def _add_output_and_check(self, name, x, outputs, output_layers):
        """
        Add an output if requested and check if all outputs are collected.

        :param name: Output name.
        :type name: str
        :param x: Output tensor to add.
        :type x: torch.Tensor
        :param outputs: OrderedDict collecting outputs.
        :type outputs: collections.OrderedDict
        :param output_layers: List of requested output keys.
        :type output_layers: list
        :return: True if all requested outputs are present.
        :rtype: bool
        """
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

