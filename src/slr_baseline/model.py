from __future__ import annotations

import torch
from torch import nn


class SignBiLSTMBaseline(nn.Module):
    """Linear -> BiLSTM -> temporal pooling -> Linear classifier."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 55 * 4,
        proj_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 55, 4] or [B, T, input_dim]
        Returns:
            logits: [B, num_classes]
        """
        if x.ndim == 4:
            bsz, t, n_pts, c = x.shape
            x = x.view(bsz, t, n_pts * c)
        elif x.ndim != 3:
            raise ValueError(f"Expected x dims 3/4, got shape {tuple(x.shape)}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dim mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits
