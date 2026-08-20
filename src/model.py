from __future__ import annotations

import torch
import torch.nn as nn


class FirstBreakGRU(nn.Module):
    """
    GRU-based first-break picking model.

    Input:
        [batch, samples]

    Output:
        [batch, samples]

    For every seismic sample, the model predicts the probability
    that the first break has occurred at that sample.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_norm = nn.LayerNorm(1)

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_output_size = (
            hidden_size * 2
            if bidirectional
            else hidden_size
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Seismic waveform.
            Shape: [batch, samples]

        Returns
        -------
        logits:
            Per-sample first-break logits.
            Shape: [batch, samples]
        """

        if x.ndim != 2:
            raise ValueError(
                f"Expected input shape [batch, samples], "
                f"got {tuple(x.shape)}"
            )

        # [B, T] -> [B, T, 1]
        x = x.unsqueeze(-1)

        # Normalize each sample dimension.
        x = self.input_norm(x)

        # GRU
        features, _ = self.gru(x)

        # Per-sample classification
        logits = self.classifier(features)

        # [B, T, 1] -> [B, T]
        logits = logits.squeeze(-1)

        return logits


def build_model(
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = True,
) -> FirstBreakGRU:
    """
    Build the first-break GRU model.
    """

    return FirstBreakGRU(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )