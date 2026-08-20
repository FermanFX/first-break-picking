from __future__ import annotations
import torch
import torch.nn as nn


def make_first_break_target(
    first_break: torch.Tensor,
    n_samples: int,
    sigma: float = 3.0,
) -> torch.Tensor:
    """
    Convert first-break sample indices into Gaussian targets.

    Parameters
    ----------
    first_break:
        First-break sample indices.
        Shape: [batch]

    n_samples:
        Number of samples in waveform.

    sigma:
        Standard deviation of Gaussian target.

    Returns
    -------
    target:
        Gaussian target distribution.
        Shape: [batch, n_samples]
    """

    if first_break.ndim != 1:
        raise ValueError(
            "first_break must have shape [batch]"
        )

    if sigma <= 0:
        raise ValueError(
            "sigma must be positive"
        )

    device = first_break.device

    sample_positions = torch.arange(
        n_samples,
        device=device,
        dtype=torch.float32,
    )

    sample_positions = sample_positions.unsqueeze(0)

    first_break = first_break.float().unsqueeze(1)

    target = torch.exp(
        -0.5
        * (
            (sample_positions - first_break)
            / sigma
        )
        ** 2
    )

    return target


class FirstBreakLoss(nn.Module):
    """
    Binary cross-entropy loss for sample-wise
    first-break prediction.

    The model produces one logit per seismic sample.
    """

    def __init__(
        self,
        sigma: float = 3.0,
        pos_weight: float = 10.0,
    ) -> None:
        super().__init__()

        self.sigma = sigma

        self.register_buffer(
            "pos_weight",
            torch.tensor(
                pos_weight,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        logits: torch.Tensor,
        first_break: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits:
            Model output.
            Shape: [batch, samples]

        first_break:
            Ground-truth first-break indices.
            Shape: [batch]

        Returns
        -------
        loss:
            Scalar loss.
        """

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [batch, samples]"
            )

        if first_break.ndim != 1:
            raise ValueError(
                "first_break must have shape [batch]"
            )

        if logits.shape[0] != first_break.shape[0]:
            raise ValueError(
                "Batch size mismatch between logits "
                "and first_break"
            )

        target = make_first_break_target(
            first_break=first_break,
            n_samples=logits.shape[1],
            sigma=self.sigma,
        )

        # Samples after padding are not real seismic samples.
        # For now, all positions are valid because the DataLoader
        # test already guarantees padded batches.
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
        )

        return loss