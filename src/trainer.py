from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.loss import FirstBreakLoss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: FirstBreakLoss,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        waveform = batch["waveform"].to(
            device,
            non_blocking=True,
        )

        first_break = batch["first_break"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(waveform)

        loss = loss_fn(
            logits,
            first_break,
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

        optimizer.step()

        batch_size = waveform.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: FirstBreakLoss,
    device: torch.device,
) -> float:
    """Evaluate model on validation data."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        waveform = batch["waveform"].to(
            device,
            non_blocking=True,
        )

        first_break = batch["first_break"].to(
            device,
            non_blocking=True,
        )

        logits = model(waveform)

        loss = loss_fn(
            logits,
            first_break,
        )

        batch_size = waveform.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    path: str | Path,
) -> None:
    """Save model checkpoint."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    device: torch.device,
) -> dict:
    """Load model checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint