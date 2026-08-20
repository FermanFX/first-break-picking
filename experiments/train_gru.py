from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm

from src.data_loader import create_dataloader
from src.loss import FirstBreakLoss
from src.model import FirstBreakGRU


# =============================================================================
# CONFIGURATION
# =============================================================================

MANIFEST_PATH = Path(
    "data/processed/shot_manifest_split.csv"
)

CHECKPOINT_DIR = Path(
    "data/processed/checkpoints"
)

RESULT_DIR = Path(
    "data/processed/training"
)

SEED = 42

# Train only one asset for the first experiment.
ASSET = "Brunswick"

# Limit the number of traces.
MAX_TRAIN_SAMPLES = 20_000
MAX_VAL_SAMPLES = 5_000

BATCH_SIZE = 64
EPOCHS = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# SUBSET CREATION
# =============================================================================

def limit_dataset(
    dataset,
    max_samples: int,
    seed: int,
):
    """
    Randomly select at most max_samples from a dataset.

    The selection is reproducible because a fixed seed is used.
    """

    if max_samples >= len(dataset):
        return dataset

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:max_samples].tolist()

    return Subset(
        dataset,
        indices,
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train model for one epoch."""

    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch in progress:

        waveform = batch["waveform"].to(
            device,
            non_blocking=True,
        )

        first_break = batch["first_break"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(waveform)

        loss = loss_fn(
            logits,
            first_break,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        current_batch_size = waveform.shape[0]

        running_loss += (
            loss.item() * current_batch_size
        )

        total_samples += current_batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / total_samples


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate validation loss."""

    model.eval()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress:

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

        current_batch_size = waveform.shape[0]

        running_loss += (
            loss.item() * current_batch_size
        )

        total_samples += current_batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / total_samples


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    path: Path,
) -> None:
    """Save model checkpoint."""

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
            "asset": ASSET,
        },
        path,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("FIRST BREAK PICKING - GRU TRAINING")
    print("=" * 80)

    print()
    print(f"Device:              {DEVICE}")
    print(f"Asset:               {ASSET}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Epochs:              {EPOCHS}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"Max train samples:   {MAX_TRAIN_SAMPLES:,}")
    print(f"Max validation:      {MAX_VAL_SAMPLES:,}")
    print()

    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # DATA
    # =========================================================================

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    train_loader_full = create_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        assets=[ASSET],
    )

    val_loader_full = create_dataloader(
        manifest_path=MANIFEST_PATH,
        split="val",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        assets=[ASSET],
    )

    print()
    print(
        f"Full {ASSET} train samples: "
        f"{len(train_loader_full.dataset):,}"
    )

    print(
        f"Full {ASSET} validation samples: "
        f"{len(val_loader_full.dataset):,}"
    )

    # -------------------------------------------------------------------------
    # LIMIT DATASET
    # -------------------------------------------------------------------------

    train_dataset = limit_dataset(
        train_loader_full.dataset,
        MAX_TRAIN_SAMPLES,
        seed=SEED,
    )

    val_dataset = limit_dataset(
        val_loader_full.dataset,
        MAX_VAL_SAMPLES,
        seed=SEED + 1,
    )

    # Re-create DataLoaders with limited datasets.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_loader_full.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_loader_full.collate_fn,
    )

    print()
    print(
        f"Training samples used: "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"Validation samples used: "
        f"{len(val_loader.dataset):,}"
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader):,}"
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    print()
    print("=" * 80)
    print("MODEL")
    print("=" * 80)

    model = FirstBreakGRU().to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(model)

    print()
    print(
        f"Total parameters:     {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    # =========================================================================
    # LOSS
    # =========================================================================

    loss_fn = FirstBreakLoss()

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # =========================================================================
    # TRAINING
    # =========================================================================

    best_val_loss = float("inf")

    history = []

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=DEVICE,
        )

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        val_loss = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=DEVICE,
        )

        # ---------------------------------------------------------------------
        # SCHEDULER
        # ---------------------------------------------------------------------

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ---------------------------------------------------------------------
        # HISTORY
        # ---------------------------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
        )

        print()
        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Val loss:   {val_loss:.6f}"
        )

        print(
            f"LR:         {current_lr:.2e}"
        )

        # ---------------------------------------------------------------------
        # LAST CHECKPOINT
        # ---------------------------------------------------------------------

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            path=CHECKPOINT_DIR / "brunswick_last.pt",
        )

        # ---------------------------------------------------------------------
        # BEST CHECKPOINT
        # ---------------------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                path=CHECKPOINT_DIR / "brunswick_best.pt",
            )

            print(
                "✓ New best model saved."
            )

    # =========================================================================
    # SAVE HISTORY
    # =========================================================================

    history_path = (
        RESULT_DIR / "brunswick_training_history.csv"
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        history_path,
        index=False,
    )

    # =========================================================================
    # FINISHED
    # =========================================================================

    print()
    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    print(
        f"Asset:               {ASSET}"
    )

    print(
        f"Train samples:       {len(train_loader.dataset):,}"
    )

    print(
        f"Validation samples:  {len(val_loader.dataset):,}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best checkpoint:     "
        f"{CHECKPOINT_DIR / 'brunswick_best.pt'}"
    )

    print(
        f"Last checkpoint:     "
        f"{CHECKPOINT_DIR / 'brunswick_last.pt'}"
    )

    print(
        f"Training history:    "
        f"{history_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
