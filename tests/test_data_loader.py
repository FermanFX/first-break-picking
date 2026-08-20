import torch

from src.data_loader import create_dataloader


MANIFEST_PATH = "data/processed/shot_manifest_split.csv"


def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - DATA LOADER TEST")
    print("=" * 80)

    # =========================================================
    # TRAIN
    # =========================================================

    train_loader = create_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    print("\nTRAIN")
    print("-" * 80)

    print(
        f"Dataset size: "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"Number of batches: "
        f"{len(train_loader):,}"
    )

    batch = next(iter(train_loader))

    waveforms = batch["waveform"]
    picks = batch["first_break"]

    print("\nFirst batch:")
    print(f"  waveform shape:    {waveforms.shape}")
    print(f"  waveform dtype:    {waveforms.dtype}")
    print(f"  first_break shape: {picks.shape}")
    print(f"  first_break dtype: {picks.dtype}")
    lengths = batch["lengths"]
    print("\nOriginal waveform lengths:")
    print(lengths.tolist())

    assert waveforms.ndim == 2
    assert picks.ndim == 1
    assert lengths.ndim == 1

    assert waveforms.shape[0] == 32
    assert picks.shape[0] == 32
    assert lengths.shape[0] == 32

    assert waveforms.dtype == torch.float32
    assert picks.dtype == torch.float32

    assert torch.isfinite(waveforms).all()
    assert torch.isfinite(picks).all()

    assert torch.all(picks >= 0)

    print("\nChecks:")
    print("  batch dimension: PASS")
    print("  padding:         PASS")
    print("  dtype:           PASS")
    print("  finite values:   PASS")
    print("  valid labels:    PASS")

    # =========================================================
    # STATISTICS
    # =========================================================

    print("\nBatch statistics:")

    print(
        f"  waveform min:  "
        f"{waveforms.min().item():.6f}"
    )

    print(
        f"  waveform max:  "
        f"{waveforms.max().item():.6f}"
    )

    print(
        f"  waveform mean: "
        f"{waveforms.mean().item():.6f}"
    )

    print(
        f"  waveform std:  "
        f"{waveforms.std().item():.6f}"
    )

    print(
        f"  pick min:      "
        f"{picks.min().item():.2f}"
    )

    print(
        f"  pick max:      "
        f"{picks.max().item():.2f}"
    )

    print(
        f"  pick mean:     "
        f"{picks.mean().item():.2f}"
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    val_loader = create_dataloader(
        manifest_path=MANIFEST_PATH,
        split="val",
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    print("\nVAL")
    print("-" * 80)

    print(
        f"Dataset size: "
        f"{len(val_loader.dataset):,}"
    )

    print(
        f"Number of batches: "
        f"{len(val_loader):,}"
    )

    # =========================================================
    # TEST
    # =========================================================

    test_loader = create_dataloader(
        manifest_path=MANIFEST_PATH,
        split="test",
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    print("\nTEST")
    print("-" * 80)

    print(
        f"Dataset size: "
        f"{len(test_loader.dataset):,}"
    )

    print(
        f"Number of batches: "
        f"{len(test_loader):,}"
    )

    # =========================================================

    print("\n" + "=" * 80)
    print("DATA LOADER TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()