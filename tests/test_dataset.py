from src.dataset import FirstBreakTraceDataset


def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - DATASET TEST")
    print("=" * 80)

    dataset = FirstBreakTraceDataset(
        split="train"
    )

    print(f"\nDataset size: {len(dataset):,}")

    sample = dataset[0]

    print("\nFirst sample:")

    print(
        f"  waveform shape: "
        f"{sample['waveform'].shape}"
    )

    print(
        f"  waveform dtype: "
        f"{sample['waveform'].dtype}"
    )

    print(
        f"  first break: "
        f"{sample['first_break'].item():.2f} samples"
    )

    print(
        f"  asset: "
        f"{sample['asset']}"
    )

    print(
        f"  shot_id: "
        f"{sample['shot_id']}"
    )

    print(
        f"  trace_index: "
        f"{sample['trace_index']}"
    )

    waveform = sample["waveform"]

    print("\nWaveform statistics:")

    print(
        f"  min:  {waveform.min().item():.6f}"
    )

    print(
        f"  max:  {waveform.max().item():.6f}"
    )

    print(
        f"  mean: {waveform.mean().item():.6f}"
    )

    print(
        f"  std:  {waveform.std().item():.6f}"
    )

    print("\nDataset test completed successfully.")


if __name__ == "__main__":
    main()