from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


RAW_DIR = Path("data/raw")
FIGURE_DIR = Path("data/processed/figures")

ASSETS = {
    "Brunswick": "Brunswick_orig_1500ms_V2.hdf5",
    "Halfmile": "Halfmile3D_add_geom_sorted.hdf5",
    "Lalor": "Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
    "Sudbury": "preprocessed_Sudbury3D.hdf",
}


def get_shot_indices(
    shot_id: np.ndarray,
    target_shot: int,
) -> np.ndarray:
    """Return indices belonging to one SHOTID."""
    indices = np.flatnonzero(shot_id == target_shot)

    if len(indices) == 0:
        raise ValueError(f"SHOTID={target_shot} was not found.")

    return indices


def normalize_image(data: np.ndarray) -> np.ndarray:
    """Normalize seismic amplitudes for visualization."""
    data = data.astype(np.float32)

    scale = np.percentile(np.abs(data), 99)

    if scale == 0:
        return data

    data = data / scale
    data = np.clip(data, -1.0, 1.0)

    return data


def get_sampling_interval_ms(
    group: h5py.Group,
) -> float:
    """
    Return sampling interval in milliseconds.

    SAMP_RATE is stored in microseconds per sample.
    Example:
        2000 us -> 2 ms
        1000 us -> 1 ms
    """
    samp_rate = np.asarray(
        group["SAMP_RATE"][:]
    ).reshape(-1)

    if len(samp_rate) == 0:
        raise ValueError("SAMP_RATE is empty.")

    sampling_interval_ms = float(samp_rate[0]) / 1000.0

    if sampling_interval_ms <= 0:
        raise ValueError(
            f"Invalid sampling interval: "
            f"{sampling_interval_ms} ms"
        )

    return sampling_interval_ms


def plot_shot(
    asset_name: str,
    h5_path: Path,
    shot_id: int,
) -> None:
    """Plot one seismic shot with manual first-break picks."""

    with h5py.File(h5_path, "r") as h5:
        group = h5["TRACE_DATA/DEFAULT"]

        # Read SHOTID metadata.
        shot_ids = group["SHOTID"][:].reshape(-1)

        # Find all traces belonging to this shot.
        indices = get_shot_indices(
            shot_ids,
            shot_id,
        )

        start = indices[0]
        end = indices[-1] + 1

        # SHOTID blocks were confirmed to be contiguous
        # during geometry exploration.
        data = group["data_array"][start:end]

        picks = group["SPARE1"][start:end].reshape(-1)

        # Sampling interval in milliseconds.
        sampling_interval_ms = get_sampling_interval_ms(group)

    print(
        f"{asset_name}: "
        f"SHOTID={shot_id}, "
        f"traces={data.shape[0]}, "
        f"samples={data.shape[1]}, "
        f"dt={sampling_interval_ms:.3f} ms"
    )

    # Normalize only for visualization.
    data = normalize_image(data)

    # ---------------------------------------------------------
    # Convert first-break time from milliseconds
    # to sample indices.
    #
    # Example:
    #   pick = 638 ms
    #   dt   = 2 ms/sample
    #   sample index = 638 / 2 = 319
    # ---------------------------------------------------------

    valid = picks > 0

    pick_samples = np.full(
        len(picks),
        np.nan,
        dtype=np.float32,
    )

    pick_samples[valid] = (
        picks[valid] / sampling_interval_ms
    )

    # ---------------------------------------------------------
    # Plot seismic image.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.imshow(
        data.T,
        cmap="gray",
        aspect="auto",
        origin="upper",
    )

    # Trace index.
    x = np.arange(len(picks))

    # Manual first-break line.
    if np.any(valid):
        ax.plot(
            x[valid],
            pick_samples[valid],
            color="red",
            linewidth=1.5,
            label="Manual first break",
        )

    # ---------------------------------------------------------
    # Plot formatting.
    # ---------------------------------------------------------

    labeled_count = int(np.sum(valid))
    unlabeled_count = len(picks) - labeled_count

    ax.set_title(
        f"{asset_name} | "
        f"SHOTID={shot_id} | "
        f"{data.shape[0]} traces | "
        f"Labeled={labeled_count}"
    )

    ax.set_xlabel("Trace index")
    ax.set_ylabel("Sample index")

    if np.any(valid):
        ax.legend()

    ax.grid(
        False
    )

    fig.tight_layout()

    # ---------------------------------------------------------
    # Save figure.
    # ---------------------------------------------------------

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIR
        / f"{asset_name.lower()}_shot_{shot_id}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"  Labeled traces:   {labeled_count:,}"
    )

    print(
        f"  Unlabeled traces: {unlabeled_count:,}"
    )

    print(
        f"  Saved: {output_path}"
    )


def get_example_shots(
    h5_path: Path,
    n: int = 3,
) -> list[int]:
    """Get first n SHOTIDs from an asset."""

    with h5py.File(h5_path, "r") as h5:
        group = h5["TRACE_DATA/DEFAULT"]

        shot_ids = group["SHOTID"][:].reshape(-1)

        unique_shots = np.unique(shot_ids)

    return unique_shots[:n].tolist()


def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - SHOT VISUALIZATION")
    print("=" * 80)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for asset_name, filename in ASSETS.items():

        path = RAW_DIR / filename

        print("\n" + "=" * 80)
        print(f"ASSET: {asset_name}")
        print("=" * 80)

        if not path.exists():
            print(f"ERROR: File not found: {path}")
            continue

        example_shots = get_example_shots(
            path,
            n=3,
        )

        print(
            f"Example SHOTIDs: {example_shots}"
        )

        for shot_id in example_shots:
            plot_shot(
                asset_name=asset_name,
                h5_path=path,
                shot_id=shot_id,
            )

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()