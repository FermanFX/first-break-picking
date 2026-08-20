from pathlib import Path

import h5py
import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "shot_manifest.csv"


ASSETS = {
    "Brunswick": "Brunswick_orig_1500ms_V2.hdf5",
    "Halfmile": "Halfmile3D_add_geom_sorted.hdf5",
    "Lalor": "Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
    "Sudbury": "preprocessed_Sudbury3D.hdf",
}


# Minimum fraction of traces that must have a manual label
# for a shot to be considered sufficiently labeled.
MIN_LABEL_COVERAGE = 0.50


def analyze_asset(
    asset_name: str,
    filename: str,
) -> list[dict]:
    """Build metadata records for all shots in one asset."""

    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    records = []

    print("\n" + "=" * 80)
    print(f"ASSET: {asset_name}")
    print("=" * 80)

    with h5py.File(path, "r") as h5:
        group = h5["TRACE_DATA/DEFAULT"]

        shot_ids = group["SHOTID"][:].reshape(-1)
        picks = group["SPARE1"][:].reshape(-1)

        source_x = group["SOURCE_X"][:].reshape(-1)
        source_y = group["SOURCE_Y"][:].reshape(-1)

        unique_shots, counts = np.unique(
            shot_ids,
            return_counts=True,
        )

        print(
            f"Total traces: {len(shot_ids):,}"
        )

        print(
            f"Total shots: {len(unique_shots):,}"
        )

        for shot_id, trace_count in zip(
            unique_shots,
            counts,
        ):
            # Find shot traces.
            indices = np.flatnonzero(
                shot_ids == shot_id
            )

            shot_picks = picks[indices]

            valid = shot_picks > 0

            labeled_count = int(
                np.sum(valid)
            )

            unlabeled_count = (
                int(trace_count)
                - labeled_count
            )

            label_coverage = (
                labeled_count / trace_count
            )

            # First-break statistics.
            if labeled_count > 0:
                valid_picks = shot_picks[valid]

                pick_min = float(
                    np.min(valid_picks)
                )

                pick_max = float(
                    np.max(valid_picks)
                )

                pick_mean = float(
                    np.mean(valid_picks)
                )

                pick_median = float(
                    np.median(valid_picks)
                )

            else:
                pick_min = np.nan
                pick_max = np.nan
                pick_mean = np.nan
                pick_median = np.nan

            # Source position is constant within a shot.
            source_x_value = float(
                source_x[indices[0]]
            )

            source_y_value = float(
                source_y[indices[0]]
            )

            # Determine whether the shot is usable.
            usable = (
                labeled_count > 0
                and label_coverage
                >= MIN_LABEL_COVERAGE
            )

            records.append(
                {
                    "asset": asset_name,
                    "filename": filename,
                    "shot_id": int(shot_id),
                    "trace_start": int(indices[0]),
                    "trace_end": int(indices[-1] + 1),
                    "n_traces": int(trace_count),
                    "n_labeled": labeled_count,
                    "n_unlabeled": unlabeled_count,
                    "label_coverage": label_coverage,
                    "first_break_min_ms": pick_min,
                    "first_break_max_ms": pick_max,
                    "first_break_mean_ms": pick_mean,
                    "first_break_median_ms": pick_median,
                    "source_x": source_x_value,
                    "source_y": source_y_value,
                    "usable": usable,
                }
            )

    asset_df = pd.DataFrame(records)

    usable_count = int(
        asset_df["usable"].sum()
    )

    print(
        f"Usable shots: "
        f"{usable_count:,} / {len(asset_df):,}"
    )

    print(
        f"Usable percentage: "
        f"{100 * usable_count / len(asset_df):.2f}%"
    )

    return records


def print_summary(df: pd.DataFrame) -> None:
    """Print manifest summary."""

    print("\n" + "=" * 80)
    print("MANIFEST SUMMARY")
    print("=" * 80)

    summary = (
        df.groupby("asset")
        .agg(
            shots=("shot_id", "count"),
            usable_shots=("usable", "sum"),
            traces=("n_traces", "sum"),
            labeled_traces=("n_labeled", "sum"),
            mean_coverage=("label_coverage", "mean"),
        )
        .reset_index()
    )

    summary["usable_percentage"] = (
        summary["usable_shots"]
        / summary["shots"]
        * 100
    )

    summary["mean_coverage"] *= 100

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nLabel coverage distribution:")

    print(
        df["label_coverage"]
        .describe()
        .to_string()
    )


def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - BUILD SHOT MANIFEST")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records = []

    for asset_name, filename in ASSETS.items():

        records = analyze_asset(
            asset_name,
            filename,
        )

        all_records.extend(records)

    df = pd.DataFrame(all_records)

    # Sort for reproducibility.
    df = df.sort_values(
        ["asset", "shot_id"]
    ).reset_index(drop=True)

    # Save manifest.
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(df)

    print("\n" + "=" * 80)
    print("MANIFEST CREATED")
    print("=" * 80)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows:   {len(df):,}")


if __name__ == "__main__":
    main()