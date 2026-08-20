from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_PATH = Path("data/processed/shot_manifest.csv")
OUTPUT_DIR = Path("data/processed")

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """Find a column using a list of possible names."""

    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find column for '{description}'.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def split_asset(
    df: pd.DataFrame,
    shot_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Split one asset by SHOTID."""

    if not np.isclose(
        train_ratio + val_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "Train/validation/test ratios must sum to 1.0."
        )

    df = df.copy()

    shot_ids = df[shot_column].unique()

    rng.shuffle(shot_ids)

    n_shots = len(shot_ids)

    n_train = int(n_shots * train_ratio)
    n_val = int(n_shots * val_ratio)

    train_shots = shot_ids[:n_train]

    val_shots = shot_ids[
        n_train:n_train + n_val
    ]

    test_shots = shot_ids[
        n_train + n_val:
    ]

    df["split"] = "test"

    df.loc[
        df[shot_column].isin(train_shots),
        "split",
    ] = "train"

    df.loc[
        df[shot_column].isin(val_shots),
        "split",
    ] = "val"

    return df


def check_leakage(
    df: pd.DataFrame,
    shot_column: str,
) -> None:
    """Check that a shot never appears in multiple splits."""

    print("\n" + "=" * 80)
    print("LEAKAGE CHECK")
    print("=" * 80)

    duplicated = (
        df.groupby(
            ["asset", shot_column]
        )["split"]
        .nunique()
    )

    leaked = duplicated[duplicated > 1]

    if len(leaked) == 0:
        print(
            "PASS: No SHOTID appears in multiple splits."
        )
    else:
        print(
            "FAIL: Shot leakage detected!"
        )
        print(leaked)


def print_summary(
    df: pd.DataFrame,
    shot_column: str,
    traces_column: str,
    labeled_column: str,
    coverage_column: str,
) -> None:
    """Print train/validation/test statistics."""

    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)

    summary = (
        df.groupby(
            ["asset", "split"]
        )
        .agg(
            shots=(shot_column, "count"),
            traces=(traces_column, "sum"),
            labeled_traces=(
                labeled_column,
                "sum",
            ),
            mean_coverage=(
                coverage_column,
                "mean",
            ),
        )
        .reset_index()
    )

    summary["mean_coverage"] *= 100

    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_coverage": "{:.2f}%".format,
            },
        )
    )

    print("\nOverall split distribution:")

    overall = (
        df.groupby("split")
        .agg(
            shots=(shot_column, "count"),
            traces=(traces_column, "sum"),
            labeled_traces=(
                labeled_column,
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        overall.to_string(index=False)
    )


def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    print(
        f"\nInput manifest: {MANIFEST_PATH}"
    )
    print(f"Rows: {len(df):,}")

    print("\nManifest columns:")
    for column in df.columns:
        print(f"  - {column}")

    # ------------------------------------------------------------------
    # Detect actual column names from the manifest.
    # ------------------------------------------------------------------

    asset_column = find_column(
        df,
        [
            "asset",
            "dataset",
            "name",
        ],
        "asset",
    )

    shot_column = find_column(
        df,
        [
            "shot_id",
            "SHOTID",
            "shotid",
            "shot",
        ],
        "shot ID",
    )

    traces_column = find_column(
        df,
        [
            "traces",
            "trace_count",
            "n_traces",
            "num_traces",
            "total_traces",
        ],
        "number of traces",
    )

    labeled_column = find_column(
        df,
        [
            "labeled_traces",
            "labelled_traces",
            "labeled",
            "labelled",
            "n_labeled",
            "n_labelled",
            "labeled_count",
            "labelled_count",
        ],
        "number of labeled traces",
    )

    coverage_column = find_column(
        df,
        [
            "label_coverage",
            "coverage",
            "labelled_coverage",
            "labeled_coverage",
        ],
        "label coverage",
    )

    print("\nDetected columns:")
    print(f"  asset:           {asset_column}")
    print(f"  shot ID:         {shot_column}")
    print(f"  traces:          {traces_column}")
    print(f"  labeled traces:  {labeled_column}")
    print(f"  coverage:        {coverage_column}")

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    split_parts = []

    for asset in sorted(
        df[asset_column].unique()
    ):
        asset_df = df[
            df[asset_column] == asset
        ].copy()

        print("\n" + "=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        print(
            f"Total shots: "
            f"{len(asset_df)}"
        )

        split_df = split_asset(
            asset_df,
            shot_column=shot_column,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            rng=rng,
        )

        for split_name in [
            "train",
            "val",
            "test",
        ]:
            count = (
                split_df["split"]
                == split_name
            ).sum()

            print(
                f"{split_name:>5}: "
                f"{count} shots"
            )

        split_parts.append(
            split_df
        )

    result = pd.concat(
        split_parts,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            asset_column,
            "split",
            shot_column,
        ]
    ).reset_index(drop=True)

    check_leakage(
        result,
        shot_column,
    )

    print_summary(
        result,
        shot_column=shot_column,
        traces_column=traces_column,
        labeled_column=labeled_column,
        coverage_column=coverage_column,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "shot_manifest_split.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("SPLIT MANIFEST CREATED")
    print("=" * 80)

    print(
        f"Output: {output_path}"
    )

    print(
        f"Rows:   {len(result):,}"
    )


if __name__ == "__main__":
    main()