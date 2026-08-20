from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "Brunswick": "Brunswick_orig_1500ms_V2.hdf5",
    "Halfmile": "Halfmile3D_add_geom_sorted.hdf5",
    "Lalor": "Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
    "Sudbury": "preprocessed_Sudbury3D.hdf",
}

REQUIRED_KEYS = [
    "data_array",
    "SHOTID",
    "SHOT_PEG",
    "SOURCE_X",
    "SOURCE_Y",
    "SOURCE_HT",
    "REC_PEG",
    "REC_X",
    "REC_Y",
    "REC_HT",
    "SAMP_RATE",
    "COORD_SCALE",
    "HT_SCALE",
    "SAMP_NUM",
    "SPARE1",
]

def print_dataset_structure(group: h5py.Group) -> None:
    """Print datasets available in the TRACE_DATA/DEFAULT group."""
    print("\n--- HDF5 datasets ---")
    for key in group.keys():
        dataset = group[key]
        print(
            f"{key:<15} "
            f"shape={str(dataset.shape):<20} "
            f"dtype={dataset.dtype}"
        )

def get_unique_preview(values: np.ndarray, limit: int = 10) -> np.ndarray:
    """Return a small preview of unique values."""
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return values
    return np.unique(values)[:limit]

def inspect_asset(name: str, filename: str) -> None:
    """Inspect one seismic asset."""
    path = RAW_DATA_DIR / filename
    print("\n" + "=" * 80)
    print(f"ASSET: {name}")
    print("=" * 80)
    print(f"File: {path}")
    if not path.exists():
        print("ERROR: File not found.")
        return
    print(
        f"File size: "
        f"{path.stat().st_size / (1024**3):.2f} GB"
    )
    try:
        with h5py.File(path, "r") as h5_file:
            print("\n--- Top-level structure ---")
            for key in h5_file.keys():
                print(f"  {key}")
            group_path = "TRACE_DATA/DEFAULT"
            if group_path not in h5_file:
                print(
                    f"\nERROR: '{group_path}' "
                    "does not exist."
                )
                return
            group = h5_file[group_path]
            print_dataset_structure(group)

            # Check expected keys
            missing = [
                key
                for key in REQUIRED_KEYS
                if key not in group
            ]
            print("\n--- Required keys ---")
            if missing:
                print("Missing:")
                for key in missing:
                    print(f"  - {key}")
            else:
                print("All expected keys are present.")
            # Trace data
            data = group["data_array"]
            n_traces, n_samples = data.shape
            print("\n--- Trace information ---")
            print(f"Number of traces:   {n_traces:,}")
            print(f"Samples per trace:  {n_samples:,}")
            print(f"Data shape:         {data.shape}")
            print(f"Data dtype:         {data.dtype}")
            # Sampling
            print("\n--- Sampling ---")
            if "SAMP_RATE" in group:
                samp_rate = np.asarray(
                    group["SAMP_RATE"][:]
                ).reshape(-1)
                print(
                    "SAMP_RATE preview:  ",
                    get_unique_preview(samp_rate),
                )
                print(
                    "SAMP_RATE min/max:  ",
                    samp_rate.min(),
                    samp_rate.max(),
                )
            if "SAMP_NUM" in group:
                samp_num = np.asarray(
                    group["SAMP_NUM"][:]
                ).reshape(-1)
                print(
                    "SAMP_NUM preview:   ",
                    get_unique_preview(samp_num),
                )
                print(
                    "SAMP_NUM min/max:   ",
                    samp_num.min(),
                    samp_num.max(),
                )
            # Coordinate scaling
            print("\n--- Coordinate scaling ---")
            if "COORD_SCALE" in group:
                coord_scale = np.asarray(
                    group["COORD_SCALE"][:]
                ).reshape(-1)
                print(
                    "COORD_SCALE unique: ",
                    get_unique_preview(coord_scale),
                )
            if "HT_SCALE" in group:
                ht_scale = np.asarray(
                    group["HT_SCALE"][:]
                ).reshape(-1)
                print(
                    "HT_SCALE unique:    ",
                    get_unique_preview(ht_scale),
                )
            # Receiver geometry
            print("\n--- Receiver geometry ---")
            for key in [
                "REC_PEG",
                "REC_X",
                "REC_Y",
                "REC_HT",
            ]:
                if key not in group:
                    continue
                values = np.asarray(
                    group[key][:]
                ).reshape(-1)
                print(
                    f"{key:<10} "
                    f"min={values.min():.3f} "
                    f"max={values.max():.3f} "
                    f"unique={np.unique(values).size:,}"
                )
            # Source / shot information
            print("\n--- Shot / source information ---")
            for key in [
                "SHOTID",
                "SHOT_PEG",
                "SOURCE_X",
                "SOURCE_Y",
                "SOURCE_HT",
            ]:
                if key not in group:
                    continue
                values = np.asarray(
                    group[key][:]
                ).reshape(-1)
                print(
                    f"{key:<10} "
                    f"min={values.min():.3f} "
                    f"max={values.max():.3f} "
                    f"unique={np.unique(values).size:,}"
                )
            # First break labels
            print("\n--- First-break labels ---")
            labels = np.asarray(
                group["SPARE1"][:]
            ).reshape(-1)
            labeled_mask = (
                (labels != 0)
                & (labels != -1)
            )
            labeled = labels[labeled_mask]
            n_labeled = int(labeled_mask.sum())
            n_unlabeled = int((~labeled_mask).sum())
            print(
                f"Total traces:       {labels.size:,}"
            )
            print(
                f"Labeled traces:     {n_labeled:,}"
            )
            print(
                f"Unlabeled traces:   {n_unlabeled:,}"
            )
            print(
                f"Labeled percentage: "
                f"{100 * n_labeled / labels.size:.2f}%"
            )

            if labeled.size > 0:
                print(
                    f"First-break min:    "
                    f"{labeled.min():.3f} ms"
                )
                print(
                    f"First-break max:    "
                    f"{labeled.max():.3f} ms"
                )
                print(
                    f"First-break mean:   "
                    f"{labeled.mean():.3f} ms"
                )
                print(
                    f"First-break median: "
                    f"{np.median(labeled):.3f} ms"
                )
            # Preview of first traces
            print("\n--- Preview ---")
            preview_count = min(5, n_traces)
            print(
                f"Reading first "
                f"{preview_count} traces..."
            )
            preview = np.asarray(
                data[:preview_count, :]
            )
            print(
                f"Preview shape:      "
                f"{preview.shape}"
            )
            print(
                f"Amplitude min:      "
                f"{preview.min():.6g}"
            )
            print(
                f"Amplitude max:      "
                f"{preview.max():.6g}"
            )
            print(
                f"Amplitude mean:     "
                f"{preview.mean():.6g}"
            )
            print(
                f"Amplitude std:      "
                f"{preview.std():.6g}"
            )
    except OSError as exc:
        print("\nERROR: Could not open HDF5 file.")
        print(exc)

def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - DATASET INSPECTION")
    print("=" * 80)
    for name, filename in DATASETS.items():
        inspect_asset(name, filename)
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()