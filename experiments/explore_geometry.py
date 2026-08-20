from pathlib import Path
import h5py
import numpy as np

RAW_DIR = Path("data/raw")

ASSETS = {
    "Brunswick": "Brunswick_orig_1500ms_V2.hdf5",
    "Halfmile": "Halfmile3D_add_geom_sorted.hdf5",
    "Lalor": "Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
    "Sudbury": "preprocessed_Sudbury3D.hdf",
}

# Metadata-only exploration. We deliberately do NOT load data_array.
def print_array_stats(name: str, arr: np.ndarray) -> None:
    """Print basic statistics for a metadata array."""
    arr = np.asarray(arr).reshape(-1)
    unique = np.unique(arr)
    print(f"\n{name}")
    print(f"  shape:       {arr.shape}")
    print(f"  dtype:       {arr.dtype}")
    print(f"  min:         {arr.min()}")
    print(f"  max:         {arr.max()}")
    print(f"  unique:      {len(unique)}")
    if len(unique) <= 20:
        print(f"  values:      {unique}")

def inspect_shots(group: h5py.Group, max_shots: int = 10) -> None:
    """Inspect trace grouping by SHOTID."""
    shot_id = group["SHOTID"][:].reshape(-1)
    rec_peg = group["REC_PEG"][:].reshape(-1)
    unique_shots, counts = np.unique(shot_id, return_counts=True)

    print("\n--- Shot grouping ---")
    print(f"Number of unique SHOTID: {len(unique_shots)}")
    print(f"Traces per shot:")
    print(f"  min:    {counts.min()}")
    print(f"  max:    {counts.max()}")
    print(f"  mean:   {counts.mean():.2f}")
    print(f"  median: {np.median(counts):.2f}")
    print("\nFirst shots:")

    for shot, count in zip(unique_shots[:max_shots], counts[:max_shots]):
        indices = np.flatnonzero(shot_id == shot)
        receivers = rec_peg[indices]

        print(
            f"  SHOTID={shot:<10} "
            f"traces={count:<6} "
            f"REC_PEG unique={len(np.unique(receivers)):<5} "
            f"index_range=({indices.min()}, {indices.max()})"
        )

def inspect_trace_order(group: h5py.Group, n: int = 30) -> None:
    """Inspect ordering of the first traces."""
    keys = [
        "SHOTID",
        "SHOT_PEG",
        "REC_PEG",
        "REC_X",
        "REC_Y",
        "SOURCE_X",
        "SOURCE_Y",
        "SPARE1",
    ]
    print("\n--- First traces / ordering ---")
    arrays = {}
    for key in keys:
        if key in group:
            arrays[key] = group[key][:n].reshape(-1)
    header = (
        f"{'idx':>5} "
        f"{'SHOTID':>10} "
        f"{'REC_PEG':>10} "
        f"{'REC_X':>12} "
        f"{'REC_Y':>12} "
        f"{'SOURCE_X':>12} "
        f"{'SOURCE_Y':>12} "
        f"{'SPARE1':>10}"
    )
    print(header)
    print("-" * len(header))
    for i in range(n):
        print(
            f"{i:5d} "
            f"{arrays['SHOTID'][i]:10.0f} "
            f"{arrays['REC_PEG'][i]:10.0f} "
            f"{arrays['REC_X'][i]:12.0f} "
            f"{arrays['REC_Y'][i]:12.0f} "
            f"{arrays['SOURCE_X'][i]:12.0f} "
            f"{arrays['SOURCE_Y'][i]:12.0f} "
            f"{arrays['SPARE1'][i]:10.0f}"
        )

def inspect_receiver_geometry(group: h5py.Group) -> None:
    """Inspect receiver coordinate geometry."""
    rec_peg = group["REC_PEG"][:].reshape(-1)
    rec_x = group["REC_X"][:].reshape(-1)
    rec_y = group["REC_Y"][:].reshape(-1)
    print("\n--- Receiver geometry ---")
    print_array_stats("REC_PEG", rec_peg)
    print_array_stats("REC_X", rec_x)
    print_array_stats("REC_Y", rec_y)
    print("\nReceiver coordinate examples:")
    # Unique receiver combinations.
    receiver_table = np.column_stack(
        [
            rec_peg,
            rec_x,
            rec_y,
        ]
    )
    unique_receivers = np.unique(receiver_table, axis=0)
    print(
        f"Unique (REC_PEG, REC_X, REC_Y) combinations: "
        f"{len(unique_receivers)}"
    )
    for row in unique_receivers[:10]:
        print(
            f"  REC_PEG={row[0]:.0f}, "
            f"REC_X={row[1]:.0f}, "
            f"REC_Y={row[2]:.0f}"
        )

def inspect_source_geometry(group: h5py.Group) -> None:
    """Inspect source geometry."""
    shot_id = group["SHOTID"][:].reshape(-1)
    shot_peg = group["SHOT_PEG"][:].reshape(-1)
    source_x = group["SOURCE_X"][:].reshape(-1)
    source_y = group["SOURCE_Y"][:].reshape(-1)
    print("\n--- Source geometry ---")
    print_array_stats("SHOTID", shot_id)
    print_array_stats("SHOT_PEG", shot_peg)
    print_array_stats("SOURCE_X", source_x)
    print_array_stats("SOURCE_Y", source_y)
    print("\nFirst unique shots and source positions:")
    unique_shot_ids = np.unique(shot_id)
    for shot in unique_shot_ids[:10]:
        indices = np.flatnonzero(shot_id == shot)
        print(
            f"  SHOTID={shot:.0f} "
            f"SHOT_PEG={shot_peg[indices[0]]:.0f} "
            f"SOURCE_X={source_x[indices[0]]:.0f} "
            f"SOURCE_Y={source_y[indices[0]]:.0f} "
            f"traces={len(indices)}"
        )

def inspect_consecutive_shots(group: h5py.Group) -> None:
    """Check whether traces belonging to the same shot are contiguous."""
    shot_id = group["SHOTID"][:].reshape(-1)
    print("\n--- Shot contiguity ---")
    changes = np.flatnonzero(shot_id[1:] != shot_id[:-1]) + 1
    n_groups = len(changes) + 1
    print(f"Number of contiguous SHOTID blocks: {n_groups}")
    unique_shots = np.unique(shot_id)
    print(f"Number of unique SHOTID values: {len(unique_shots)}")
    if n_groups == len(unique_shots):
        print("RESULT: Every SHOTID appears as one contiguous block.")
    else:
        print(
            "WARNING: Some SHOTID values appear in multiple "
            "non-contiguous blocks."
        )
    print("\nFirst block boundaries:")
    boundaries = np.concatenate(
        [
            np.array([0]),
            changes,
            np.array([len(shot_id)]),
        ]
    )

    for i in range(min(10, len(boundaries) - 1)):
        start = boundaries[i]
        end = boundaries[i + 1]
        print(
            f"  block {i:3d}: "
            f"indices [{start}, {end}) "
            f"SHOTID={shot_id[start]}"
        )

def inspect_receiver_sorting(group: h5py.Group) -> None:
    """Check whether receiver coordinates are ordered within shots."""
    shot_id = group["SHOTID"][:].reshape(-1)
    rec_peg = group["REC_PEG"][:].reshape(-1)
    rec_x = group["REC_X"][:].reshape(-1)
    rec_y = group["REC_Y"][:].reshape(-1)
    unique_shots = np.unique(shot_id)
    print("\n--- Receiver ordering within shots ---")
    checked = 0
    for shot in unique_shots[:20]:
        indices = np.flatnonzero(shot_id == shot)
        if len(indices) < 3:
            continue
        peg = rec_peg[indices]
        x = rec_x[indices]
        y = rec_y[indices]
        peg_increasing = np.all(np.diff(peg) >= 0)
        x_increasing = np.all(np.diff(x) >= 0)
        y_increasing = np.all(np.diff(y) >= 0)
        print(
            f"SHOTID={shot:.0f} "
            f"n={len(indices):5d} | "
            f"REC_PEG increasing={peg_increasing} | "
            f"REC_X increasing={x_increasing} | "
            f"REC_Y increasing={y_increasing}"
        )

        if checked >= 9:
            break
        checked += 1

def inspect_asset(name: str, filename: str) -> None:
    """Inspect one seismic asset."""
    path = RAW_DIR / filename
    print("\n" + "=" * 80)
    print(f"ASSET: {name}")
    print("=" * 80)
    print(f"File: {path}")
    if not path.exists():
        print("ERROR: File does not exist.")
        return
    with h5py.File(path, "r") as h5:
        group = h5["TRACE_DATA/DEFAULT"]
        print("\n--- Basic information ---")
        n_traces = group["data_array"].shape[0]
        n_samples = group["data_array"].shape[1]
        print(f"Number of traces:   {n_traces:,}")
        print(f"Samples per trace:  {n_samples:,}")
        samp_rate = np.unique(group["SAMP_RATE"][:])
        print(f"SAMP_RATE:          {samp_rate}")
        print("\n--- Geometry analysis ---")
        inspect_shots(group)
        inspect_trace_order(group)
        inspect_receiver_geometry(group)
        inspect_source_geometry(group)
        inspect_consecutive_shots(group)
        inspect_receiver_sorting(group)

def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - GEOMETRY EXPLORATION")
    print("=" * 80)
    for name, filename in ASSETS.items():
        inspect_asset(name, filename)
    print("\n" + "=" * 80)
    print("GEOMETRY EXPLORATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()