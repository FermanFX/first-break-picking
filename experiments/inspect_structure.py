from __future__ import annotations
from pathlib import Path
from typing import Any
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ASSETS = {
    "Brunswick": "Brunswick_orig_1500ms_V2.hdf5",
    "Halfmile": "Halfmile3D_add_geom_sorted.hdf5",
    "Lalor": "Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
    "Sudbury": "preprocessed_Sudbury3D.hdf",
}

def print_attributes(
    obj: Any,
    indent: str = "    ",
) -> None:
    """Print HDF5/HDF object attributes."""
    if not obj.attrs:
        return
    print(f"{indent}Attributes:")
    for key, value in obj.attrs.items():
        print(f"{indent}  {key}: {value}")

def inspect_hdf5_group(
    group: h5py.Group,
    prefix: str = "",
) -> None:
    """Recursively inspect an HDF5 group."""
    for name, obj in group.items():
        full_name = f"{prefix}/{name}" if prefix else name

        if isinstance(obj, h5py.Dataset):
            print(f"\nDATASET: {full_name}")
            print(f"  shape:       {obj.shape}")
            print(f"  dtype:       {obj.dtype}")
            print(f"  chunks:      {obj.chunks}")
            print(f"  compression: {obj.compression}")
            print_attributes(obj)
        elif isinstance(obj, h5py.Group):
            print(f"\nGROUP: {full_name}")
            print_attributes(obj)
            inspect_hdf5_group(obj, full_name)

def inspect_asset(name: str, filename: str) -> None:
    """Inspect one HDF5 asset."""
    path = RAW_DIR / filename
    print("=" * 80)
    print(f"ASSET: {name}")
    print("=" * 80)
    print(f"File: {path}")

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return
    print(f"Size: {path.stat().st_size / (1024**3):.2f} GB")

    try:
        with h5py.File(path, "r") as f:
            print("\n--- Root keys ---")
            if len(f.keys()) == 0:
                print("  <empty>")
            else:
                for key in f.keys():
                    print(f"  {key}")
            print("\n--- HDF5 structure ---")
            inspect_hdf5_group(f)

    except OSError as exc:
        print(f"\nERROR: Could not open file with h5py.")
        print(f"Reason: {exc}")

def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - DATASET STRUCTURE INSPECTION")
    print("=" * 80)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DIR}")
    for name, filename in ASSETS.items():
        inspect_asset(name, filename)
        print()

if __name__ == "__main__":
    main()