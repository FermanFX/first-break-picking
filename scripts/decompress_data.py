from __future__ import annotations
import lzma
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "Brunswick": {
        "source": "Brunswick_orig_1500ms_V2.hdf5.xz",
        "output": "Brunswick_orig_1500ms_V2.hdf5",
    },
    "Halfmile": {
        "source": "Halfmile3D_add_geom_sorted.hdf5.xz",
        "output": "Halfmile3D_add_geom_sorted.hdf5",
    },
    "Lalor": {
        "source": "Lalor_raw_z_1500ms_norp_geom_v3.hdf5.xz",
        "output": "Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
    },
    "Sudbury": {
        "source": "preprocessed_Sudbury3D.hdf.xz",
        "output": "preprocessed_Sudbury3D.hdf",
    },
}

def format_size(size_bytes: int) -> str:
    """Convert bytes to a readable size."""
    size_gb = size_bytes / (1024**3)
    if size_gb >= 1:
        return f"{size_gb:.2f} GB"
    size_mb = size_bytes / (1024**2)
    return f"{size_mb:.2f} MB"

def decompress_dataset(name: str) -> None:
    """Decompress one seismic dataset."""
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {available}"
        )
    config = DATASETS[name]
    source = RAW_DATA_DIR / config["source"]
    output = RAW_DATA_DIR / config["output"]
    temporary = RAW_DATA_DIR / f"{config['output']}.part"
    print("=" * 70)
    print(f"Decompressing: {name}")
    print("=" * 70)
    if not source.exists():
        raise FileNotFoundError(
            f"Source file not found:\n{source}"
        )
    if output.exists():
        print(f"Output already exists:")
        print(f"  {output}")
        print(f"  Size: {format_size(output.stat().st_size)}")
        print("\nNothing to do.")
        return
    print(f"Source:")
    print(f"  {source}")
    print(f"  Size: {format_size(source.stat().st_size)}")
    print("\nOutput:")
    print(f"  {output}")
    print("\nStarting decompression...")

    try:
        with lzma.open(source, "rb") as compressed:
            with temporary.open("wb") as decompressed:
                shutil.copyfileobj(
                    compressed,
                    decompressed,
                    length=1024 * 1024,
                )
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    print("\nDecompression completed successfully.")
    print(f"Output size: {format_size(output.stat().st_size)}")
    print(f"Output file: {output}")

def print_usage() -> None:
    print("Usage:")
    print("  python scripts/decompress_data.py <dataset>")
    print()
    print("Available datasets:")
    for name in DATASETS:
        print(f"  - {name}")

def main() -> int:
    if len(sys.argv) != 2:
        print_usage()
        return 1
    dataset_name = sys.argv[1]
    try:
        decompress_dataset(dataset_name)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())