from pathlib import Path
import h5py
import numpy as np

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Brunswick_orig_1500ms_V2.hdf5"
)
TRACE_DATASET = "TRACE_DATA/DEFAULT/data_array"
N_TRACES = 10

# ============================================================
# Helpers
# ============================================================
def print_dataset_info(h5file: h5py.File, name: str) -> None:
    """Print shape, dtype and basic statistics for a dataset."""
    dataset = h5file[name]
    print(f"\n{name}")
    print(f"  shape: {dataset.shape}")
    print(f"  dtype: {dataset.dtype}")
    if dataset.size == 0:
        print("  empty dataset")
        return
    # Read only a small amount for large datasets
    if dataset.ndim >= 1 and dataset.shape[0] > 10000:
        sample = dataset[:10000]
    else:
        sample = dataset[:]
    if np.issubdtype(sample.dtype, np.number):
        sample = np.asarray(sample)
        print(f"  min:  {np.nanmin(sample)}")
        print(f"  max:  {np.nanmax(sample)}")
        print(f"  mean: {np.nanmean(sample):.6f}")
        print(f"  std:  {np.nanstd(sample):.6f}")
        print(f"  NaN:  {np.isnan(sample).sum()}")
        print(f"  Inf:  {np.isinf(sample).sum()}")

# ============================================================
# Main inspection
# ============================================================

def main() -> None:
    print("=" * 70)
    print("SEISMIC DATA INSPECTION")
    print("=" * 70)
    print(f"\nData file:")
    print(f"  {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Please place the HDF5 file inside data/raw/."
        )
    with h5py.File(DATA_PATH, "r") as f:
        # ----------------------------------------------------
        # 1. Main seismic data
        # ----------------------------------------------------
        print("\n" + "-" * 70)
        print("1. MAIN TRACE DATA")
        print("-" * 70)
        data = f[TRACE_DATASET]
        print(f"Dataset: {TRACE_DATASET}")
        print(f"Shape:   {data.shape}")
        print(f"Dtype:   {data.dtype}")
        n_traces, n_samples = data.shape
        print(f"\nNumber of traces:       {n_traces:,}")
        print(f"Samples per trace:     {n_samples}")
        print(f"Expected trace length: {n_samples}")

        # ----------------------------------------------------
        # 2. Sampling information
        # ----------------------------------------------------
        print("\n" + "-" * 70)
        print("2. SAMPLING INFORMATION")
        print("-" * 70)
        samp_rate = f["TRACE_DATA/DEFAULT/SAMP_RATE"][:10]
        samp_num = f["TRACE_DATA/DEFAULT/SAMP_NUM"][:10]
        print(f"SAMP_RATE first values: {samp_rate.flatten()}")
        print(f"SAMP_NUM first values:  {samp_num.flatten()}")
        sampling_interval_us = float(samp_rate[0, 0])
        sampling_interval_ms = sampling_interval_us / 1000.0
        record_length_ms = (n_samples - 1) * sampling_interval_ms
        print(f"\nSampling interval: {sampling_interval_ms:.3f} ms")
        print(f"Record length:     {record_length_ms:.3f} ms")

        # ----------------------------------------------------
        # 3. Read a few traces
        # ----------------------------------------------------
        print("\n" + "-" * 70)
        print("3. SAMPLE TRACE STATISTICS")
        print("-" * 70)
        count = min(N_TRACES, n_traces)
        traces = data[:count]

        for i, trace in enumerate(traces):
            trace = np.asarray(trace)
            print(f"\nTrace {i}:")
            print(f"  shape: {trace.shape}")
            print(f"  min:   {trace.min():.6f}")
            print(f"  max:   {trace.max():.6f}")
            print(f"  mean:  {trace.mean():.6f}")
            print(f"  std:   {trace.std():.6f}")
            print(f"  abs max: {np.abs(trace).max():.6f}")
            print(f"  NaN:   {np.isnan(trace).sum()}")
            print(f"  Inf:   {np.isinf(trace).sum()}")
        # ----------------------------------------------------
        # 4. Header information
        # ----------------------------------------------------
        print("\n" + "-" * 70)
        print("4. IMPORTANT TRACE HEADERS")
        print("-" * 70)

        headers = [
            "LINE",
            "REEL",
            "RECORDNUM",
            "SHOTID",
            "OFFSET",
            "SOURCE_X",
            "SOURCE_Y",
            "REC_X",
            "REC_Y",
            "SOURCE_HT",
            "REC_HT",
            "SOURCE_UPHOLE_TIME",
            "REC_UPHOLE_TIME",
            "SOURCE_STATIC",
            "REC_STATIC",
            "FTRACE",
            "SOURCENUM",
        ]

        for header in headers:
            path = f"TRACE_DATA/DEFAULT/{header}"
            if path in f:
                values = f[path][:5].flatten()
                print(f"{header:25s}: {values}")

        # ----------------------------------------------------
        # 5. First-break related datasets
        # ----------------------------------------------------
        print("\n" + "-" * 70)
        print("5. FIRST-BREAK RELATED DATASETS")
        print("-" * 70)
        first_break_candidates = [
            "FIRST_BREAK_TIME",
            "FIRST_BREAK_AMPLIT",
            "FIRST_BREAK_VELOCITY",
            "MODELLED_BREAK_TIME",
            "REF_MODEL_REC_STATIC",
            "REF_MODEL_SRC_STATIC",
            "REF_RESID_REC_STATIC",
            "REF_RESID_SRC_STATIC",
            "REF_TOTAL_STATIC",
        ]

        for name in first_break_candidates:
            path = f"TRACE_DATA/DEFAULT/{name}"
            if path not in f:
                print(f"{name:30s}: NOT FOUND")
                continue
            dataset = f[path]

            # Read a small subset
            values = dataset[:10000]
            print(f"\n{name}")
            print(f"  shape: {dataset.shape}")
            print(f"  dtype: {dataset.dtype}")
            print(f"  min:  {np.min(values)}")
            print(f"  max:  {np.max(values)}")
            print(f"  mean: {np.mean(values)}")
            print(f"  non-zero: {np.count_nonzero(values):,}")

        # ----------------------------------------------------
        # 6. Search for first-break-like dataset names
        # ----------------------------------------------------
        print("\n" + "-" * 70)
        print("6. SEARCHING FOR FIRST-BREAK / PICK DATASETS")
        print("-" * 70)

        keywords = [
            "break",
            "first",
            "pick",
            "arrival",
            "time",
        ]
        matching_paths = []

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in keywords):
                matching_paths.append(name)
        f.visititems(visitor)
        if matching_paths:
            print("\nPotentially relevant datasets/groups:")
            for path in matching_paths:
                print(f"  {path}")
        else:
            print("\nNo first-break-like dataset names found.")

        # ----------------------------------------------------
        # 7. Check trace/header consistency
        # ----------------------------------------------------
        print("\n" + "-" * 70)
        print("7. TRACE / HEADER CONSISTENCY")
        print("-" * 70)
        important_paths = [
            "LINE",
            "OFFSET",
            "SOURCE_X",
            "SOURCE_Y",
            "REC_X",
            "REC_Y",
            "SHOTID",
            "FTRACE",
        ]
        for name in important_paths:
            path = f"TRACE_DATA/DEFAULT/{name}"
            if path in f:
                shape = f[path].shape
                status = "OK" if shape[0] == n_traces else "MISMATCH"
                print(
                    f"{name:15s}: "
                    f"{shape[0]:,} values -> {status}"
                )

        # ----------------------------------------------------
        # 8. Summary
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(f"""
Dataset:
  traces       = {n_traces:,}
  samples      = {n_samples}
  sample rate  = {sampling_interval_ms:.3f} ms
  record       = {record_length_ms:.3f} ms

Main seismic data:
  {TRACE_DATASET}

Important:
  FIRST_BREAK_TIME appears to contain zeros.
  Therefore, we should NOT use it as a training target yet.

Next step:
  Inspect the actual trace waveforms and determine
  how the first-break labels are represented/provided.
""")

    print("=" * 70)
    print("Inspection completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()