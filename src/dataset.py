from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

RAW_DIR = Path("/kaggle/input/datasets/frmanxankiiyev/first-break-picking-data")
MANIFEST_PATH = Path("/kaggle/working/first-break-picking/data/processed/shot_manifest_split.csv")


ASSETS = {
    "Brunswick": "Brunswick_orig_1500ms_V2.hdf5",
    "Halfmile": "Halfmile3D_add_geom_sorted.hdf5",
    "Lalor": "Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
    "Sudbury": "preprocessed_Sudbury3D.hdf",
}


class FirstBreakTraceDataset(Dataset):
    """
    Trace-level dataset for first-break picking.

    Only traces with valid manual first-break labels are included.

    Each item contains:
        waveform: [samples]
        first_break: first-break sample index
        asset: asset name
        shot_id: shot identifier
        trace_index: original HDF5 trace index
    """

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_PATH,
        split: str = "train",
        assets: list[str] | None = None,
    ) -> None:

        self.manifest_path = Path(manifest_path)
        self.split = split

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of: train, val, test"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        df = pd.read_csv(self.manifest_path)

        df = df[df["split"] == split].copy()

        if assets is not None:
            df = df[df["asset"].isin(assets)].copy()

        if len(df) == 0:
            raise ValueError(
                f"No shots found for split='{split}'."
            )

        self.shots = df.reset_index(drop=True)

        self._files: dict[str, h5py.File] = {}

        # -----------------------------------------------------
        # Build trace-level sample index.
        #
        # IMPORTANT:
        # We inspect SPARE1 and include ONLY traces whose
        # first-break label is valid.
        # -----------------------------------------------------

        self.samples: list[
            tuple[str, int, int]
        ] = []

        for _, row in self.shots.iterrows():

            asset = str(row["asset"])

            trace_start = int(row["trace_start"])
            trace_end = int(row["trace_end"])

            h5 = self._get_file(asset)
            group = h5["TRACE_DATA/DEFAULT"]

            picks = np.asarray(
                group["SPARE1"][trace_start:trace_end]
            ).reshape(-1)

            valid = np.isfinite(picks) & (picks > 0)

            valid_indices = np.flatnonzero(valid)

            for local_index in valid_indices:

                trace_index = (
                    trace_start + int(local_index)
                )

                self.samples.append(
                    (
                        asset,
                        int(row["shot_id"]),
                        trace_index,
                    )
                )

    def _get_file(
        self,
        asset: str,
    ) -> h5py.File:
        """Open HDF5 file lazily."""

        if asset not in self._files:

            if asset not in ASSETS:
                raise KeyError(
                    f"Unknown asset: {asset}"
                )

            filename = ASSETS[asset]
            path = RAW_DIR / filename

            if not path.exists():
                raise FileNotFoundError(
                    f"Raw file not found: {path}"
                )

            self._files[asset] = h5py.File(
                path,
                "r",
            )

        return self._files[asset]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | int | str]:

        asset, shot_id, trace_index = (
            self.samples[index]
        )

        h5 = self._get_file(asset)

        group = h5["TRACE_DATA/DEFAULT"]

        waveform = np.asarray(
            group["data_array"][trace_index],
            dtype=np.float32,
        ).reshape(-1)

        pick_ms = float(
            np.asarray(
                group["SPARE1"][trace_index]
            ).reshape(-1)[0]
        )

        samp_rate = float(
            np.asarray(
                group["SAMP_RATE"][trace_index]
            ).reshape(-1)[0]
        )

        # milliseconds -> sample index
        pick_sample = (
            pick_ms * samp_rate / 1000.0
        )

        return {
            "waveform": torch.from_numpy(waveform),
            "first_break": torch.tensor(
                pick_sample,
                dtype=torch.float32,
            ),
            "asset": asset,
            "shot_id": shot_id,
            "trace_index": trace_index,
        }

    def close(self) -> None:
        """Close all open HDF5 files."""

        for file in self._files.values():
            file.close()

        self._files.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
