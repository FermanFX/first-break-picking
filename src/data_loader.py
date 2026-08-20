from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import FirstBreakTraceDataset


def collate_first_break_batch(
    batch: list[dict],
) -> dict:
    """
    Collate seismic traces with different lengths.

    Waveforms are zero-padded to the longest waveform
    in the batch.
    """

    waveforms = [
        item["waveform"]
        for item in batch
    ]

    picks = torch.stack(
        [
            item["first_break"]
            for item in batch
        ]
    )

    lengths = torch.tensor(
        [
            waveform.shape[0]
            for waveform in waveforms
        ],
        dtype=torch.long,
    )

    max_length = max(
        waveform.shape[0]
        for waveform in waveforms
    )

    padded_waveforms = torch.zeros(
        len(waveforms),
        max_length,
        dtype=torch.float32,
    )

    for i, waveform in enumerate(waveforms):
        padded_waveforms[
            i,
            : waveform.shape[0],
        ] = waveform

    return {
        "waveform": padded_waveforms,
        "first_break": picks,
        "lengths": lengths,
        "asset": [
            item["asset"]
            for item in batch
        ],
        "shot_id": torch.tensor(
            [
                item["shot_id"]
                for item in batch
            ],
            dtype=torch.long,
        ),
        "trace_index": torch.tensor(
            [
                item["trace_index"]
                for item in batch
            ],
            dtype=torch.long,
        ),
    }


def create_dataloader(
    manifest_path: str | Path,
    split: str,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:

    if split not in {
        "train",
        "val",
        "test",
    }:
        raise ValueError(
            "split must be one of: train, val, test"
        )

    dataset = FirstBreakTraceDataset(
        manifest_path=manifest_path,
        split=split,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_first_break_batch,
    )