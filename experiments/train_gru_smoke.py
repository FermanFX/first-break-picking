from __future__ import annotations
from pathlib import Path
import torch
from src.data_loader import create_dataloader
from src.loss import FirstBreakLoss
from src.model import build_model

MANIFEST_PATH = Path(
    "data/processed/shot_manifest_split.csv"
)

def main() -> None:
    
    print("=" * 80)
    print("GRU TRAINING SMOKE TEST")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    loader = create_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        batch_size=4,
        shuffle=True,
    )

    model = build_model(
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        bidirectional=True,
    ).to(device)

    loss_fn = FirstBreakLoss(
        sigma=3.0,
        pos_weight=10.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    batch = next(iter(loader))

    waveform = batch["waveform"].to(device)
    first_break = batch["first_break"].to(device)

    print("\nBatch:")
    print(f"  waveform:     {waveform.shape}")
    print(f"  first_break:  {first_break.shape}")

    optimizer.zero_grad()

    logits = model(waveform)

    print("\nModel output:")
    print(f"  logits:       {logits.shape}")

    loss = loss_fn(
        logits,
        first_break,
    )

    print(
        f"\nInitial loss: "
        f"{loss.item():.6f}"
    )

    loss.backward()

    optimizer.step()

    print("\nGradient check:")

    gradient_found = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            if torch.isfinite(
                parameter.grad
            ).all():

                gradient_found = True
                break

    assert gradient_found

    print("  gradients: PASS")
    print("  forward:   PASS")
    print("  backward:  PASS")
    print("  optimizer:  PASS")

    print("\n" + "=" * 80)
    print("GRU TRAINING SMOKE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()