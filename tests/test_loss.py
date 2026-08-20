import torch

from src.loss import (
    FirstBreakLoss,
    make_first_break_target,
)


def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - LOSS TEST")
    print("=" * 80)

    batch_size = 4
    n_samples = 1501

    first_break = torch.tensor(
        [300.0, 500.0, 700.0, 900.0],
        dtype=torch.float32,
    )

    print("\nFirst-break labels:")
    print(first_break)

    target = make_first_break_target(
        first_break=first_break,
        n_samples=n_samples,
        sigma=3.0,
    )

    print("\nTarget:")
    print(f"  shape: {target.shape}")
    print(f"  dtype: {target.dtype}")

    assert target.shape == (
        batch_size,
        n_samples,
    )

    assert torch.isfinite(target).all()

    # Target maximum should be close to 1.
    max_values = target.max(dim=1).values

    print("\nTarget maximum values:")
    print(max_values)

    assert torch.allclose(
        max_values,
        torch.ones(batch_size),
        atol=1e-5,
    )

    # Check that the maximum occurs near the
    # ground-truth first-break position.
    max_indices = target.argmax(dim=1)

    print("\nTarget peak positions:")
    print(max_indices)

    assert torch.all(
        torch.abs(
            max_indices
            - first_break.long()
        )
        <= 1
    )

    # Test loss.
    logits = torch.randn(
        batch_size,
        n_samples,
        dtype=torch.float32,
    )

    loss_fn = FirstBreakLoss(
        sigma=3.0,
        pos_weight=10.0,
    )

    loss = loss_fn(
        logits,
        first_break,
    )

    print("\nLoss:")
    print(f"  value: {loss.item():.6f}")

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Test gradient.
    logits.requires_grad_(True)

    loss = loss_fn(
        logits,
        first_break,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    print("\nChecks:")
    print("  target shape:      PASS")
    print("  finite target:     PASS")
    print("  target peak:       PASS")
    print("  loss scalar:       PASS")
    print("  finite loss:       PASS")
    print("  backward:          PASS")

    print("\n" + "=" * 80)
    print("LOSS TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()