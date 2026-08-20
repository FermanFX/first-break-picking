import torch

from src.model import FirstBreakGRU, build_model


def main() -> None:
    print("=" * 80)
    print("FIRST BREAK PICKING - MODEL TEST")
    print("=" * 80)

    batch_size = 4
    n_samples = 1501

    x = torch.randn(
        batch_size,
        n_samples,
        dtype=torch.float32,
    )

    print("\nInput:")
    print(f"  shape: {x.shape}")
    print(f"  dtype: {x.dtype}")

    model = build_model(
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
    )

    model.eval()

    with torch.no_grad():
        logits = model(x)

    print("\nModel:")
    print(model)

    print("\nOutput:")
    print(f"  shape: {logits.shape}")
    print(f"  dtype: {logits.dtype}")

    assert logits.shape == (
        batch_size,
        n_samples,
    )

    assert logits.dtype == torch.float32

    assert torch.isfinite(logits).all()

    print("\nChecks:")

    print(
        "  output shape:     PASS"
        if logits.shape == (batch_size, n_samples)
        else "  output shape:     FAIL"
    )

    print(
        "  dtype:             PASS"
        if logits.dtype == torch.float32
        else "  dtype:             FAIL"
    )

    print(
        "  finite values:     PASS"
        if torch.isfinite(logits).all()
        else "  finite values:     FAIL"
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nParameters:")
    print(f"  total:     {parameters:,}")
    print(f"  trainable: {trainable_parameters:,}")

    print("\n" + "=" * 80)
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()