"""
CPU smoke tests for the benchmark harness.

Tests adapter wrapping, shape correctness, decoder norm checks, and
CosineAutoEncoder tuple handling — all without GPU or model downloads.

Run: python benchmarks/test_harness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.adapter import BenchSAE, SAEFeaturizer, SAEInverseFeaturizer


def _make_random_bench_sae(
    d_in: int = 64, d_sae: int = 128, device: str = "cpu"
) -> BenchSAE:
    """Create a BenchSAE with random weights for testing."""
    W_enc = torch.randn(d_in, d_sae)
    W_dec = F.normalize(torch.randn(d_sae, d_in), dim=1)
    b_enc = torch.zeros(d_sae)
    b_dec = torch.zeros(d_in)

    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ W_enc + b_enc)

    def decode_fn(f: torch.Tensor) -> torch.Tensor:
        return f @ W_dec + b_dec

    return BenchSAE(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name="test-model",
        hook_layer=0,
        device=torch.device(device),
        dtype=torch.float32,
    )


def test_bench_sae_shapes():
    """BenchSAE encode/decode/forward produce correct shapes."""
    sae = _make_random_bench_sae(d_in=64, d_sae=128)
    x = torch.randn(4, 64)

    encoded = sae.encode(x)
    assert encoded.shape == (4, 128), f"encode shape: {encoded.shape}"

    decoded = sae.decode(encoded)
    assert decoded.shape == (4, 64), f"decode shape: {decoded.shape}"

    reconstructed = sae.forward(x)
    assert reconstructed.shape == (4, 64), f"forward shape: {reconstructed.shape}"

    print("  PASS: shapes correct")


def test_bench_sae_config():
    """BenchSAE cfg has expected fields."""
    sae = _make_random_bench_sae(d_in=64, d_sae=128)

    assert sae.cfg.d_in == 64
    assert sae.cfg.d_sae == 128
    assert sae.cfg.model_name == "test-model"
    assert sae.cfg.hook_layer == 0
    assert sae.cfg.hook_name == "blocks.0.hook_resid_post"

    print("  PASS: config correct")


def test_decoder_norms():
    """BenchSAE passes decoder norm check."""
    sae = _make_random_bench_sae()
    assert sae.check_decoder_norms(), "Decoder norms should be unit"
    print("  PASS: decoder norms unit")


def test_weight_shapes():
    """W_enc and W_dec have SAEBench-expected shapes."""
    sae = _make_random_bench_sae(d_in=64, d_sae=128)

    assert sae.W_enc.shape == (64, 128), f"W_enc: {sae.W_enc.shape}"
    assert sae.W_dec.shape == (128, 64), f"W_dec: {sae.W_dec.shape}"
    assert sae.b_enc.shape == (128,), f"b_enc: {sae.b_enc.shape}"
    assert sae.b_dec.shape == (64,), f"b_dec: {sae.b_dec.shape}"

    print("  PASS: weight shapes correct")


def test_cosine_autoencoder_wrapping():
    """CosineAutoEncoder tuple returns are handled correctly."""
    from src.custom_encoder.architectures import CosineAutoEncoder

    ae = CosineAutoEncoder(activation_dim=64, dict_size=128)
    ae.eval()

    sae = BenchSAE._from_dictionary(
        ae,
        model_name="test-model",
        hook_layer=0,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(4, 64)
    encoded = sae.encode(x)
    assert isinstance(encoded, torch.Tensor), "encode should return single Tensor"
    assert encoded.shape == (4, 128), f"encode shape: {encoded.shape}"

    decoded = sae.decode(encoded)
    assert decoded.shape == (4, 64), f"decode shape: {decoded.shape}"

    reconstructed = sae.forward(x)
    assert reconstructed.shape == (4, 64), f"forward shape: {reconstructed.shape}"

    assert sae.check_decoder_norms(), "Decoder norms should be unit"
    print("  PASS: CosineAutoEncoder wrapping correct")


def test_cosine_batchtopk_wrapping():
    """CosineBatchTopKAutoEncoder wraps cleanly (single-tensor encode/decode)."""
    from src.custom_encoder.architectures import CosineBatchTopKAutoEncoder

    ae = CosineBatchTopKAutoEncoder(activation_dim=64, dict_size=128, k=10)
    ae.eval()

    sae = BenchSAE._from_dictionary(
        ae,
        model_name="test-model",
        hook_layer=0,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(4, 64)
    encoded = sae.encode(x)
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (4, 128)

    reconstructed = sae.forward(x)
    assert reconstructed.shape == (4, 64)

    assert sae.check_decoder_norms()
    print("  PASS: CosineBatchTopKAutoEncoder wrapping correct")


def test_batchtopk_wrapping():
    """BatchTopKAutoEncoder wraps cleanly."""
    from src.custom_encoder.architectures import BatchTopKAutoEncoder

    ae = BatchTopKAutoEncoder(activation_dim=64, dict_size=128, k=10)
    ae.eval()

    sae = BenchSAE._from_dictionary(
        ae,
        model_name="test-model",
        hook_layer=0,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(4, 64)
    encoded = sae.encode(x)
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (4, 128)

    reconstructed = sae.forward(x)
    assert reconstructed.shape == (4, 64)

    assert sae.check_decoder_norms()
    print("  PASS: BatchTopKAutoEncoder wrapping correct")


def test_mib_featurizer():
    """SAEFeaturizer and SAEInverseFeaturizer produce correct shapes."""
    sae = _make_random_bench_sae(d_in=64, d_sae=128)
    featurizer = SAEFeaturizer(sae)
    inverse = SAEInverseFeaturizer(sae)

    x = torch.randn(4, 64)
    features, error = featurizer(x)

    assert features.shape == (4, 128), f"features: {features.shape}"
    assert error.shape == (4, 64), f"error: {error.shape}"

    x_hat = inverse(features, error)
    assert x_hat.shape == (4, 64), f"x_hat: {x_hat.shape}"

    # Reconstruction with error should be exact
    assert torch.allclose(x, x_hat, atol=1e-5), "featurizer roundtrip failed"
    print("  PASS: MIB featurizer roundtrip correct")


def test_adamkarvonen_loader():
    """from_adamkarvonen loads the real checkpoint (requires HF download)."""
    try:
        sae = BenchSAE.from_adamkarvonen(
            hook_layer=18,
            device="cpu",
            dtype=torch.float32,
        )
    except Exception as e:
        print(f"  SKIP: adamkarvonen loader ({e})")
        return

    assert sae.cfg.d_in == 4096, f"d_in: {sae.cfg.d_in}"
    assert sae.cfg.d_sae == 65536, f"d_sae: {sae.cfg.d_sae}"
    assert sae.cfg.model_name == "Qwen/Qwen3-8B"
    assert sae.check_decoder_norms(), "Decoder norms should be unit"

    # Test encode on random input
    x = torch.randn(2, 4096)
    encoded = sae.encode(x)
    assert encoded.shape == (2, 65536)

    # Should be sparse
    sparsity = (encoded > 0).float().mean().item()
    assert sparsity < 0.1, f"Expected sparse, got density {sparsity}"

    print(f"  PASS: adamkarvonen loader (d_sae={sae.cfg.d_sae}, density={sparsity:.4f})")


if __name__ == "__main__":
    print("Benchmark harness smoke tests")
    print("=" * 40)

    tests = [
        ("BenchSAE shapes", test_bench_sae_shapes),
        ("BenchSAE config", test_bench_sae_config),
        ("Decoder norms", test_decoder_norms),
        ("Weight shapes", test_weight_shapes),
        ("CosineAutoEncoder wrapping", test_cosine_autoencoder_wrapping),
        ("CosineBatchTopKAutoEncoder wrapping", test_cosine_batchtopk_wrapping),
        ("BatchTopKAutoEncoder wrapping", test_batchtopk_wrapping),
        ("MIB featurizer roundtrip", test_mib_featurizer),
        ("adamkarvonen loader", test_adamkarvonen_loader),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        try:
            print(f"\n[{name}]")
            test_fn()
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
                print(f"  FAIL: {e}")
                failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        sys.exit(1)
