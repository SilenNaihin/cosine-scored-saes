"""
Universal SAE adapter for MI benchmarks.

Wraps any SAE source (adamkarvonen HF checkpoints, custom training pipeline,
sae-lens) into SAEBench's BaseSAE interface. Also provides helpers for MIB
Track 2 featurizer format.

Usage:
    # Existing inner-product SAE from HuggingFace
    sae = BenchSAE.from_adamkarvonen(hook_layer=18, device="cuda")

    # Custom-trained SAE (CosineBatchTopK, BatchTopK, etc.)
    sae = BenchSAE.from_custom_checkpoint("path/to/ae.pt", hook_layer=18, device="cuda")

    # sae-lens SAE
    sae = BenchSAE.from_saelens("pythia-70m-deduped-res-sm", hook_layer=3, device="cuda")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_bench.custom_saes.base_sae import BaseSAE

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class BenchSAE(BaseSAE):
    """Wraps any SAE into SAEBench's BaseSAE interface.

    Rather than subclassing per-architecture, this takes encode/decode callables
    and weight tensors, keeping the adapter minimal and generic.
    """

    def __init__(
        self,
        *,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        model_name: str,
        hook_layer: int,
        device: torch.device | str,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        device = torch.device(device) if isinstance(device, str) else device
        d_sae, d_in = W_dec.shape

        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
            hook_name=hook_name,
        )

        with torch.no_grad():
            self.W_enc.copy_(W_enc.to(dtype=dtype))
            self.W_dec.copy_(W_dec.to(dtype=dtype))
            self.b_enc.copy_(b_enc.to(dtype=dtype))
            self.b_dec.copy_(b_dec.to(dtype=dtype))

        self._encode_fn = encode_fn
        self._decode_fn = decode_fn

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_fn(x)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return self._decode_fn(feature_acts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    # Factory: adamkarvonen/qwen3-8b-saes (BatchTopK from dictionary_learning)
    # ------------------------------------------------------------------
    @classmethod
    def from_adamkarvonen(
        cls,
        hook_layer: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        repo_id: str = "adamkarvonen/qwen3-8b-saes",
        filename: str | None = None,
        model_name: str = "Qwen/Qwen3-8B",
    ) -> BenchSAE:
        """Load a BatchTopK SAE from adamkarvonen's HuggingFace repo.

        Checkpoint format (flat dict):
            encoder.weight: (d_sae, d_in)
            encoder.bias:   (d_sae,)
            decoder.weight: (d_in, d_sae)   # NOTE: transposed vs SAEBench convention
            b_dec:          (d_in,)
            k:              scalar
            threshold:      scalar
        """
        from huggingface_hub import hf_hub_download

        if filename is None:
            filename = (
                f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{hook_layer}"
                f"/trainer_2/ae.pt"
            )

        path = hf_hub_download(repo_id=repo_id, filename=filename)
        # Always load to CPU first (checkpoint may have CUDA locations)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Extract weights — decoder.weight is (d_in, d_sae), we need (d_sae, d_in)
        W_enc = ckpt["encoder.weight"]  # (d_sae, d_in)
        b_enc = ckpt["encoder.bias"]  # (d_sae,)
        W_dec_raw = ckpt["decoder.weight"]  # (d_in, d_sae)
        b_dec = ckpt["b_dec"]  # (d_in,)
        k = int(ckpt["k"].item())
        threshold = ckpt["threshold"].float().item()

        # SAEBench wants W_dec as (d_sae, d_in) with unit-norm rows
        W_dec = W_dec_raw.T.contiguous()  # (d_sae, d_in)
        W_dec = F.normalize(W_dec, dim=1)

        # Build encode/decode closures that capture the weights
        _W_enc = W_enc.to(device=device, dtype=dtype)
        _b_enc = b_enc.to(device=device, dtype=dtype)
        _W_dec_raw = W_dec_raw.to(device=device, dtype=dtype)
        _b_dec = b_dec.to(device=device, dtype=dtype)
        _threshold = threshold

        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            pre_acts = (x - _b_dec) @ _W_enc.T + _b_enc
            acts = torch.relu(pre_acts)
            return torch.where(acts >= _threshold, acts, torch.zeros_like(acts))

        def decode_fn(f: torch.Tensor) -> torch.Tensor:
            return f @ _W_dec_raw.T + _b_dec

        return cls(
            W_enc=W_enc.T,  # BaseSAE wants (d_in, d_sae)
            W_dec=W_dec,
            b_enc=b_enc,
            b_dec=b_dec,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Factory: custom training pipeline checkpoint
    # ------------------------------------------------------------------
    @classmethod
    def from_custom_checkpoint(
        cls,
        checkpoint_path: str | Path,
        hook_layer: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        model_name: str = "Qwen/Qwen3-8B",
    ) -> BenchSAE:
        """Load an SAE from the custom training pipeline (src/custom_encoder/train.py).

        Checkpoint format:
            model_state_dict: the architecture's state_dict
            config: Hydra config dict with config["arch"]["name"] identifying the class
        """
        from src.custom_encoder.architectures import ARCHITECTURE_REGISTRY

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        arch_name = config["arch"]["name"]

        ae_cls = ARCHITECTURE_REGISTRY[arch_name]
        activation_dim = config["model"].get("d_model") or config.get(
            "activation_dim"
        )
        dict_size = config["arch"].get("dict_size") or config.get("dict_size")

        # If dict_size is a multiplier, compute actual size
        if dict_size is None:
            dict_size = int(
                config["arch"].get("dict_multiplier", 4) * activation_dim
            )

        # Reconstruct the architecture
        from omegaconf import OmegaConf

        arch_cfg = OmegaConf.create(config["arch"])
        ae = ae_cls.from_arch_config(activation_dim, dict_size, arch_cfg)
        ae.load_state_dict(ckpt["model_state_dict"])
        ae = ae.to(device=device, dtype=dtype).eval()

        return cls._from_dictionary(
            ae,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Factory: sae-lens
    # ------------------------------------------------------------------
    @classmethod
    def from_saelens(
        cls,
        release: str,
        sae_id: str,
        hook_layer: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        model_name: str | None = None,
    ) -> BenchSAE:
        """Load an SAE from the sae-lens library."""
        from sae_lens import SAE

        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))
        sae = sae.to(dtype=dtype).eval()

        if model_name is None:
            model_name = sae.cfg.model_name

        W_dec = sae.W_dec.detach()  # (d_sae, d_in)
        W_dec = F.normalize(W_dec, dim=1)

        W_enc = sae.W_enc.detach().T  # sae-lens: (d_in, d_sae) -> we need same
        b_enc = sae.b_enc.detach()
        b_dec = sae.b_dec.detach()

        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            return sae.encode(x)

        def decode_fn(f: torch.Tensor) -> torch.Tensor:
            return sae.decode(f)

        return cls(
            W_enc=sae.W_enc.detach(),  # (d_in, d_sae)
            W_dec=W_dec,
            b_enc=b_enc,
            b_dec=b_dec,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Internal: wrap a Dictionary subclass from src/custom_encoder
    # ------------------------------------------------------------------
    @classmethod
    def _from_dictionary(
        cls,
        ae: nn.Module,
        model_name: str,
        hook_layer: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> BenchSAE:
        """Wrap any src.custom_encoder.architectures.Dictionary instance."""
        from src.custom_encoder.architectures import CosineAutoEncoder

        d_in = ae.activation_dim
        d_sae = ae.dict_size

        if isinstance(ae, CosineAutoEncoder):
            # CosineAutoEncoder.encode returns (features, x_norm) — strip the norm
            # CosineAutoEncoder.decode expects (features, x_norm) — we can't call it
            # without the norm, so we store it between encode/decode calls.
            _stored_norm = {}

            def encode_fn(x: torch.Tensor) -> torch.Tensor:
                features, x_norm = ae.encode(x)
                _stored_norm["val"] = x_norm
                return features

            def decode_fn(f: torch.Tensor) -> torch.Tensor:
                x_norm = _stored_norm.get("val")
                if x_norm is None:
                    # Fallback: assume unit norm (decode without magnitude info)
                    return ae.decoder(f)
                return ae.decode((f, x_norm))

            # CosineAutoEncoder has encoder_weight (d_sae, d_in), not nn.Linear
            W_enc = ae._normalized_encoder_weight().detach().T  # (d_in, d_sae)
            W_dec = ae.decoder.weight.T.detach().contiguous()  # (d_sae, d_in)
            W_dec = F.normalize(W_dec, dim=1)
            b_enc = (-ae.b).detach()  # bias acts as negative threshold
            b_dec = torch.zeros(d_in, device=device, dtype=dtype)
        else:
            # Standard architectures: AutoEncoder, BatchTopKAutoEncoder,
            # CosineBatchTopKAutoEncoder — all have single-tensor encode/decode

            def encode_fn(x: torch.Tensor) -> torch.Tensor:
                return ae.encode(x)

            def decode_fn(f: torch.Tensor) -> torch.Tensor:
                return ae.decode(f)

            # All standard architectures use nn.Linear encoder/decoder
            W_enc = ae.encoder.weight.detach().T  # (d_in, d_sae)
            W_dec = ae.decoder.weight.T.detach().contiguous()  # (d_sae, d_in)
            W_dec = F.normalize(W_dec, dim=1)
            b_enc = ae.encoder.bias.detach()
            b_dec = ae.b_dec.detach()

        return cls(
            W_enc=W_enc,
            W_dec=W_dec,
            b_enc=b_enc,
            b_dec=b_dec,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
        )


# ------------------------------------------------------------------
# MIB Track 2 helpers
# ------------------------------------------------------------------


class SAEFeaturizer(nn.Module):
    """MIB featurizer: x -> (features, reconstruction_error)."""

    def __init__(self, sae: BenchSAE):
        super().__init__()
        self.sae = sae

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.sae.encode(x)
        reconstruction = self.sae.decode(features)
        error = x - reconstruction
        return features, error


class SAEInverseFeaturizer(nn.Module):
    """MIB inverse featurizer: (features, error) -> x_hat."""

    def __init__(self, sae: BenchSAE):
        super().__init__()
        self.sae = sae

    def forward(
        self, features: torch.Tensor, error: torch.Tensor
    ) -> torch.Tensor:
        return self.sae.decode(features) + error
