"""Re-run SAEBench sparse_probing + absorption for exp42c NoC_SAE.
Core eval already completed. This picks up where it left off."""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 18
D_MODEL = 4096
D_SAE = 65536
K = 80
NORM_EPS = 1e-8

CKPT_PATH = "/mnt/nvme0/checkpoints/exp42c/no_C_L18_final.pt"
RESULTS_PATH = "experiments/exp42c_noc_qwen_results.json"


# --- NoCBatchTopKSAE (must match training definition exactly) ---

@torch.no_grad()
def set_decoder_norm_to_unit_norm(W_dec):
    norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_dec.div_(norms)
    return W_dec


class NoCBatchTopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.tensor(-1.0))
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_active=False):
        x_c = x - self.b_dec
        x_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_u = x_c / x_norm
        w_u = F.normalize(self.W_enc, dim=-1, eps=NORM_EPS)
        post_relu = F.relu(x_u @ w_u.T)
        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)
        self._cached_x_norm = x_norm
        if return_active:
            active_indices = encoded.sum(0) > 0
            return encoded, active_indices, post_relu
        return encoded

    def decode(self, f, x_norm=None):
        w_u = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w_u
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        if x_norm is None:
            x_norm = getattr(self, "_cached_x_norm", None)
        if x_norm is not None:
            x_raw = x_raw * (x_norm / raw_norm)
        return x_raw + self.b_dec

    def forward(self, x, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x)
        return self.decode(f), f


def main():
    print("Loading NoC_SAE checkpoint...")
    sae = NoCBatchTopKSAE(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    print(f"  Loaded from {CKPT_PATH}")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmarks.adapter import BenchSAE
    from benchmarks.run_saebench import run_saebench

    def encode_fn(x):
        return sae.encode(x)

    def decode_fn(f):
        return sae.decode(f)

    W_dec = sae.W_dec.detach()
    W_dec_normed = F.normalize(W_dec, dim=1)

    bench_sae = BenchSAE(
        W_enc=sae.W_enc.detach().T,
        W_dec=W_dec_normed,
        b_enc=torch.zeros(D_SAE, device=DEVICE, dtype=sae.W_enc.dtype),
        b_dec=sae.b_dec.detach(),
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name=MODEL_NAME,
        hook_layer=LAYER,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\nRunning SAEBench (sparse_probing + absorption)...")
    results = run_saebench(
        bench_sae,
        sae_name=f"exp42c-no_C-L{LAYER}",
        eval_types=["sparse_probing", "absorption"],
        output_dir="/mnt/nvme0/saebench_results/exp42c",
        llm_batch_size=4,
        device=DEVICE,
    )

    # Merge into existing results
    if Path(RESULTS_PATH).exists():
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    run = all_results.setdefault("no_C_L18", {})
    sb = run.setdefault("saebench", {})
    # Keep existing core results
    if results:
        sb.update(results)

    # Also load core results from saved file
    core_path = Path("/mnt/nvme0/saebench_results/exp42c/core/exp42c-no_C-L18_custom_sae_eval_results.json")
    if core_path.exists():
        import json as _json
        with open(core_path) as f:
            core_data = _json.load(f)
        sb["core"] = core_data.get("eval_result_metrics", core_data.get("metrics", core_data))

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults merged into {RESULTS_PATH}")
    print(f"SAEBench results: {results}")


if __name__ == "__main__":
    main()
