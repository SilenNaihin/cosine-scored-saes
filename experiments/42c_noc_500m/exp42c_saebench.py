"""exp42c: SAEBench (core + sparse_probing) on the 16k width-matched SAEs."""
import sys, gc, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from benchmarks.adapter import BenchSAE
from benchmarks.run_saebench import run_saebench
from exp40g_saebench import wrap_for_saebench
from exp39_norm_preserving_sae import BatchTopKSAE
from exp39d_leave_one_out import NoC_SAE

DEVICE = "cuda"
out_dir = Path("experiments/exp42c_saebench_results")
out_dir.mkdir(parents=True, exist_ok=True)
summary = {}
for vname, cls in [("standard", BatchTopKSAE), ("no_C", NoC_SAE)]:
    sae = cls(2304, 16384, 80).to(DEVICE)
    sd = torch.load(f"checkpoints/exp42c/{vname}_L13.pt", map_location=DEVICE)
    sae.load_state_dict({k: v.float() for k, v in sd.items()})
    sae.eval()
    bench = wrap_for_saebench(sae, vname, 13)
    t0 = time.time()
    run_saebench(
        bench, sae_name=f"exp42c_{vname}_16k_L13",
        eval_types=["core", "sparse_probing"],
        output_dir=str(out_dir),
        llm_batch_size=8, llm_dtype="bfloat16", device=DEVICE,
    )
    summary[vname] = {"elapsed_s": time.time() - t0}
    del sae, bench
    gc.collect()
    torch.cuda.empty_cache()
with open(out_dir / "exp42c_saebench_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DONE")
