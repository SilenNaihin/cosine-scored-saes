"""SAEBench (core + sparse_probing) on exp42c-long intermediate checkpoints
50M / 100M / 150M to chart how no_C 16k closes (or doesn't) the gap to
published 16k 500M BatchTopK as training scale increases."""
import sys, gc, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from benchmarks.adapter import BenchSAE
from benchmarks.run_saebench import run_saebench
from exp40g_saebench import wrap_for_saebench
from exp39d_leave_one_out import NoC_SAE

DEVICE = "cuda"
out_dir = Path("experiments/exp42c_long_saebench_results")
out_dir.mkdir(parents=True, exist_ok=True)

# Find all intermediate checkpoints
ckpt_dir = Path("checkpoints/exp42c_long")
ckpts = sorted(ckpt_dir.glob("no_C_L13_step*_tok*.pt")) + [ckpt_dir / "no_C_L13_150M.pt"]
ckpts = [c for c in ckpts if c.exists()]
print(f"Found {len(ckpts)} checkpoints to eval:")
for c in ckpts:
    print(f"  {c.name}")

summary = {}
for c in ckpts:
    # Extract token count from filename
    name = c.stem
    if "step" in name:
        # no_C_L13_step12207_tok50M
        tag = name.split("tok")[-1]
    else:
        tag = "150M"
    sae_name = f"exp42c_long_no_C_16k_{tag}"

    sae = NoC_SAE(2304, 16384, 80).to(DEVICE)
    sd = torch.load(c, map_location=DEVICE)
    sae.load_state_dict({k: v.float() for k, v in sd.items()})
    sae.eval()
    bench = wrap_for_saebench(sae, "no_C", 13)

    print(f"\n=== {sae_name} ===")
    t0 = time.time()
    run_saebench(
        bench, sae_name=sae_name,
        eval_types=["core", "sparse_probing"],
        output_dir=str(out_dir),
        llm_batch_size=8, llm_dtype="bfloat16", device=DEVICE,
    )
    summary[sae_name] = {"elapsed_s": time.time() - t0, "tag": tag,
                          "checkpoint": str(c)}
    del sae, bench
    gc.collect()
    torch.cuda.empty_cache()

with open(out_dir / "exp42c_long_saebench_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DONE")
