#!/usr/bin/env bash
# Run from your local shell to pull all outstanding exp40 results from the A100
# and finalize the analysis. The Bash tool failed in the Claude session that
# launched the chain, so 40c L27 results + 40g (SAEBench) outputs never made it
# back. Everything 40a/40b/40d/40f and 40c-L18 is already local.
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

REMOTE=<user>@<ip>

echo "=== exp40g status ==="
ssh "$REMOTE" "cd ~/MechInter--RNH && tail -40 experiments/exp40g_watcher.log 2>/dev/null; echo --- 40g output ---; tail -80 experiments/exp40g_output.log 2>/dev/null; echo --- saebench dir ---; ls -la experiments/exp40g_saebench_results/ 2>/dev/null"

echo "=== pulling files ==="
ssh "$REMOTE" "cd ~/MechInter--RNH && tar cf - \
    experiments/exp40c_scale_multilayer.json \
    experiments/exp40c_output.log \
    experiments/exp40g_watcher.log \
    experiments/exp40g_output.log \
    experiments/exp40g_saebench_results/ \
    2>/dev/null" | tar xf -

echo "=== exp40c L27 numbers ==="
python3 -c '
import json
d = json.load(open("experiments/exp40c_scale_multilayer.json"))
for layer, res in d.get("by_layer", {}).items():
    print(f"L{layer}:")
    for v, r in res.items():
        rec = r.get("reconstruction", {})
        fd = r.get("freq_diag", {}).get("alive_counts", {})
        print(f"  {v}: FVE={rec.get(\"fve\",0):.4f} cos={rec.get(\"cos_recon\",0):.4f} "
              f"dead={rec.get(\"dead_frac\",0)*100:.1f}% alive_gt1e-4={fd.get(\"gt_1e-4\",0)}")
'

echo "=== to run exp40e (LLM auto-interp) ==="
echo "Set ANTHROPIC_API_KEY in ~/.config/.env.global, then:"
echo "  .venv/bin/python -u experiments/exp40e_llm_interp.py"
