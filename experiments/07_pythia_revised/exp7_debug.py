"""Quick debug: why is exp7 producing empty results?"""
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

DEVICE = "cuda"
MODEL_NAME = "pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
LAYER = 3

# Load 10 samples
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
texts = []
for row in ds:
    if len(row["text"]) > 100:
        texts.append(row["text"][:512])
    if len(texts) >= 10:
        break

model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"blocks.{LAYER}.hook_resid_post", device=DEVICE)

# Collect activations
hook_name = f"blocks.{LAYER}.hook_resid_post"
tokens = model.to_tokens(texts[:2], prepend_bos=True)[:, :64]
with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
acts = cache[hook_name]  # (batch, seq, d_model)
print(f"Activation shape: {acts.shape}, dtype: {acts.dtype}")

# Encode with SAE
flat_acts = acts.reshape(-1, acts.shape[-1])
encoded = sae.encode(flat_acts[:100])
print(f"Encoded shape: {encoded.shape}, dtype: {encoded.dtype}")
print(f"Encoded nonzero per token: {(encoded > 0).sum(dim=-1).float().mean():.1f}")
print(f"Encoded max: {encoded.max():.4f}")

# Pick one feature with nonzero activations
feat_sums = encoded.sum(dim=0)
top_feat = feat_sums.argmax().item()
n_active = (encoded[:, top_feat] > 0).sum().item()
print(f"\nTest feature {top_feat}: n_active={n_active}")

# Get feature direction
feature_dir = sae.W_dec[top_feat]
print(f"W_dec shape: {sae.W_dec.shape}")
print(f"Feature dir norm: {feature_dir.norm():.4f}")
feature_dir = feature_dir / feature_dir.norm()

# Find an active token
active_idx = (encoded[:, top_feat] > 0).nonzero()[0].item()
x = flat_acts[active_idx]
print(f"Activation norm: {x.norm():.4f}")

# Try ablation
projection = (x @ feature_dir) * feature_dir
x_ablated = x - projection
print(f"Projection norm: {projection.norm():.6f}")
print(f"Original norm: {x.norm():.4f}, Ablated norm: {x_ablated.norm():.4f}")

# Run through model with hooks
bos = model.to_tokens("", prepend_bos=True)[:, :1]
print(f"BOS shape: {bos.shape}")

def orig_hook(act, hook):
    print(f"  orig_hook called: act.shape={act.shape}")
    act[:, -1, :] = x
    return act

with torch.no_grad():
    orig_logits = model.run_with_hooks(bos, fwd_hooks=[(hook_name, orig_hook)])
print(f"Orig logits shape: {orig_logits.shape}, dtype: {orig_logits.dtype}")
orig_logits = orig_logits[:, -1, :].float()

def ablated_hook(act, hook):
    act[:, -1, :] = x_ablated
    return act

with torch.no_grad():
    ablated_logits = model.run_with_hooks(bos, fwd_hooks=[(hook_name, ablated_hook)])
ablated_logits = ablated_logits[:, -1, :].float()

print(f"Logit diff: {(orig_logits - ablated_logits).abs().mean():.6f}")
print(f"Logit range orig: [{orig_logits.min():.2f}, {orig_logits.max():.2f}]")
print(f"Logit range ablated: [{ablated_logits.min():.2f}, {ablated_logits.max():.2f}]")

orig_probs = torch.softmax(orig_logits, dim=-1)
ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)

print(f"orig_probs min: {orig_probs.min():.2e}, zeros: {(orig_probs == 0).sum().item()}")
print(f"orig_probs.log() has nan: {torch.isnan(orig_probs.log()).any()}")
print(f"orig_probs.log() has -inf: {torch.isinf(orig_probs.log()).any()}")

kl_per_token = orig_probs * (orig_probs.log() - ablated_log_probs)
print(f"KL per-token has nan: {torch.isnan(kl_per_token).any()}")
print(f"KL per-token has inf: {torch.isinf(kl_per_token).any()}")

kl = torch.sum(kl_per_token).item()
print(f"\nFinal KL: {kl}")
print(f"KL is nan: {np.isnan(kl)}, KL < 0: {kl < 0}")
