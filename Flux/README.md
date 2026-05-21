# Flux — FLUX.1-schnell Experiments

This sub-module contains the FLUX.1-schnell implementation of the **DP** method described in the CVPR 2026 paper. See the [main README](../README.md) for installation instructions and method details.

---

## Why FLUX.1-schnell?

Rather than applying concept erasure directly to the full FLUX.1-dev model, we use **FLUX.1-schnell** — a distilled variant that requires only **4 inference steps** compared to 20–50 for the base model. This provides a substantial **5–10× acceleration** in image generation speed, making large-scale concept erasure evaluations (1,000 images × 10 concepts × 2 methods) practical on a single GPU. The weight-edit approach (DP/UCE) operates identically on the transformer's attention layers regardless of the distillation, so the erasure results transfer faithfully.

Key FLUX-specific implementation notes:
- **Dual text encoders**: FLUX uses both a T5-XXL encoder (4096-dim, per-token) and a CLIP encoder (768-dim, pooled). The DP algorithm is applied separately to `context_embedder` (T5 path) and `text_embedder.linear_1` (CLIP path).
- **No classifier-free guidance**: FLUX.1-schnell is guidance-free (`guidance_scale=0.0`).
- **Token layout**: T5 tokenizer has no BOS token; valid content tokens are indices `0 … (N_valid - 2)`, excluding the final EOS.

---

## Usage

### Step 0 — Generate Baseline Images (required once)

```bash
cd Flux
nohup sh generate_original_flux.sh > original.log 2>&1 &
```

### Step 1–3 — Run Full Pipeline per Concept

```bash
# DP (our method) — runs on GPU:1
nohup sh scripts/Double_CassettePlayer_flux.sh > Double_CassettePlayer.log 2>&1 &

# UCE (baseline) — runs on GPU:3
nohup sh scripts/UCE_CassettePlayer_flux.sh > UCE_CassettePlayer.log 2>&1 &
```


---

## Results on FLUX.1-schnell — ImageNette (10 concepts)

Metric definitions:
- **Erased Acc. ↓** — top-1 semantic accuracy on the target concept after erasure (lower is better)
- **Other Cls. Drop ↑** — average accuracy drop across the 9 non-target classes (lower absolute value is better; negative means slight improvement)

| Concept | UCE Erased Acc. ↓ | UCE Other Cls. Drop | DP Erased Acc. ↓ | DP Other Cls. Drop |
|---|---|---|---|---|
| Cassette Player | 2.0% | 1.9% | **0.0%** | **-0.2%** |
| Chain Saw | **0.0%** | 2.4% | **0.0%** | **1.6%** |
| Church | 46.0% | 2.2% | **9.0%** | **-0.3%** |
| English Springer | **0.0%** | 3.8% | **0.0%** | **2.8%** |
| French Horn | **0.0%** | 3.4% | **0.0%** | **1.2%** |
| Garbage Truck | 14.0% | 3.0% | **0.0%** | **0.3%** |
| Gas Pump | 95.0% | 2.1% | **0.0%** | **-0.2%** |
| Golf Ball | 100.0% | 2.9% | **1.0%** | **-0.3%** |
| Parachute | **0.0%** | 2.9% | **0.0%** | **1.1%** |
| Tench | **0.0%** | 2.7% | **0.0%** | **0.7%** |



## Data

Prompts are shared with the SD experiments: `../SD/data/small_imagenet_prompts.csv`.
