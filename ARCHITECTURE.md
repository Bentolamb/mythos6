# Fable/Mythos 6 — Technical Architecture & Implementation Plan

## 0. Honest scope statement (read this first)

The original brief frames Mythos 6 as competing with frontier models (GPT/Claude/Gemini-class).
That is not achievable here, and this document does not pretend otherwise. Frontier pretraining
runs are O(10,000–100,000) GPU-days on datacenter accelerators. This project's actual budget is:

- **Training compute:** Kaggle free tier (2× T4, 16GB each, ~30 GPU-hours/week quota, 12h session
  cap, no NVLink) and/or Colab (free single T4, or Pro/Pro+ credits toward an A100 40GB). Compute
  availability is treated as unreliable and bursty — the pipeline must checkpoint/resume across
  sessions, not assume a continuous multi-day run.
- **Realistic token budget:** on the order of **1–5B training tokens** for the first real runs
  (see §6.3 for the derivation). SmolLM2 and Qwen2.5's small models were trained on **trillions**
  of tokens using lab-scale clusters. We cannot out-token them. We can only out-*curate* them on a
  per-token basis, and target specific benchmark slices rather than broad parity.
- **Agreed success criterion** (per user decision): train a small (~100M–1.5B total parameter)
  efficiency-first model from scratch and benchmark it against same-scale open peers — SmolLM2
  (135M/360M), Qwen2.5 (0.5B/1.5B), Pythia (410M/1B), TinyLlama (1.1B) — with an honest accounting
  of quality-per-FLOP, quality-per-VRAM, and quality-per-token-trained. Beating them on *every*
  benchmark is unlikely given the token-budget gap above; beating them on a defensible subset
  (code, math/reasoning-with-curated-data, instruction following) by spending the compute budget
  on **data quality over data quantity** is the actual bet this project is making, following the
  precedent set by the Phi model family (curated/synthetic "textbook-quality" data punching above
  its token-count weight).
- Every number in this document below is a *design estimate*, not a measured result, until a
  benchmark in `MILESTONES.md` says otherwise. Anything marked **[experimental]** has not been
  validated and should not be relied on.

## 1. Repository layout

```
mythos6/
  ARCHITECTURE.md        this file
  MILESTONES.md          phased plan with exit criteria
  requirements.txt
  src/mythos/
    config.py            ModelConfig dataclass + named presets
    model.py             transformer implementation (attention, MLP, block, causal LM head)
    moe.py                mixture-of-experts FFN layer (Milestone 3+)
    data.py               streaming dataset / tokenization pipeline (Milestone 1)
    train.py              training loop (AMP, grad accum, checkpointing)
    eval/                 benchmark harness (Milestone 7)
  scripts/
    smoke_test.py         tiny-config correctness check, runs on CPU or the local 4GB GPU
  notebooks/
    kaggle_train.ipynb    thin wrapper to run train.py on Kaggle (2x T4)
    colab_train.ipynb     thin wrapper to run train.py on Colab (T4 or A100)
```

## 2. Architecture decisions

All choices below favor techniques with existing, well-validated implementations (avoiding
hand-written CUDA kernels, which is out of scope for this budget) so effort goes into data
quality and ablation rigor rather than low-level kernel engineering.

| Component | Choice | Rationale |
|---|---|---|
| Backbone | Decoder-only Transformer | Standard, best tooling support (PyTorch SDPA, HF ecosystem) |
| Normalization | RMSNorm, pre-norm, + QK-norm on Q/K before RoPE | Cheaper than LayerNorm; QK-norm is a near-free stability win at small batch sizes reported by Gemma2/OLMo2 — **validate via loss-curve stability ablation, not assumed** |
| Attention | Grouped-Query Attention (GQA), 4 KV heads | Shrinks KV-cache ~3-4x vs MHA at negligible quality cost (established result); critical since VRAM is the binding constraint for both training (T4 16GB) and target inference tiers |
| Attention kernel | PyTorch `scaled_dot_product_attention` (memory-efficient / flash backend) | True FlashAttention-2 kernels are weakly supported on Turing (T4, sm75); PyTorch SDPA auto-selects the best available backend per-GPU (memory-efficient on T4, flash on A100) — avoids hand-rolling kernels |
| Long-context pattern | Alternating local sliding-window (window ≈2048) and full-attention layers, ratio 3:1 | Sub-quadratic compute at long context while periodic full layers preserve long-range retrieval — **[experimental]**, must pass needle-in-haystack eval before being trusted past training length |
| Position encoding | RoPE, trained at 4096 ctx, extended via NTK-aware/YaRN scaling | Cheap to implement (no learned params), good extrapolation track record | Extension beyond trained length is **[experimental]** until validated |
| FFN | SwiGLU, d_ff ≈ 2.67×d_model (matches GELU-MLP param count) | Consistently outperforms ReLU/GELU MLP at equal params in published ablations |
| Sparsity | Dense baseline first; standard top-2-of-8 MoE FFN as Milestone 3 | MoE only kept if it beats the dense model at matched **active**-param compute — this is a benchmarked decision, not an assumption (see §6.2) |
| Embeddings | Tied input/output, vocab 49,152 (BPE, trained on our corpus) | Tying saves ~15-45% of params at these small scales; vocab size matches SmolLM2 for a fair comparison baseline |
| Quantization target | INT4 weight-only (GPTQ/AWQ) + INT8 KV-cache for inference | Must be benchmarked for quality degradation per §7, never assumed lossless |

## 3. Candidate configs (design estimates — to be measured, not trusted yet)

Formulas: attention params/layer = `2·d_model² + 2·d_model·(n_kv_heads·d_head)`; SwiGLU MLP
params/layer = `3·d_model·d_ff`; embedding = `vocab·d_model` (tied).

| Config | d_model | layers | heads (kv) | d_ff | Total params | Active params/tok | FLOPs/tok (fwd, approx 2×active) |
|---|---|---|---|---|---|---|---|
| `mythos6-130m-dense` | 768 | 12 | 12 (4) | 2048 | **113M** | 113M | ~226M |
| `mythos6-320m-dense` | 1024 | 24 | 16 (4) | 2816 | **321M** | 321M | ~642M |
| `mythos6-moe-a320m` | 1024 | 24 | 16 (4) | 8×1408 experts, top-2 | **944M total** | **321M** | ~642M |

The MoE config is deliberately matched in *active* params to `mythos6-320m-dense` — same
inference-time compute, ~3x more total capacity. Milestone 3 exists specifically to test whether
that extra capacity converts to measurably better quality, per token of training compute spent on
each. If it doesn't clear the bar, we ship the dense model — no sunk-cost attachment to MoE.

### KV-cache footprint (fp16), the actual long-context bottleneck

`bytes/token = 2 (K&V) × n_layers × n_kv_heads × d_head × 2 bytes`

| Config | bytes/token | @ 8K ctx | @ 32K ctx | @ 128K ctx |
|---|---|---|---|---|
| 130m-dense | 12,288 B | 96 MB | 384 MB | 1.5 GB |
| 320m-dense / moe-a320m | 24,576 B | 192 MB | 768 MB | 3.1 GB |

GQA with 4 KV heads is what makes the "128K context on modest VRAM" claim survivable at all —
worth stating plainly since it's the load-bearing decision behind the long-context goal. With
INT8 KV-cache quantization (also to be benchmarked, not assumed lossless) these numbers halve
again.

## 4. Hardware tiers (reframed around actual access)

### Training tiers
| Tier | Hardware | Notes |
|---|---|---|
| Kaggle free | 2× T4 16GB, no NVLink | ~30 GPU-hr/week quota (2 GPUs = 2x quota burn per wall-clock hour), 12h session cap — pipeline must checkpoint/resume |
| Colab free | 1× T4 16GB | Unpredictable session length/disconnects, no guaranteed weekly hours |
| Colab Pro/Pro+ | 1× A100 40GB (when available) | Best per-session throughput; billed in compute units, not unlimited |

### Inference tiers (post-training, post-quantization)
| Tier | Example hardware | What fits |
|---|---|---|
| Extreme | Multi-GPU workstation | Full fp16 `moe-a320m`, huge batch/context, speculative decoding |
| High-end | Single 24GB+ GPU (A100/4090-class) | Full fp16 any candidate config, long context |
| Enthusiast | 8-12GB consumer GPU | INT8 weights, any candidate config, moderate context |
| **Minimum (this machine)** | **GTX 1050 Ti, 4GB VRAM** | INT4 `130m-dense` or `320m-dense` weights (~65-160MB), several thousand tokens of context — this is the real floor test, run locally |

The local 1050 Ti stops being "too weak to matter" and becomes the actual acceptance test for the
"minimum tier" row — if quantized Mythos 6 doesn't run acceptably there, the minimum-tier claim in
this doc is wrong and gets corrected, not fudged.

## 5. Data pipeline (Milestone 1)

Given the token-budget reality in §0, the entire strategy leans on curation quality:

1. **Sources** (all pulled via HF `datasets` streaming, no bulk local storage needed): a FineWeb-Edu
   or Cosmopedia-style filtered web/synthetic-textbook mix for general text, a code subset (The
   Stack / StarCoder-data, filtered), a math/reasoning subset (OpenWebMath-style), and a small
   curated instruction set held out for Milestone 5.
2. **Dedup:** MinHash-LSH near-dedup + exact-hash dedup before any training run.
3. **Contamination check:** n-gram overlap scan of the training corpus against every benchmark in
   `MILESTONES.md` §7 *before* training starts; contaminated documents are dropped, not the
   benchmark. This runs as an automated script and its output is a gate, not a formality.
4. **Tokenizer:** BPE, vocab 49,152, trained on a representative sample of the deduped corpus.

## 6. Training pipeline (Milestone 2+)

### 6.1 Loop mechanics
Mixed precision (fp16 + dynamic loss scaling on T4, which lacks native bf16 tensor cores; bf16
directly on A100), gradient accumulation to reach an effective batch size independent of
per-step microbatch/VRAM limits, checkpoint every N steps to survive Kaggle/Colab session death,
resumable optimizer state.

### 6.2 Ablation discipline
Every architectural claim in §2 marked for validation gets an actual ablation run at the smallest
config that can show a signal (usually `130m-dense`) before being trusted at larger scale:
dense-vs-MoE at matched active FLOPs, QK-norm on/off, sliding-window-ratio sweep, RoPE-scaling
method comparison. Results and configs get logged, not just the final number.

### 6.3 Token budget derivation (the number behind §0's honesty warning)
Conservative sustained throughput estimate for a single T4 running this stack (no custom kernels,
realistic MFU for a small unoptimized model): **~15 TFLOP/s effective**. With, say, 60 GPU-hours
accumulated over several weeks of Kaggle/Colab access:

`total FLOPs ≈ 60 hr × 15e12 FLOP/s × 3600 s/hr ≈ 3.2e18 FLOPs`

Training FLOPs ≈ `6 × N_active × D_tokens` (standard approximation). For `N_active = 320M`:

`D_tokens ≈ 3.2e18 / (6 × 3.2e8) ≈ 1.7B tokens`

That's roughly **5 tokens/param**, far below Chinchilla-optimal (~20/param) and orders of
magnitude below what SmolLM2/Qwen2.5 used. This is the actual, math-backed reason §0 says we
compete on curation, not scale — and it's why Milestone 2 explicitly budgets multiple *small*
config runs (130m first) rather than one long run at the biggest config, since compute-starved
regimes get more signal per GPU-hour from smaller models.

## 7. Evaluation plan
See `MILESTONES.md` §7 for the full benchmark suite, peer models, and the quality/FLOP,
quality/VRAM, quality/watt, tokens/sec tracking table. Independent held-out benchmarks only —
no training on eval-adjacent data (enforced by the contamination check in §5.3).

## 8. Recursive improvement (Mythos 6 → 7)
Deferred until Milestone 7 produces real numbers. Any proposed Mythos 7 change must cite the
specific ablation or benchmark result that motivates it — "a model suggested this" is not a
justification accepted by this pipeline.
