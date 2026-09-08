# Mythos 6 — Milestones

Each milestone has an explicit exit criterion. No milestone is "done" on vibes — it's done when
its benchmark/check passes and the result is recorded here (a results table gets appended to this
file as runs complete).

## M0 — Repo scaffold + smoke test (local, this machine)
- Minimal model implementation (`src/mythos/model.py`) with the M2-decided architecture (RoPE,
  GQA, SwiGLU, RMSNorm, tied embeddings), config-driven so `moe.py` can swap the FFN later.
- `scripts/smoke_test.py`: instantiate the smallest config (a few M params), run a handful of
  optimizer steps on synthetic random-token data on CPU or the local GTX 1050 Ti, assert loss
  decreases and no NaNs. This validates the code is correct before spending any Kaggle/Colab
  quota on it.
- **Exit criterion:** smoke test passes reproducibly.

## M1 — Tokenizer + data pipeline
- Train the 49,152-vocab BPE tokenizer on a sample of the curated corpus.
- Build the streaming dataset loader (`src/mythos/data.py`) over the HF-hosted sources named in
  `ARCHITECTURE.md` §5.
- Implement dedup (MinHash-LSH + exact hash) and the contamination scanner against every eval
  benchmark listed in §7 below.
- **Exit criterion:** contamination scanner reports near-zero overlap between training corpus and
  eval sets; tokenizer round-trips correctly; data loader sustains throughput without stalling the
  GPU (measured, not assumed).

## M2 — Dense baseline training run(s)
- Train `mythos6-130m-dense` to convergence within the realistic token budget derived in
  `ARCHITECTURE.md` §6.3, on Kaggle (2×T4) and/or Colab.
- Then `mythos6-320m-dense` at the same per-token compute recipe, to get a 2-point mini
  scaling-law reference.
- Run the QK-norm on/off ablation and the sliding-window-ratio sweep here, at the 130m scale,
  since it's cheap and this is where the signal-per-GPU-hour is highest.
- **Exit criterion:** both dense models trained, loss curves and ablation results logged in this
  file, no NaN/divergence, checkpoints saved and reloadable.

## M3 — MoE ablation
- Train `mythos6-moe-a320m` (top-2-of-8, matched active params to `320m-dense`) on the same token
  budget as the 320m dense run.
- Direct comparison: does MoE beat dense at matched active-param compute, on held-out validation
  loss and on the benchmark suite (§7)? This decides whether Mythos 6's shipped model is MoE or
  dense — **the benchmark decides, not a preference for sparsity**.
- **Exit criterion:** head-to-head result recorded; a model is chosen to carry forward.

## M4 — Long context
- Extend the chosen model's context via RoPE scaling (NTK-aware or YaRN) beyond its 4096 training
  length.
- Validate with a needle-in-haystack retrieval eval at 8K/32K/128K context, not just a claimed
  context length.
- **Exit criterion:** retrieval accuracy reported at each context length; if it degrades badly
  past some point, that becomes the model's *honest* max useful context, not the theoretical one.

## M5 — Instruction tuning
- SFT on the curated instruction set held out in M1.
- Light preference optimization (DPO on a small curated preference set) if time/compute allows —
  labeled **[stretch]**, not required for M5 to close.
- **Exit criterion:** instruction-following benchmark score recorded; model responds coherently to
  held-out prompts (spot-checked, not just scored).

## M6 — Quantization
- INT4 weight-only (GPTQ or AWQ) and INT8 KV-cache quantization of the M5 model.
- Benchmark quality degradation vs fp16 baseline explicitly — quantization is not assumed free.
- **Run inference on the local GTX 1050 Ti (4GB)** as the literal "Minimum tier" acceptance test
  from `ARCHITECTURE.md` §4: does it load, does it generate at a usable tokens/sec, at what
  context length before OOM.
- **Exit criterion:** quality-degradation numbers and local-GPU tokens/sec recorded.

## M7 — Full evaluation vs peers
Benchmark suite (independent/held-out where possible):
- Reasoning: ARC, HellaSwag
- Math: GSM8K, a curated arithmetic/word-problem set
- Code: HumanEval (subset appropriate to model scale)
- Factual/general knowledge: MMLU (subset)
- Instruction following: IFEval or equivalent
- Long-context retrieval: needle-in-haystack (from M4)
- Hallucination resistance: a curated refusal/uncertainty probe set

Peer models: SmolLM2-135M/360M, Qwen2.5-0.5B/1.5B, Pythia-410M/1B, TinyLlama-1.1B, run through the
same harness for a fair comparison.

Tracked metrics table (populated here once measured): quality/FLOP, quality/VRAM, quality/watt,
tokens/sec (generation), prompt tokens/sec (prefill), memory usage, KV-cache size, latency.

- **Exit criterion:** comparison table complete; honest writeup of where Mythos 6 wins, ties, and
  loses against each peer — a loss is a valid, reportable outcome.

## M8 — Recursive improvement proposal (Mythos 6 → 7)
- Using only the M7 results, identify the weakest benchmark category(ies).
- Propose specific, falsifiable architectural or data changes to address them.
- Each proposal must reference the M7/ablation evidence that motivates it.
- **Exit criterion:** a written Mythos 7 proposal doc, explicitly out of scope to implement until
  approved.

---

## Results log
*(Empty until M0 runs. Populated as milestones close — do not backfill estimates as if measured.)*
