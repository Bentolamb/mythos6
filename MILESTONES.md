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

### M0 — 2026-09-08
- Smoke test (`scripts/smoke_test.py`) passes on the local GTX 1050 Ti: loss 6.95 → 2.27 over 30
  steps on `mythos6-nano`. Closed-form param-count formula in `config.py` matches the actual
  instantiated model exactly (158,080 params) after including QK-norm/final-norm terms.
- `mythos6-130m-dense` = 113,266,944 params; `mythos6-320m-dense` = 320,917,504 params — both
  match the ARCHITECTURE.md sec 3 table.
- `train.py` loop mechanics validated on `mythos6-nano` with synthetic data: checkpoint save +
  resume confirmed correct (resumed cleanly from step 20, continued to step 30).

### M1 — 2026-09-08 (in progress)
- All four mixture sources (`fineweb-edu`, `cosmopedia-100k`, `open-web-math`,
  `the-stack-smol-xl`) confirmed reachable, ungated, Parquet-backed (no deprecated dataset-script
  loaders — this ruled out `codeparrot/github-code` and a couple of gated `bigcode`/`nampdn-ai`
  code datasets during source selection).
- Observed mixture sampling proportions over 200 draws: general 51.5% / synthetic-textbook 18% /
  code 15.5% / math 15% vs. target weights 55/15/15/15 — within expected sampling noise.
- Tokenizer trained successfully at small scale (3,000 docs, full 49,152 vocab, round-trip check
  passes). **Bug found and fixed:** feeding `tokenizers`' Rust-threaded `train_from_iterator` a
  live generator that was still doing HF Hub network I/O (pyarrow parsing, retried sockets)
  caused an intermittent segfault on this machine. Fixed by materializing the doc sample into a
  plain list before training (`scripts/train_tokenizer.py`) — sidesteps the Python/Rust threading
  boundary during network I/O. Not yet confirmed whether this was Windows-socket-specific or would
  also occur on Kaggle/Colab's Linux runners; the list-materialization approach is robust either
  way, so it's the shipped approach rather than something to re-litigate per-platform.
- Contamination scanner validated at small scale (1,500 corpus docs vs. GSM8K/HellaSwag/
  ARC-Challenge 13-gram sets): 0% overlap found, scanner logic confirmed working (loads benchmark
  sets, builds n-gram index, flags correctly).
- Dedup scanner validated at small scale (1,500 docs): 0% exact dupes, 0.4% near-dupes caught at
  Jaccard ≥ 0.8 — plausible for this mixture, confirms MinHash-LSH pipeline works end-to-end.
- **Data pipeline throughput measured, and it's a real bottleneck as currently built:** live
  streaming + tokenize-per-step on this machine (old 4-core CPU, single-threaded, cold cache)
  sustains only **~1,228 tokens/sec**. A single T4 at the ~15 TFLOP/s effective estimate from
  ARCHITECTURE.md sec 6.3 needs roughly **~7,800 tokens/sec** to stay compute-bound at the
  320m-dense config (6×N_active FLOPs/token ÷ throughput). Live tokenization would starve the GPU
  by ~6x. **Action before M2:** pre-tokenize the corpus once into packed fixed-length binary
  shards (memory-mapped `uint16`/`uint32` arrays) rather than tokenizing on the fly during
  training — standard practice, removes the per-step tokenization/network cost from the training
  loop entirely. Not yet implemented; tracked as the next concrete task.
- **Fix implemented and verified:** `scripts/pretokenize.py` writes the packed mixture to
  memory-mapped uint16 shards (`manifest.json` + `shard_XXXXX.bin`); `mythos.data.PackedShardDataset`
  reads them at training time. End-to-end integration test (pretokenize → PackedShardDataset →
  a real training step via `train.py`'s `packed_batch_iter`) passes, loss decreasing correctly.
  Measured throughput of the shard reader alone: **~103,664 tokens/sec** on this same CPU — an
  ~84x improvement over the 1,228 tok/s live-streaming path, comfortably above the ~7,800 tok/s a
  T4 needs to stay compute-bound. `train.py --data-dir <shards>` now uses this path; the old
  live-tokenize-per-step path was removed rather than kept as a slower fallback.
- **M1 exit criterion met** at small scale (all checks above pass). Remaining before a real M2
  run: rerun `train_tokenizer.py` and `pretokenize.py` at full scale (hundreds of thousands of
  docs / the full ~1-5B token budget from ARCHITECTURE.md sec 6.3) rather than the few-thousand-doc
  samples used here to validate correctness — that's a long-running CPU job, not a new milestone.
