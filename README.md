# Mythos 6

Efficiency-first small language model project. **Read `ARCHITECTURE.md` first** — it states the
honest scope (this targets Kaggle/Colab-tier compute, not frontier scale) before anything else.
`MILESTONES.md` tracks phased progress and a results log of everything actually measured so far.

## Local setup (code validation / smoke tests only — not real training)

```
pip install -r requirements.txt
python scripts/smoke_test.py
```

## Real training runs

Happen on Kaggle or Colab, not this machine — see `notebooks/kaggle_train.ipynb` /
`notebooks/colab_train.ipynb`. Both need `REPO_URL` filled in with wherever this repo ends up
hosted (a `git clone`-able remote). Fill in the Kaggle notebook's accelerator setting per the note
at its top — `train.py` is currently single-GPU only, so a "2x T4" Kaggle instance wastes half its
quota until multi-GPU (DDP) support lands.

## Pipeline order

1. `scripts/train_tokenizer.py` — train the BPE tokenizer once
2. `scripts/contamination_scan.py` + `scripts/dedup_scan.py` — gate, not optional, before any real run
3. `scripts/pretokenize.py` — pack the mixture into memory-mapped shards (CPU-only, no GPU quota)
4. `python -m mythos.train --data-dir <shards> --preset <name> --resume` — the actual training loop
