"""Exact-hash + MinHash-LSH near-dedup over a mixture sample. See ARCHITECTURE.md sec 5 / MILESTONES.md M1.

This is real, runnable dedup logic (not a stub) intended to run over the actual corpus shard
selected for a training run -- at our ~1-5B token budget the document count is in the low
millions, which is tractable for MinHash-LSH on CPU without needing a distributed job. For a
quick correctness check it's run here on a small sample; the full-corpus pass is a longer CPU-only
job (no GPU needed) that can run locally overnight or as a Kaggle/Colab CPU-only step before a
training run starts.

Usage: python scripts/dedup_scan.py [--n-docs 5000] [--threshold 0.8]
"""
import argparse
import hashlib
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasketch import MinHash, MinHashLSH

from mythos.data import iter_mixture_text

NUM_PERM = 128
SHINGLE_SIZE = 5  # word shingles


def shingles(text: str, k: int = SHINGLE_SIZE):
    words = text.split()
    if len(words) < k:
        yield " ".join(words)
        return
    for i in range(len(words) - k + 1):
        yield " ".join(words[i:i + k])


def exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity threshold for near-dup")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lsh = MinHashLSH(threshold=args.threshold, num_perm=NUM_PERM)
    seen_exact: set[str] = set()

    n_exact_dupes = 0
    n_near_dupes = 0
    n_kept = 0
    n_total = 0

    stream = iter_mixture_text(seed=args.seed)
    for i, (category, text) in enumerate(itertools.islice(stream, args.n_docs)):
        n_total += 1
        h = exact_hash(text)
        if h in seen_exact:
            n_exact_dupes += 1
            continue
        seen_exact.add(h)

        mh = MinHash(num_perm=NUM_PERM)
        for sh in shingles(text):
            mh.update(sh.encode("utf-8", errors="ignore"))

        key = f"doc-{i}"
        if lsh.query(mh):
            n_near_dupes += 1
            continue
        lsh.insert(key, mh)
        n_kept += 1

        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{args.n_docs}  kept={n_kept} exact_dup={n_exact_dupes} near_dup={n_near_dupes}")

    print(f"\nscanned {n_total} docs:")
    print(f"  kept          {n_kept:6d}  ({n_kept/n_total:.2%})")
    print(f"  exact dupes   {n_exact_dupes:6d}  ({n_exact_dupes/n_total:.2%})")
    print(f"  near dupes    {n_near_dupes:6d}  ({n_near_dupes/n_total:.2%})  (Jaccard >= {args.threshold})")


if __name__ == "__main__":
    main()
