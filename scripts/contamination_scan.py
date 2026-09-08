"""13-gram overlap contamination check between the training mixture and every eval benchmark
in MILESTONES.md sec 7 (GSM8K, HellaSwag, ARC-Challenge here; extend as the eval suite grows in
M7). This is a gate, not a formality -- per ARCHITECTURE.md sec 5.3, any corpus document that
overlaps a benchmark gets DROPPED from training, the benchmark set is never touched.

13-gram word overlap is the standard contamination-check granularity used by GPT-3/PaLM-style
reports: short enough to catch paraphrased-but-copied passages, long enough that coincidental
overlap is negligible.

Usage: python scripts/contamination_scan.py [--n-corpus-docs 5000]
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset

from mythos.data import iter_mixture_text

NGRAM_SIZE = 13


def ngrams(text: str, n: int = NGRAM_SIZE):
    words = text.split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_benchmark_ngrams() -> dict[str, set]:
    """Build the union of n-grams for each benchmark's held-out text. Returns {benchmark_name: ngram_set}."""
    benchmarks = {}

    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    ng = set()
    for ex in gsm8k:
        ng |= ngrams(ex["question"] + " " + ex["answer"])
    benchmarks["gsm8k"] = ng

    hellaswag = load_dataset("Rowan/hellaswag", split="validation")
    ng = set()
    for ex in hellaswag:
        ng |= ngrams(ex["ctx"])
        for ending in ex["endings"]:
            ng |= ngrams(ending)
    benchmarks["hellaswag"] = ng

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ng = set()
    for ex in arc:
        ng |= ngrams(ex["question"])
    benchmarks["arc_challenge"] = ng

    return benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-corpus-docs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading benchmark n-gram sets...")
    benchmarks = load_benchmark_ngrams()
    for name, ng in benchmarks.items():
        print(f"  {name}: {len(ng)} unique {NGRAM_SIZE}-grams")

    print(f"\nScanning {args.n_corpus_docs} corpus documents for overlap...")
    hits = {name: 0 for name in benchmarks}
    n_scanned = 0

    stream = iter_mixture_text(seed=args.seed)
    for category, text in itertools.islice(stream, args.n_corpus_docs):
        n_scanned += 1
        doc_ngrams = ngrams(text)
        if not doc_ngrams:
            continue
        for name, bench_ngrams in benchmarks.items():
            if doc_ngrams & bench_ngrams:
                hits[name] += 1

    print(f"\nscanned {n_scanned} corpus docs against {len(benchmarks)} benchmarks:")
    total_flagged = 0
    for name, count in hits.items():
        print(f"  {name:15s} {count:5d} contaminated docs ({count/n_scanned:.3%})")
        total_flagged += count
    if total_flagged == 0:
        print("\nPASS: zero contamination detected in this sample.")
    else:
        print(f"\n{total_flagged} contaminated docs found -- these must be dropped from the "
              f"training corpus before M2, not the benchmark sets.")


if __name__ == "__main__":
    main()
