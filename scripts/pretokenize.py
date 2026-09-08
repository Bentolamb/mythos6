"""Pre-tokenize the training mixture into fixed-length, memory-mapped shards.

Fixes the throughput bottleneck found in MILESTONES.md M1: live tokenize-per-step during training
sustains only ~1,228 tokens/sec on this machine's CPU, vs. the ~7,800 tokens/sec a single T4 needs
to stay compute-bound at the 320m-dense config. Pre-tokenizing once and memory-mapping during
training removes tokenization and network I/O from the training loop's critical path entirely.

Output: `<out-dir>/shard_XXXXX.bin` -- raw uint16 token ids (vocab_size=49152 fits in uint16),
each shard `shard_size` sequences of `seq_len` tokens, contiguous and directly np.memmap-able.
Plus `<out-dir>/manifest.json` recording seq_len, dtype, vocab_size, and per-shard sequence counts.

Usage: python scripts/pretokenize.py --tokenizer artifacts/tokenizer.json --out-dir data/packed \
           --n-sequences 500000 --seq-len 4096
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from tokenizers import Tokenizer

from mythos.data import build_training_stream

DTYPE = np.uint16  # valid up to vocab_size 65536; assert-checked against the tokenizer below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--n-sequences", type=int, required=True, help="total packed sequences to write")
    ap.add_argument("--sequences-per-shard", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size <= np.iinfo(DTYPE).max + 1, (
        f"tokenizer vocab_size={vocab_size} exceeds {DTYPE} range; widen DTYPE"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream = build_training_stream(tokenizer, args.seq_len, seed=args.seed)

    shard_sizes = []  # sequences actually written per shard
    n_written = 0
    shard_idx = 0
    t0 = time.time()

    while n_written < args.n_sequences:
        n_this_shard = min(args.sequences_per_shard, args.n_sequences - n_written)
        buf = np.empty((n_this_shard, args.seq_len), dtype=DTYPE)
        for i in range(n_this_shard):
            buf[i] = next(stream)
        shard_path = out_dir / f"shard_{shard_idx:05d}.bin"
        buf.tofile(shard_path)
        shard_sizes.append(n_this_shard)
        n_written += n_this_shard
        shard_idx += 1
        elapsed = time.time() - t0
        rate = n_written * args.seq_len / elapsed
        print(f"shard {shard_idx-1}: wrote {n_this_shard} seqs -> {shard_path.name}  "
              f"(total {n_written}/{args.n_sequences} seqs, {rate:.0f} tok/s avg)")

    manifest = {
        "seq_len": args.seq_len,
        "dtype": str(np.dtype(DTYPE)),
        "vocab_size": vocab_size,
        "shard_sizes": shard_sizes,
        "total_sequences": n_written,
        "total_tokens": n_written * args.seq_len,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {n_written} sequences ({n_written * args.seq_len:,} tokens) across "
          f"{len(shard_sizes)} shards in {out_dir}. Manifest written to {out_dir/'manifest.json'}.")


if __name__ == "__main__":
    main()
