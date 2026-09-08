"""Train the 49,152-vocab byte-level BPE tokenizer on a sample of the training mixture.
See ARCHITECTURE.md sec 2/5.

Usage: python scripts/train_tokenizer.py [--n-docs 200000] [--out artifacts/tokenizer.json]
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

from mythos.data import EOS_TOKEN, iter_mixture_text

VOCAB_SIZE = 49152
SPECIAL_TOKENS = [EOS_TOKEN, "<|pad|>", "<|unk|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=200_000,
                     help="number of mixture documents to train the tokenizer on")
    ap.add_argument("--out", type=str, default="artifacts/tokenizer.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # Materialize documents into a plain list before handing them to the trainer: the tokenizers
    # library consumes the iterator from Rust-managed threads, and feeding it a live generator
    # that's still doing HF Hub network I/O (pyarrow parsing, retried sockets) caused an
    # intermittent segfault during testing on this machine. A plain in-memory list sidesteps the
    # Python/Rust threading boundary during network I/O entirely, at the cost of holding
    # n_docs documents in memory (a few GB at most at the doc counts used here).
    print(f"Fetching {args.n_docs} mixture docs...")
    docs = []
    stream = iter_mixture_text(seed=args.seed)
    for i, (_category, text) in enumerate(itertools.islice(stream, args.n_docs)):
        if i % 10_000 == 0:
            print(f"  ...{i}/{args.n_docs} docs", flush=True)
        docs.append(text)

    print(f"Training BPE tokenizer, vocab_size={VOCAB_SIZE}, on {len(docs)} mixture docs...")
    tokenizer.train_from_iterator(docs, trainer=trainer)

    tokenizer.save(str(out_path))
    print(f"Saved tokenizer to {out_path}, actual vocab size = {tokenizer.get_vocab_size()}")

    # round-trip sanity check
    sample = "The quick brown fox jumps over the lazy dog. def foo(x): return x**2  ∑_{i=1}^n"
    ids = tokenizer.encode(sample).ids
    back = tokenizer.decode(ids)
    assert back == sample, f"round-trip mismatch:\n  in:  {sample!r}\n  out: {back!r}"
    print(f"Round-trip check passed ({len(ids)} tokens for {len(sample)} chars).")


if __name__ == "__main__":
    main()
