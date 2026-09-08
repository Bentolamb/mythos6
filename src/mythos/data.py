"""Streaming data mixture + packing for Mythos 6. See ARCHITECTURE.md sec 5.

All sources are pulled via HF `datasets` streaming -- no bulk local download needed, which
matters both for the 29GB free on this machine's C: drive and for running unmodified on
Kaggle/Colab's small local disks. Every entry below was verified reachable (unauthenticated,
un-gated, Parquet-backed -- no deprecated dataset-script loaders) as of this pipeline's build.

Mixture weights are a first-pass judgment call, not a tuned result -- revisit once M2 loss curves
say something about over/under-represented sources.
"""
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset


@dataclass
class Source:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    text_field: str
    weight: float
    category: str  # for reporting / contamination-check bucketing


MIXTURE: list[Source] = [
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 0.55, "general"),
    Source("cosmopedia", "HuggingFaceTB/cosmopedia-100k", None, "train", "text", 0.15, "synthetic-textbook"),
    Source("open-web-math", "open-web-math/open-web-math", None, "train", "text", 0.15, "math"),
    Source("the-stack-smol-xl", "bigcode/the-stack-smol-xl", None, "train", "content", 0.15, "code"),
]

EOS_TOKEN = "<|endoftext|>"


def iter_source_text(source: Source) -> Iterator[str]:
    ds = load_dataset(source.hf_path, source.hf_config, split=source.split, streaming=True)
    for ex in ds:
        text = ex.get(source.text_field)
        if text:
            yield text


def iter_mixture_text(sources: list[Source] = MIXTURE, seed: int = 0) -> Iterator[tuple[str, str]]:
    """Weighted round-robin over sources, yielding (category, text). Infinite -- callers slice
    with itertools.islice or stop once a token budget is hit."""
    rng = random.Random(seed)
    iters = {s.name: iter_source_text(s) for s in sources}
    weights = {s.name: s.weight for s in sources}
    categories = {s.name: s.category for s in sources}
    names = list(iters.keys())
    while iters:
        name = rng.choices(names, weights=[weights[n] for n in names], k=1)[0]
        try:
            yield categories[name], next(iters[name])
        except StopIteration:
            # source exhausted (e.g. the small cosmopedia-100k) -- restart it so weights still
            # hold over a long run; documents will repeat, which is expected at our token budget
            iters[name] = iter_source_text(next(s for s in sources if s.name == name))
            names = list(iters.keys())


def pack_sequences(token_iter: Iterator[list[int]], seq_len: int, eos_id: int) -> Iterator[list[int]]:
    """Concatenate token streams with EOS separators and slice into fixed-length blocks -- the
    standard 'packing' approach so no compute is wasted on padding."""
    buffer: list[int] = []
    for tokens in token_iter:
        buffer.extend(tokens)
        buffer.append(eos_id)
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]


def build_training_stream(tokenizer, seq_len: int, sources: list[Source] = MIXTURE, seed: int = 0):
    """End-to-end: mixture -> tokenize -> pack. `tokenizer` is a `tokenizers.Tokenizer` (see
    scripts/train_tokenizer.py)."""
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    def token_iter():
        for _category, text in iter_mixture_text(sources, seed=seed):
            yield tokenizer.encode(text).ids

    return pack_sequences(token_iter(), seq_len, eos_id)


class PackedShardDataset:
    """Reads shards written by scripts/pretokenize.py via np.memmap -- O(1) memory, random-access
    batches, no tokenization or network I/O on the training loop's critical path. This is the
    real training-time data path; iter_mixture_text/build_training_stream above are the one-time
    corpus-preparation path (tokenizer training, pretokenize.py, dedup/contamination scanning).
    """

    def __init__(self, shard_dir: str):
        self.shard_dir = Path(shard_dir)
        with open(self.shard_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.seq_len = self.manifest["seq_len"]
        self.dtype = np.dtype(self.manifest["dtype"])
        self.shard_sizes = self.manifest["shard_sizes"]
        self._mmaps = [
            np.memmap(self.shard_dir / f"shard_{i:05d}.bin", dtype=self.dtype, mode="r",
                       shape=(n, self.seq_len))
            for i, n in enumerate(self.shard_sizes)
        ]
        # flat index -> (shard_idx, row_idx)
        self._index = [(s, r) for s, n in enumerate(self.shard_sizes) for r in range(n)]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> np.ndarray:
        s, r = self._index[i]
        return self._mmaps[s][r]

    def batches(self, batch_size: int, seed: int = 0, shuffle: bool = True):
        rng = random.Random(seed)
        order = list(range(len(self)))
        while True:
            if shuffle:
                rng.shuffle(order)
            for i in range(0, len(order) - batch_size + 1, batch_size):
                idxs = order[i:i + batch_size]
                yield np.stack([self[j] for j in idxs]).astype(np.int64)


if __name__ == "__main__":
    # quick manual check: mixture proportions actually observed over N draws
    from collections import Counter

    n = 200
    counts = Counter(cat for cat, _ in itertools.islice(iter_mixture_text(), n))
    print(f"observed category proportions over {n} draws:")
    for cat, c in counts.most_common():
        print(f"  {cat:20s} {c/n:.2%}")
