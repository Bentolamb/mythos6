"""Training loop for Mythos 6. See ARCHITECTURE.md sec 6.1.

Designed around the actual constraint: Kaggle/Colab sessions die unpredictably, so checkpointing
and resume are not optional extras -- every run must be safely interruptible. Mixed precision
picks bf16 on GPUs that support it natively (Ampere+, e.g. Colab's A100) and falls back to fp16 +
GradScaler on ones that don't (Turing, e.g. Kaggle/Colab's T4), since T4 lacks bf16 tensor cores.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mythos.config import PRESETS
from mythos.model import MythosCausalLM


def pick_amp_dtype(device: str) -> torch.dtype | None:
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: str) -> torch.Tensor:
    """Placeholder data source for testing the loop mechanics without a trained tokenizer or
    network access -- NOT used for real training runs, which go through mythos.data instead."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def packed_batch_iter(shard_dir: str, batch_size: int, device: str, seed: int = 0):
    """Real training-time data path -- see MILESTONES.md M1 for why live tokenize-per-step
    (the old real_batch_iter, removed) was replaced with pre-tokenized memory-mapped shards."""
    from mythos.data import PackedShardDataset

    ds = PackedShardDataset(shard_dir)
    for batch in ds.batches(batch_size, seed=seed):
        yield torch.from_numpy(batch).to(device)


def save_checkpoint(out_dir: Path, step: int, model, optimizer, scaler):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    path = out_dir / "latest.pt"
    tmp = out_dir / "latest.pt.tmp"
    torch.save(ckpt, tmp)
    tmp.replace(path)  # atomic-ish: never leaves a half-written "latest.pt"


def load_checkpoint(out_dir: Path, model, optimizer, scaler, device: str) -> int:
    path = out_dir / "latest.pt"
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    print(f"Resumed from step {ckpt['step']}")
    return ckpt["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mythos6-130m-dense", choices=list(PRESETS.keys()))
    ap.add_argument("--data-dir", default=None,
                     help="dir of pre-tokenized shards from scripts/pretokenize.py; omit to use synthetic data")
    ap.add_argument("--out-dir", default="runs/default")
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PRESETS[args.preset]
    out_dir = Path(args.out_dir)

    model = MythosCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    amp_dtype = pick_amp_dtype(device)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(out_dir, model, optimizer, scaler if use_scaler else None, device)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        return args.lr

    if args.data_dir:
        batches = packed_batch_iter(args.data_dir, args.micro_batch_size, device)
    else:
        print("WARNING: no --data-dir given, training on synthetic random tokens (loop-mechanics test only).")
        batches = (synthetic_batch(cfg.vocab_size, args.micro_batch_size, cfg.max_seq_len, device)
                   for _ in iter(int, 1))

    model.train()
    t0 = time.time()
    step = start_step
    micro_step = 0
    accum_loss = 0.0

    for input_ids in batches:
        if step >= args.max_steps:
            break
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(input_ids, labels=input_ids)
            loss = loss / args.grad_accum_steps

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()
        micro_step += 1

        if micro_step == args.grad_accum_steps:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            micro_step = 0
            if step % 10 == 0:
                elapsed = time.time() - t0
                print(f"step {step:5d}  loss {accum_loss:.4f}  lr {lr_at(step):.2e}  {elapsed:.1f}s")
            accum_loss = 0.0

            if step % args.save_every == 0:
                save_checkpoint(out_dir, step, model, optimizer, scaler if use_scaler else None)
                print(f"  saved checkpoint at step {step}")

    save_checkpoint(out_dir, step, model, optimizer, scaler if use_scaler else None)
    print(f"Done. Final step {step}, checkpoint saved to {out_dir/'latest.pt'}")


if __name__ == "__main__":
    main()
