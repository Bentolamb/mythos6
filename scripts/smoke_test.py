"""M0 exit criterion: instantiate the smallest config, run a few optimizer steps on synthetic
random-token data, assert the loss decreases and nothing NaNs. This is a correctness check for
the code, not a quality check for the model -- it runs on CPU or the local 4GB GPU in seconds,
specifically so we validate the implementation before spending any Kaggle/Colab quota on it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from mythos.config import PRESETS, param_count
from mythos.model import MythosCausalLM


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PRESETS["mythos6-nano"]
    print(f"device={device}  config={cfg}")
    print(f"param_count (closed-form) = {param_count(cfg):,}")

    model = MythosCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param_count (actual, incl. tied embed once) = {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    torch.manual_seed(0)
    batch_size, seq_len = 4, cfg.max_seq_len
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    losses = []
    for step in range(30):
        logits, loss = model(input_ids, labels=input_ids)
        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss}"
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 5 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}")

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"\nPASS: loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps on {device}")


if __name__ == "__main__":
    main()
