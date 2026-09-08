"""Mythos 6 dense transformer. See ARCHITECTURE.md sec 2 for the rationale behind each choice.

Deliberately built only on stock PyTorch ops (no custom CUDA/Triton kernels) so it runs
unmodified on Turing (T4) and Ampere+ (A100) alike -- SDPA auto-selects the best backend per GPU.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def rotary_freqs(seq_len: int, d_head: int, theta: float, device, dtype) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, d_head/2)
    return torch.cat([freqs, freqs], dim=-1).to(dtype)  # (seq_len, d_head)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq, d_head); freqs: (seq, d_head)
    cos, sin = freqs.cos(), freqs.sin()
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, is_windowed: bool):
        super().__init__()
        self.cfg = cfg
        self.is_windowed = is_windowed
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.d_head, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.d_head, cfg.norm_eps)
            self.k_norm = RMSNorm(self.d_head, cfg.norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, attn_mask) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # GQA: repeat kv heads to match query heads for SDPA's expected shape
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.d_head)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        is_windowed = cfg.window_every > 0 and (layer_idx % cfg.window_every != cfg.window_every - 1)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, is_windowed=is_windowed)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, freqs, causal_mask, window_mask):
        mask = window_mask if self.attn.is_windowed else causal_mask
        x = x + self.attn(self.attn_norm(x), freqs, mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


def build_masks(seq_len: int, window_size: int, device, dtype):
    """Boolean SDPA masks: True = attend. Causal-only, and causal+local-window."""
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = j <= i
    window = causal & (i - j < window_size)
    # SDPA wants shape broadcastable to (batch, heads, seq, seq)
    return causal.view(1, 1, seq_len, seq_len), window.view(1, 1, seq_len, seq_len)


class MythosCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        b, t = input_ids.shape
        cfg = self.cfg
        x = self.embed(input_ids)
        freqs = rotary_freqs(t, cfg.d_head, cfg.rope_theta, x.device, x.dtype)
        causal_mask, window_mask = build_masks(t, cfg.window_size, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, freqs, causal_mask, window_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss
