from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm: bool = True
    # sliding-window attention: layers whose index % window_every != 0 use a local window;
    # every `window_every`-th layer (0-indexed) is full attention. See ARCHITECTURE.md sec 2.
    window_size: int = 2048
    window_every: int = 4
    tie_embeddings: bool = True
    dropout: float = 0.0

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"


# Named presets matching ARCHITECTURE.md sec 3. Total/active param counts there are derived from
# these exact configs -- keep them in sync if either changes.
PRESETS = {
    # tiny config for scripts/smoke_test.py only -- not a real training target
    "mythos6-nano": ModelConfig(
        vocab_size=1024, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=176,
        max_seq_len=256, window_size=64, window_every=2,
    ),
    "mythos6-130m-dense": ModelConfig(
        vocab_size=49152, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, d_ff=2048,
        max_seq_len=4096,
    ),
    "mythos6-320m-dense": ModelConfig(
        vocab_size=49152, d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, d_ff=2816,
        max_seq_len=4096,
    ),
}


def param_count(cfg: ModelConfig) -> int:
    """Closed-form param count matching the formulas in ARCHITECTURE.md sec 3 (dense FFN only;
    MoE configs are counted separately in moe.py)."""
    attn_per_layer = 2 * cfg.d_model * cfg.d_model + 2 * cfg.d_model * (cfg.n_kv_heads * cfg.d_head)
    mlp_per_layer = 3 * cfg.d_model * cfg.d_ff
    norms_per_layer = 2 * cfg.d_model  # attn_norm + mlp_norm
    if cfg.qk_norm:
        norms_per_layer += 2 * cfg.d_head  # q_norm + k_norm
    per_layer = attn_per_layer + mlp_per_layer + norms_per_layer
    embed = cfg.vocab_size * cfg.d_model
    total = per_layer * cfg.n_layers + embed + cfg.d_model  # + final_norm
    if not cfg.tie_embeddings:
        total += embed
    return total
