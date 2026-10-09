from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class JointCConfig:
    """Configuration for the public JointC wrapper.

    The values describe interfaces rather than a particular private run.
    ``hidden_size`` should normally be read from the loaded Qwen config.
    """

    hidden_size: int = 1536
    semantic_vocab_size: int = 32768
    latent_dim: int = 96
    condition_dim: int = 1152
    max_audio_tokens: int = 8192
    dit_hidden_size: int = 1152
    dit_depth: int = 28
    dit_num_heads: int = 16
    dit_mlp_ratio: float = 4.0
    dit_cond_drop_prob: float = 0.1
    dit_activation_checkpointing: bool = False
    dit_activation_checkpointing_use_reentrant: bool = False
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    continuous_head_layers: int = 2
    fm_loss_weight: float = 1.0
    d_corrupt_prob: float = 0.5
    warmup_lr: float = 5e-5
    jointc_dit_lr: float = 5e-5
    jointc_ar_lr: float = 1e-5
    freeze_input_embedding: bool = True
    unfreeze_last_backbone_layers: int = 28
    unfreeze_audio_embedding: bool = True
    ar_gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        self.hidden_size = int(self.hidden_size)
        self.semantic_vocab_size = int(self.semantic_vocab_size)
        self.latent_dim = int(self.latent_dim)
        self.condition_dim = int(self.condition_dim)
        self.max_audio_tokens = int(self.max_audio_tokens)
        self.dit_hidden_size = int(self.dit_hidden_size)
        self.dit_depth = int(self.dit_depth)
        self.dit_num_heads = int(self.dit_num_heads)
        self.dit_mlp_ratio = float(self.dit_mlp_ratio)
        self.dit_cond_drop_prob = float(self.dit_cond_drop_prob)
        self.dit_activation_checkpointing = bool(self.dit_activation_checkpointing)
        self.dit_activation_checkpointing_use_reentrant = bool(
            self.dit_activation_checkpointing_use_reentrant
        )
        self.ar_gradient_checkpointing = bool(self.ar_gradient_checkpointing)
        self.bos_token_id = (
            self.semantic_vocab_size
            if self.bos_token_id is None
            else int(self.bos_token_id)
        )
        self.eos_token_id = (
            self.semantic_vocab_size + 1
            if self.eos_token_id is None
            else int(self.eos_token_id)
        )
        self.pad_token_id = (
            self.semantic_vocab_size + 2
            if self.pad_token_id is None
            else int(self.pad_token_id)
        )
        if self.latent_dim <= 0 or self.condition_dim <= 0:
            raise ValueError("latent_dim and condition_dim must be positive")
        if self.continuous_head_layers < 1:
            raise ValueError("continuous_head_layers must be at least one")
        if self.dit_depth < 1 or self.dit_num_heads < 1:
            raise ValueError("dit_depth and dit_num_heads must be positive")
        if self.dit_hidden_size % self.dit_num_heads:
            raise ValueError("dit_hidden_size must be divisible by dit_num_heads")
