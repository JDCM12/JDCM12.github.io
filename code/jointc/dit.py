"""Self-contained 1-D DiT for ID/Qwen-state -> SSL-VAE latents.

The private SSL tokenizer, SSL-VAE encoder and LeVo VAE are intentionally
outside this module. The DiT itself is kept explicit so the offline warmup and
JointC stages remain reproducible.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return a frozen [length, dim] 1-D sinusoidal position table."""

    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    half = dim // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half, 1)
    )[None, :]
    values = position * frequency
    # Match the frozen 1-D table used by the original DiT utility:
    # first sine features, then cosine features.
    table = torch.cat([torch.sin(values), torch.cos(values)], dim=-1)
    if dim % 2:
        table = F.pad(table, (0, 1))
    return table


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half, device=timesteps.device, dtype=torch.float32
            )
            / max(half, 1)
        )
        args = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).reshape(batch, length, self.hidden_size)
        return self.proj(output)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        width = int(hidden_size * mlp_ratio)
        # Keep the conventional DiT/timm MLP parameter names so warmup
        # checkpoints can be loaded without a custom key rewrite.
        self.fc1 = nn.Linear(hidden_size, width)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(width, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """AdaLN-zero transformer block used by the semantic latent DiT."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        x = x + gate_msa[:, None, :] * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
        )
        self.linear = nn.Linear(hidden_size, latent_dim, bias=True)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class Id2LatentDiT(nn.Module):
    """Full sequence 1-D DiT that predicts the SSL-VAE latent velocity.

    Inputs:
      - ``x``: noisy continuous target, [B, T, latent_dim]
      - ``timesteps``: flow time, [B]
      - ``semantic_ids``: [B, T] (or [B, 2T] for the optional 50-Hz path)
      - ``cond_z``: optional LM readout, [B, T, z_dim]

    The output is a velocity field with the same shape as ``x``.
    """

    def __init__(
        self,
        latent_dim: int = 96,
        vocab_size: int = 32768,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 8192,
        cond_drop_prob: float = 0.1,
        z_cond_dim: int = 0,
        activation_checkpointing: bool = False,
        activation_checkpointing_use_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.max_seq_len = int(max_seq_len)
        self.cond_drop_prob = float(cond_drop_prob)
        self.z_cond_dim = int(z_cond_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.activation_checkpointing_use_reentrant = bool(
            activation_checkpointing_use_reentrant
        )
        self.use_cfg = self.cond_drop_prob > 0.0
        self.uncond_id = self.vocab_size
        self.prevq_dim = self.latent_dim

        self.x_embedder = nn.Linear(self.latent_dim, self.hidden_size)
        self.audio_id_embedder = nn.Embedding(
            self.vocab_size + (1 if self.use_cfg else 0),
            self.hidden_size,
        )
        self.z_proj = (
            nn.Linear(self.z_cond_dim, self.hidden_size)
            if self.z_cond_dim > 0
            else None
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.hidden_size, num_heads, mlp_ratio)
                for _ in range(int(depth))
            ]
        )
        self.final_layer = FinalLayer(self.hidden_size, self.latent_dim)
        self.pos_embed = nn.Parameter(
            _sinusoidal_positions(
                self.max_seq_len, self.hidden_size, torch.device("cpu")
            )[None],
            requires_grad=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.audio_id_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.z_proj is not None:
            # Warmup compatibility: z is initially ignored by the DiT.
            nn.init.zeros_(self.z_proj.weight)
            nn.init.zeros_(self.z_proj.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _prepare_ids(
        self,
        semantic_ids: torch.Tensor,
        output_len: int,
        batch: int,
        force_uncond: bool,
        cond_is_50hz: bool,
    ) -> torch.Tensor:
        ids = semantic_ids.to(dtype=torch.long)
        if self.use_cfg and self.training and not force_uncond:
            drop = torch.rand(batch, device=ids.device) < self.cond_drop_prob
            if drop.any():
                ids = ids.clone()
                ids[drop] = self.uncond_id
        elif force_uncond:
            if not self.use_cfg:
                raise ValueError(
                    "force_uncond requires cond_drop_prob > 0"
                )
            ids = torch.full_like(ids, self.uncond_id)
        embedded = self.audio_id_embedder(ids)
        if cond_is_50hz:
            if embedded.shape[1] >= output_len * 2:
                embedded = embedded[:, : output_len * 2].reshape(
                    batch, output_len, 2, self.hidden_size
                ).sum(dim=2)
            else:
                # Keep the legacy fallback explicit for boundary cases where
                # the 50-Hz ID sequence is shorter than exactly 2T.
                embedded = embedded.mean(dim=1, keepdim=True).expand(
                    batch, output_len, self.hidden_size
                )
        if embedded.shape[1] != output_len:
            raise ValueError(
                f"ID condition length {embedded.shape[1]} does not match "
                f"latent length {output_len}"
            )
        return embedded

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        semantic_ids: torch.Tensor,
        *,
        cond_z: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
        cond_is_50hz: bool = False,
        history_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, output_len, latent_dim = x.shape
        if latent_dim != self.latent_dim:
            raise ValueError("latent dimension does not match DiT configuration")
        model_x = x
        if history_x is not None:
            if history_x.shape != x.shape:
                raise ValueError("history_x must have the same shape as x")
            model_x = torch.cat([history_x, x], dim=1)
        model_len = model_x.shape[1]
        if model_len > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")

        h = self.x_embedder(model_x)
        id_condition = self._prepare_ids(
            semantic_ids,
            output_len,
            batch,
            force_uncond,
            cond_is_50hz,
        )
        if history_x is not None:
            id_condition = torch.cat(
                [
                    torch.zeros_like(id_condition),
                    id_condition,
                ],
                dim=1,
            )
        h = h + id_condition + self.pos_embed[:, :model_len].to(
            device=h.device,
            dtype=h.dtype,
        )

        if self.z_proj is not None and cond_z is not None:
            z = cond_z
            if z.shape[1] == 1:
                z = z.expand(batch, model_len, z.shape[-1])
            elif history_x is not None and z.shape[1] == output_len:
                z = torch.cat([z, z], dim=1)
            if z.shape[1] != model_len:
                raise ValueError("LM condition length does not match DiT sequence")
            h = h + self.z_proj(z.to(dtype=h.dtype))

        condition = self.t_embedder(
            timesteps.to(device=h.device, dtype=h.dtype)
        )
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                h = checkpoint(
                    block,
                    h,
                    condition,
                    use_reentrant=self.activation_checkpointing_use_reentrant,
                )
            else:
                h = block(h, condition)
        output = self.final_layer(h, condition)
        return output[:, -output_len:] if history_x is not None else output


def dit_s_id2latent(**kwargs) -> Id2LatentDiT:
    return Id2LatentDiT(hidden_size=384, depth=12, num_heads=6, **kwargs)


def dit_b_id2latent(**kwargs) -> Id2LatentDiT:
    return Id2LatentDiT(hidden_size=768, depth=12, num_heads=12, **kwargs)


def dit_l_id2latent(**kwargs) -> Id2LatentDiT:
    return Id2LatentDiT(hidden_size=1024, depth=24, num_heads=16, **kwargs)


def dit_xl_id2latent(**kwargs) -> Id2LatentDiT:
    return Id2LatentDiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)
