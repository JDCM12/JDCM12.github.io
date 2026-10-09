from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import JointCConfig
from .dit import Id2LatentDiT


def _get_input_embeddings(backbone: nn.Module) -> nn.Module:
    getter = getattr(backbone, "get_input_embeddings", None)
    if callable(getter):
        embeddings = getter()
        if embeddings is not None:
            return embeddings
    for name in ("embed_tokens", "wte"):
        embeddings = getattr(backbone, name, None)
        if embeddings is not None:
            return embeddings
    raise AttributeError("backbone must expose get_input_embeddings() or embed_tokens")


def _get_backbone_layers(backbone: nn.Module) -> list[nn.Module]:
    for owner_name in ("model", ""):
        owner = backbone if not owner_name else getattr(backbone, owner_name, None)
        if owner is None:
            continue
        layers = getattr(owner, "layers", None)
        if layers is not None:
            return list(layers)
    return []


def flow_matching_loss(
    predictor: nn.Module,
    times: torch.Tensor,
    condition: torch.Tensor | None,
    target: torch.Tensor,
    noise: torch.Tensor,
    *,
    semantic_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the flow-matching loss and the predicted velocity.

    The interpolation is ``x_t=(1-t)epsilon+t*x_1`` and the target velocity
    is ``x_1-epsilon``. ``predictor`` is the full sequence semantic latent DiT,
    so each frame can attend to the rest of the song sequence.
    """

    t = times.reshape(-1, *([1] * (target.ndim - 1))).to(
        dtype=target.dtype,
        device=target.device,
    )
    noisy = (1.0 - t) * noise + t * target
    velocity_target = target - noise
    if semantic_ids is None:
        raise ValueError("semantic_ids are required by the ID-conditioned DiT")
    velocity_pred = predictor(
        noisy,
        times,
        semantic_ids,
        cond_z=condition,
    )
    return (
        F.mse_loss(velocity_pred.float(), velocity_target.float()),
        velocity_pred,
    )


class LatentReadout(nn.Module):
    """Maps causal Qwen hidden states to the DiT conditioning width."""

    def __init__(self, hidden_size: int, condition_dim: int, layers: int = 2) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        width = int(hidden_size)
        for _ in range(max(0, int(layers) - 1)):
            blocks.extend([nn.Linear(width, condition_dim), nn.SiLU()])
            width = int(condition_dim)
        blocks.append(nn.Linear(width, condition_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class JointCModel(nn.Module):
    """Qwen backbone + full ID-conditioned semantic latent DiT.

    There are two explicit training stages:

    ``warmup_forward``
        Train the DiT with reference semantic IDs and continuous SSL-VAE
        targets.  The Qwen planner is not involved.

    ``jointc_forward``
        Run teacher-forced Qwen hidden states through ``latent_head`` and train
        ``L_ID + lambda * L_FM``.  The DiT receives IDs plus the differentiable
        hidden-state readout, so FM gradients reach the selected Qwen layers.

    SSL discrete-token extraction, SSL-VAE target extraction, and LeVo VAE
    decoding are intentionally injected at the data/adapter boundary.
    """

    def __init__(self, backbone: nn.Module, config: JointCConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        input_embeddings = _get_input_embeddings(backbone)
        if config.freeze_input_embedding:
            input_embeddings.requires_grad_(False)

        self.audio_vocab_size = max(
            config.semantic_vocab_size,
            config.bos_token_id + 1,
            config.eos_token_id + 1,
            config.pad_token_id + 1,
        )
        self.audio_embedding = nn.Embedding(
            self.audio_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.id_head = nn.Linear(config.hidden_size, self.audio_vocab_size, bias=False)
        self.latent_head = LatentReadout(
            config.hidden_size,
            config.condition_dim,
            layers=config.continuous_head_layers,
        )
        self.dit = Id2LatentDiT(
            latent_dim=config.latent_dim,
            vocab_size=config.semantic_vocab_size,
            hidden_size=config.dit_hidden_size,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            mlp_ratio=config.dit_mlp_ratio,
            max_seq_len=config.max_audio_tokens,
            cond_drop_prob=config.dit_cond_drop_prob,
            z_cond_dim=config.condition_dim,
            activation_checkpointing=config.dit_activation_checkpointing,
            activation_checkpointing_use_reentrant=(
                config.dit_activation_checkpointing_use_reentrant
            ),
        )
        self._configure_trainable_backbone()

    @property
    def bos_token_id(self) -> int:
        return int(self.config.bos_token_id)

    @property
    def input_embeddings(self) -> nn.Module:
        """Return Qwen's text embedding without registering a duplicate alias."""

        return _get_input_embeddings(self.backbone)

    @property
    def eos_token_id(self) -> int:
        return int(self.config.eos_token_id)

    @property
    def uncond_id(self) -> int:
        return int(self.dit.uncond_id)

    def _configure_trainable_backbone(self) -> None:
        """Freeze Qwen, then unfreeze only the requested JointC parameters."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        layers = _get_backbone_layers(self.backbone)
        count = min(self.config.unfreeze_last_backbone_layers, len(layers))
        for layer in layers[-count:] if count else []:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        norm = getattr(self.backbone, "norm", None)
        if norm is not None and count:
            for parameter in norm.parameters():
                parameter.requires_grad_(True)

        if count and self.config.ar_gradient_checkpointing:
            enable = getattr(self.backbone, "gradient_checkpointing_enable", None)
            if callable(enable):
                try:
                    enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    enable()

        self.id_head.requires_grad_(True)
        self.latent_head.requires_grad_(True)
        self.dit.requires_grad_(True)
        self.audio_embedding.requires_grad_(self.config.unfreeze_audio_embedding)

    def unfreeze_last_backbone_layers(self, count: int) -> None:
        self.config.unfreeze_last_backbone_layers = int(count)
        self._configure_trainable_backbone()

    def _backbone_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise AttributeError("backbone output must contain last_hidden_state")
        return hidden

    def _teacher_forced(
        self,
        text_input_ids: torch.Tensor,
        semantic_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Build the Qwen sequence and align hidden states to target frames."""

        device = semantic_ids.device
        text_input_ids = text_input_ids.to(device=device, dtype=torch.long)
        semantic_ids = semantic_ids.to(device=device, dtype=torch.long)
        audio_mask = audio_mask.to(device=device, dtype=torch.bool)
        batch, audio_len = semantic_ids.shape
        text_len = text_input_ids.shape[1]

        bos = torch.full(
            (batch, 1),
            self.bos_token_id,
            device=device,
            dtype=torch.long,
        )
        eos = torch.full(
            (batch, 1),
            self.eos_token_id,
            device=device,
            dtype=torch.long,
        )
        audio_input_ids = torch.cat([bos, semantic_ids, eos], dim=1)
        text_emb = self.input_embeddings(text_input_ids)
        audio_emb = self.audio_embedding(audio_input_ids)
        inputs_embeds = torch.cat([text_emb, audio_emb], dim=1)

        text_mask = (
            torch.ones_like(text_input_ids, dtype=torch.bool)
            if text_attention_mask is None
            else text_attention_mask.to(device=device, dtype=torch.bool)
        )
        audio_attention = torch.cat(
            [
                torch.ones_like(bos, dtype=torch.bool),
                audio_mask,
                torch.ones_like(eos, dtype=torch.bool),
            ],
            dim=1,
        )
        attention_mask = torch.cat([text_mask, audio_attention], dim=1)
        hidden = self._backbone_forward(inputs_embeds, attention_mask)
        logits = self.id_head(hidden)

        # The BOS state predicts semantic frame 0; each previous ID state
        # predicts the next ID.  This gives exactly T hidden states.
        aligned_hidden = hidden[:, text_len : text_len + audio_len]
        shift_logits = logits[:, :-1]
        targets = torch.cat(
            [
                torch.full_like(text_input_ids, -100),
                audio_input_ids,
            ],
            dim=1,
        )
        shift_targets = targets[:, 1:]
        id_mask = torch.cat(
            [
                torch.zeros_like(text_mask, dtype=torch.bool),
                audio_attention,
            ],
            dim=1,
        )[:, 1:]
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
            shift_targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shift_targets)
        mask_float = id_mask.to(dtype=token_loss.dtype)
        id_loss = (token_loss * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        predictions = shift_logits.argmax(dim=-1)
        id_accuracy = (
            ((predictions == shift_targets) & id_mask).sum().float()
            / id_mask.sum().clamp_min(1).float()
        )
        return {
            "hidden": aligned_hidden,
            "logits": logits,
            "id_loss": id_loss,
            "id_accuracy": id_accuracy,
            "attention_mask": attention_mask,
        }

    def _masked_flow_loss(
        self,
        semantic_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        target: torch.Tensor,
        *,
        cond_z: Optional[torch.Tensor] = None,
        d_corrupt_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        target = target.to(device=semantic_ids.device, dtype=torch.float32)
        audio_mask = audio_mask.to(device=target.device, dtype=torch.bool)
        if target.shape[:2] != semantic_ids.shape:
            raise ValueError("target, semantic_ids and audio_mask must align")
        noise = torch.randn_like(target)
        times = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        dit_ids = semantic_ids
        if d_corrupt_prob > 0 and self.training:
            if self.dit.cond_drop_prob <= 0:
                raise ValueError("d_corrupt_prob requires dit_cond_drop_prob > 0")
            drop = torch.rand_like(semantic_ids, dtype=torch.float32) < float(
                d_corrupt_prob
            )
            dit_ids = torch.where(
                drop,
                torch.full_like(semantic_ids, self.uncond_id),
                semantic_ids,
            )
        fm_raw, prediction = flow_matching_loss(
            self.dit,
            times,
            cond_z,
            target,
            noise,
            semantic_ids=dit_ids,
        )
        per_token = F.mse_loss(
            prediction.float(),
            (target - noise).float(),
            reduction="none",
        ).mean(dim=-1)
        mask_float = audio_mask.to(dtype=per_token.dtype)
        denom = mask_float.sum().clamp_min(1.0)
        masked = (per_token * mask_float).sum() / denom
        per_token_l1 = (
            prediction.float() - (target - noise).float()
        ).abs().mean(dim=-1)
        masked_l1 = (per_token_l1 * mask_float).sum() / denom
        return {
            "fm_loss": masked,
            "fm_loss_unmasked": fm_raw.detach(),
            "flow_l1": masked_l1.detach(),
            "prediction": prediction,
            "noise": noise,
            "times": times,
        }

    def warmup_forward(
        self,
        *,
        semantic_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        continuous_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Stage A/B: warm up the full DiT from reference IDs and latents."""

        flow = self._masked_flow_loss(
            semantic_ids,
            audio_mask,
            continuous_targets,
            cond_z=None,
            d_corrupt_prob=0.0,
        )
        return {
            "loss": flow["fm_loss"],
            "fm_loss": flow["fm_loss"],
            "flow_l1": flow["flow_l1"],
            "prediction": flow["prediction"],
        }

    def jointc_forward(
        self,
        *,
        text_input_ids: torch.Tensor,
        semantic_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        continuous_targets: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        d_corrupt_prob: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """Stage C: ``CE + FM`` with FM gradients reaching Qwen readouts."""

        teacher = self._teacher_forced(
            text_input_ids,
            semantic_ids,
            audio_mask,
            text_attention_mask,
        )
        z = self.latent_head(teacher["hidden"])
        flow = self._masked_flow_loss(
            semantic_ids,
            audio_mask,
            continuous_targets,
            cond_z=z,
            d_corrupt_prob=(
                self.config.d_corrupt_prob
                if d_corrupt_prob is None
                else float(d_corrupt_prob)
            ),
        )
        total = teacher["id_loss"] + self.config.fm_loss_weight * flow["fm_loss"]
        return {
            "loss": total,
            "id_loss": teacher["id_loss"],
            "id_accuracy": teacher["id_accuracy"],
            "fm_loss": flow["fm_loss"],
            "flow_l1": flow["flow_l1"],
            "z_abs": z.float().abs().mean().detach(),
            "condition": z,
            "prediction": flow["prediction"],
        }

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Default to JointC when text IDs are present, otherwise warmup."""

        if "text_input_ids" in batch:
            return self.jointc_forward(
                text_input_ids=batch["text_input_ids"],
                semantic_ids=batch["semantic_ids"],
                audio_mask=batch["audio_mask"],
                continuous_targets=batch["continuous_targets"],
                text_attention_mask=batch.get("text_attention_mask"),
            )
        return self.warmup_forward(
            semantic_ids=batch["semantic_ids"],
            audio_mask=batch["audio_mask"],
            continuous_targets=batch["continuous_targets"],
        )

    @torch.no_grad()
    def condition_for_ids(
        self,
        text_input_ids: torch.Tensor,
        semantic_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the causal LM readout aligned with each completed ID frame."""

        mask = torch.ones_like(semantic_ids, dtype=torch.bool)
        hidden = self._teacher_forced(
            text_input_ids,
            semantic_ids,
            mask,
        )["hidden"]
        return self.latent_head(hidden).detach()

    @torch.no_grad()
    def next_token_logits(
        self,
        text_input_ids: torch.Tensor,
        semantic_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for the next semantic ID."""

        device = text_input_ids.device
        batch = text_input_ids.shape[0]
        bos = torch.full(
            (batch, 1),
            self.bos_token_id,
            device=device,
            dtype=torch.long,
        )
        audio_ids = torch.cat([bos, semantic_ids.to(dtype=torch.long)], dim=1)
        embeddings = torch.cat(
            [
                self.input_embeddings(text_input_ids),
                self.audio_embedding(audio_ids),
            ],
            dim=1,
        )
        mask = torch.ones(embeddings.shape[:2], device=device, dtype=torch.bool)
        hidden = self._backbone_forward(embeddings, mask)
        return self.id_head(hidden[:, -1])

    @torch.no_grad()
    def sample_latent(
        self,
        semantic_ids: torch.Tensor,
        *,
        cond_z: Optional[torch.Tensor] = None,
        steps: int = 32,
        cfg_scale: float = 1.0,
        cond_is_50hz: bool = False,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Euler flow sampling through the full semantic latent DiT."""

        self.eval()
        batch, condition_len = semantic_ids.shape
        if int(steps) < 1:
            raise ValueError("steps must be at least one")
        if cond_is_50hz and condition_len % 2:
            raise ValueError(
                f"50-Hz condition length must be even, got {condition_len}"
            )
        if cfg_scale != 1.0 and not self.dit.use_cfg:
            raise ValueError(
                "cfg_scale requires a DiT trained with dit_cond_drop_prob > 0"
            )
        latent_len = condition_len // 2 if cond_is_50hz else condition_len
        dtype = next(self.dit.parameters()).dtype
        generator = None
        if seed is not None:
            generator = torch.Generator(device=semantic_ids.device).manual_seed(
                int(seed)
            )
        x = torch.randn(
            batch,
            latent_len,
            self.config.latent_dim,
            device=semantic_ids.device,
            dtype=torch.float32,
            generator=generator,
        )
        times = torch.linspace(
            0.0,
            1.0,
            int(steps) + 1,
            device=semantic_ids.device,
            dtype=torch.float32,
        )
        for step in range(int(steps)):
            t = times[step].expand(batch)
            dt = times[step + 1] - times[step]
            if cfg_scale != 1.0:
                conditional = self.dit(
                    x.to(dtype=dtype),
                    t.to(dtype=dtype),
                    semantic_ids,
                    cond_z=cond_z,
                    cond_is_50hz=cond_is_50hz,
                ).float()
                unconditional = self.dit(
                    x.to(dtype=dtype),
                    t.to(dtype=dtype),
                    semantic_ids,
                    cond_z=cond_z,
                    force_uncond=True,
                    cond_is_50hz=cond_is_50hz,
                ).float()
                velocity = unconditional + cfg_scale * (conditional - unconditional)
            else:
                velocity = self.dit(
                    x.to(dtype=dtype),
                    t.to(dtype=dtype),
                    semantic_ids,
                    cond_z=cond_z,
                    cond_is_50hz=cond_is_50hz,
                ).float()
            x = x + dt * velocity
        return x
