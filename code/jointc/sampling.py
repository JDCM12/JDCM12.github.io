from __future__ import annotations

import torch

from .model import JointCModel


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0 and top_k < filtered.shape[-1]:
        values, _ = torch.topk(filtered, top_k, dim=-1)
        cutoff = values[..., -1, None]
        filtered[filtered < cutoff] = -torch.inf
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative - sorted_logits.softmax(dim=-1) > top_p
        remove[..., 0] = False
        sorted_logits[remove] = -torch.inf
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            -1, sorted_indices, sorted_logits
        )
    return filtered


@torch.no_grad()
def sample_semantic_ids(
    model: JointCModel,
    text_input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int | None = None,
) -> torch.Tensor:
    """Autoregressively sample semantic IDs from the Qwen hidden state."""

    if text_input_ids.shape[0] != 1:
        raise ValueError(
            "sample_semantic_ids currently expects batch size 1 so EOS and "
            "the generated latent length remain unambiguous"
        )
    model.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=text_input_ids.device).manual_seed(
            int(seed)
        )
    generated = torch.empty(
        text_input_ids.shape[0], 0, dtype=torch.long, device=text_input_ids.device
    )
    for _ in range(int(max_new_tokens)):
        logits = model.next_token_logits(text_input_ids, generated)
        logits[..., model.config.bos_token_id] = -torch.inf
        logits[..., model.config.pad_token_id] = -torch.inf
        logits = logits / max(float(temperature), 1e-5)
        logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
        token = torch.multinomial(
            logits.softmax(dim=-1),
            1,
            generator=generator,
        )
        if bool((token == model.config.eos_token_id).all()):
            break
        # EOS is a control symbol, not a renderer condition.
        generated = torch.cat([generated, token], dim=1)
    return generated


@torch.no_grad()
def sample_jointc_latents(
    model: JointCModel,
    text_input_ids: torch.Tensor,
    semantic_ids: torch.Tensor,
    *,
    steps: int = 32,
    cfg_scale: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample semantic latents with the full sequence DiT."""

    condition = model.condition_for_ids(text_input_ids, semantic_ids)
    return model.sample_latent(
        semantic_ids,
        cond_z=condition,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
