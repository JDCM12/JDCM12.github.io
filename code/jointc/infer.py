from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import JointCConfig
from .model import JointCModel
from .sampling import sample_jointc_latents, sample_semantic_ids
from .train import load_backbone
from .text import build_text_condition


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public JointC inference")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument(
        "--structure",
        default="",
        help="Optional fallback section structure when lyrics have no inline tags.",
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", default="jointc_sample.pt")
    parser.add_argument("--max-text-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for inference") from exc

    tokenizer_name = args.tokenizer or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    condition_text = build_text_condition(
        args.prompt,
        args.lyrics,
        structure=args.structure,
    )
    tokens = tokenizer(
        condition_text,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    backbone = load_backbone(args.model_name_or_path)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_config = state.get("config", {})
    config = JointCConfig(
        hidden_size=int(backbone.config.hidden_size),
        semantic_vocab_size=int(saved_config.get("semantic_vocab_size", 32768)),
        latent_dim=int(saved_config.get("latent_dim", 96)),
        condition_dim=int(saved_config.get("condition_dim", 1152)),
        max_audio_tokens=int(saved_config.get("max_audio_tokens", 8192)),
        dit_hidden_size=int(saved_config.get("dit_hidden_size", 1152)),
        dit_depth=int(saved_config.get("dit_depth", 28)),
        dit_num_heads=int(saved_config.get("dit_num_heads", 16)),
        dit_mlp_ratio=float(saved_config.get("dit_mlp_ratio", 4.0)),
        dit_cond_drop_prob=float(saved_config.get("dit_cond_drop_prob", 0.1)),
        dit_activation_checkpointing=bool(
            saved_config.get("dit_activation_checkpointing", False)
        ),
        dit_activation_checkpointing_use_reentrant=bool(
            saved_config.get("dit_activation_checkpointing_use_reentrant", False)
        ),
        bos_token_id=saved_config.get("bos_token_id"),
        eos_token_id=saved_config.get("eos_token_id"),
        pad_token_id=saved_config.get("pad_token_id"),
        continuous_head_layers=int(saved_config.get("continuous_head_layers", 2)),
        d_corrupt_prob=float(saved_config.get("d_corrupt_prob", 0.5)),
        fm_loss_weight=float(saved_config.get("fm_loss_weight", 1.0)),
        unfreeze_last_backbone_layers=0,
        unfreeze_audio_embedding=False,
        ar_gradient_checkpointing=False,
    )
    model = JointCModel(backbone, config)
    model.load_state_dict(state.get("model", state), strict=True)
    device = torch.device(args.device)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and args.precision == "bf16"
        else torch.float32
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    text_input_ids = tokens["input_ids"][:, : args.max_text_tokens].to(device)
    if text_input_ids.shape[1] == 0:
        raise ValueError("the text condition produced no tokenizer IDs")
    if args.max_new_tokens > config.max_audio_tokens:
        raise ValueError(
            "max_new_tokens cannot exceed the DiT max_audio_tokens setting"
        )
    ids = sample_semantic_ids(
        model,
        text_input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    if ids.shape[1] == 0:
        raise RuntimeError("the semantic-ID sampler produced EOS before any audio ID")
    latent = sample_jointc_latents(
        model,
        text_input_ids,
        ids,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text": condition_text,
            "semantic_ids": ids.cpu(),
            "latent": latent.cpu(),
        },
        output,
    )
    print(json.dumps({"output": str(output), "frames": int(ids.shape[1])}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
