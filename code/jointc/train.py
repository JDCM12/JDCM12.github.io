from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import JointCConfig
from .data import (
    JsonlJointCDataset,
    collate_jointc,
    materialize_continuous_targets,
)
from .model import JointCModel


def load_backbone(model_name_or_path: str) -> torch.nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for a real Qwen run; install code/requirements.txt"
        ) from exc
    return AutoModel.from_pretrained(model_name_or_path)


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _configure_runtime(seed: int, tf32: bool) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)


def _load_partial(module: torch.nn.Module, path: str, prefixes: tuple[str, ...] = ()) -> None:
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    normalized = {}
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        normalized[new_key] = value
    state = normalized
    own = module.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in own and own[key].shape == value.shape
    }
    module.load_state_dict(filtered, strict=False)
    print(f"loaded {len(filtered)} compatible tensors from {path}")


def train(args: argparse.Namespace) -> None:
    _configure_runtime(args.seed, args.tf32)
    backbone = load_backbone(args.model_name_or_path)
    config = JointCConfig(
        hidden_size=int(backbone.config.hidden_size),
        semantic_vocab_size=args.semantic_vocab_size,
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        max_audio_tokens=args.max_audio_tokens,
        dit_hidden_size=args.dit_hidden_size,
        dit_depth=args.dit_depth,
        dit_num_heads=args.dit_num_heads,
        dit_mlp_ratio=args.dit_mlp_ratio,
        dit_cond_drop_prob=args.dit_cond_drop_prob,
        dit_activation_checkpointing=args.activation_checkpointing,
        d_corrupt_prob=args.d_corrupt_prob,
        fm_loss_weight=args.fm_loss_weight,
        unfreeze_last_backbone_layers=args.unfreeze_last_backbone_layers,
        unfreeze_audio_embedding=args.unfreeze_audio_embedding,
        ar_gradient_checkpointing=args.ar_gradient_checkpointing,
        jointc_dit_lr=args.dit_lr,
        jointc_ar_lr=args.ar_lr,
    )
    model = JointCModel(backbone, config)
    _load_partial(
        model.backbone,
        args.ar_checkpoint,
        prefixes=(
            "model.model.",
            "model.backbone.",
            "ar_model.",
            "backbone.",
            "model.",
        ),
    )
    if args.ar_checkpoint:
        # Original AR checkpoints commonly store these two public-interface
        # modules beside the Qwen transformer. Load compatible tensors when
        # present, while allowing a plain Hugging Face Qwen checkpoint too.
        _load_partial(
            model.id_head,
            args.ar_checkpoint,
            prefixes=("model.id_head.", "ar_model.id_head.", "id_head."),
        )
        _load_partial(
            model.audio_embedding,
            args.ar_checkpoint,
            prefixes=(
                "model.audio_embedding.",
                "ar_model.audio_embedding.",
                "audio_embedding.",
            ),
        )
    if args.warmup_checkpoint:
        _load_partial(
            model.dit,
            args.warmup_checkpoint,
            prefixes=("model.dit.", "dit.", "model."),
        )

    dataset = JsonlJointCDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_jointc,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    parameter_groups = [
        {
            "params": [
                parameter
                for module in (model.dit, model.latent_head)
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
            "lr": args.dit_lr,
        },
        {
            "params": [
                parameter
                for module in (model.backbone, model.id_head, model.audio_embedding)
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
            "lr": args.ar_lr,
        },
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    device = torch.device(args.device)
    model.to(device)
    model.train()
    iterator = iter(loader)
    for step in range(args.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        batch = {key: value.to(device) for key, value in batch.items()}
        continuous_targets = materialize_continuous_targets(
            batch,
            sample_posterior=args.sample_vae_target,
        )
        with _autocast_context(device, args.precision):
            output = model.jointc_forward(
                text_input_ids=batch["text_input_ids"],
                semantic_ids=batch["semantic_ids"],
                audio_mask=batch["audio_mask"],
                continuous_targets=continuous_targets,
                text_attention_mask=batch.get("text_attention_mask"),
            )
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in parameter_groups for parameter in group["params"]],
            args.gradient_clip,
        )
        optimizer.step()
        if step % args.log_every == 0:
            print(
                f"stage=jointc step={step} loss={output['loss'].item():.5f} "
                f"id_loss={output['id_loss'].item():.5f} "
                f"fm_loss={output['fm_loss'].item():.5f} "
                f"id_acc={output['id_accuracy'].item():.5f} "
                f"flow_l1={output['flow_l1'].item():.5f} "
                f"z_abs={output['z_abs'].item():.5f}"
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": "jointc",
                "model": model.state_dict(),
                "config": asdict(config),
                "sample_vae_target": bool(args.sample_vae_target),
                "precision": args.precision,
                "seed": int(args.seed),
                "tf32": bool(args.tf32),
            },
            output_path,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JointC stage: Qwen CE + full semantic-latent DiT FM"
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="jointc_public.pt")
    parser.add_argument("--ar-checkpoint", default="")
    parser.add_argument("--warmup-checkpoint", default="")
    parser.add_argument("--semantic-vocab-size", type=int, default=32768)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--condition-dim", type=int, default=1152)
    parser.add_argument("--max-audio-tokens", type=int, default=8192)
    parser.add_argument("--dit-hidden-size", type=int, default=1152)
    parser.add_argument("--dit-depth", type=int, default=28)
    parser.add_argument("--dit-num-heads", type=int, default=16)
    parser.add_argument("--dit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dit-cond-drop-prob", type=float, default=0.1)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--unfreeze-last-backbone-layers", type=int, default=28)
    parser.add_argument(
        "--unfreeze-audio-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ar-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--d-corrupt-prob", type=float, default=0.5)
    parser.add_argument("--fm-loss-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dit-lr", type=float, default=5e-5)
    parser.add_argument("--ar-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--sample-vae-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample from precomputed vae_mu/vae_logvar instead of using the mean.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="bf16",
        help="CUDA compute precision; CPU runs stay in fp32.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    train(build_parser().parse_args())
