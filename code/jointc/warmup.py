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
from .dit import Id2LatentDiT


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


def train(args: argparse.Namespace) -> None:
    _configure_runtime(args.seed, args.tf32)
    config = JointCConfig(
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
        unfreeze_last_backbone_layers=0,
        unfreeze_audio_embedding=False,
        ar_gradient_checkpointing=False,
    )
    dit = Id2LatentDiT(
        latent_dim=config.latent_dim,
        vocab_size=config.semantic_vocab_size,
        hidden_size=config.dit_hidden_size,
        depth=config.dit_depth,
        num_heads=config.dit_num_heads,
        mlp_ratio=config.dit_mlp_ratio,
        max_seq_len=config.max_audio_tokens,
        cond_drop_prob=config.dit_cond_drop_prob,
        z_cond_dim=0,
        activation_checkpointing=config.dit_activation_checkpointing,
    )
    dataset = JsonlJointCDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_jointc,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        foreach=False,
    )
    device = torch.device(args.device)
    dit.to(device)
    dit.train()
    iterator = iter(loader)
    for step in range(args.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        ids = batch["semantic_ids"]
        mask = batch["audio_mask"]
        target = materialize_continuous_targets(
            batch,
            sample_posterior=args.sample_vae_target,
        )
        noise = torch.randn_like(target)
        times = torch.rand(target.shape[0], device=device)
        t = times[:, None, None]
        noisy = (1.0 - t) * noise + t * target
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, args.precision):
            prediction = dit(noisy, times, ids)
            velocity_target = target - noise
            per_token = (
                prediction.float() - velocity_target.float()
            ).pow(2).mean(dim=-1)
            mask_float = mask.to(dtype=per_token.dtype)
            loss = (per_token * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), args.gradient_clip)
        optimizer.step()
        if step % args.log_every == 0:
            print(f"stage=warmup step={step} fm_loss={loss.item():.5f}")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": "warmup",
                "model": dit.state_dict(),
                "config": asdict(config),
                "sample_vae_target": bool(args.sample_vae_target),
                "precision": args.precision,
                "seed": int(args.seed),
                "tf32": bool(args.tf32),
            },
            path,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warm up the full ID-conditioned semantic latent DiT"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="dit_warmup_public.pt")
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
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
