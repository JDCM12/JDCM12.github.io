from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


class JsonlJointCDataset(Dataset):
    """Small, dependency-light dataset contract for public examples."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"no examples found in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        required = {"semantic_ids"}
        missing = required.difference(row)
        if missing:
            raise KeyError(f"row {index} is missing {sorted(missing)}")
        has_targets = "continuous_targets" in row
        has_posterior = "vae_mu" in row and "vae_logvar" in row
        if not has_targets and not has_posterior:
            raise KeyError(
                f"row {index} needs continuous_targets or vae_mu + vae_logvar"
            )
        if "audio_mask" not in row:
            row["audio_mask"] = [True] * len(row["semantic_ids"])
        return row


def _pad_1d(values: Iterable[Iterable[int]], pad_value: int) -> torch.Tensor:
    rows = [list(row) for row in values]
    width = max(len(row) for row in rows)
    out = torch.full((len(rows), width), pad_value, dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            out[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def collate_jointc(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad direct targets or a precomputed SSL-VAE posterior."""

    if not rows:
        raise ValueError("cannot collate an empty batch")

    text_ids = None
    text_presence = ["text_input_ids" in row for row in rows]
    if any(text_presence) and not all(text_presence):
        raise ValueError("text_input_ids must be present in every row or no rows")
    if all(text_presence):
        text_ids = _pad_1d((row["text_input_ids"] for row in rows), 0)
    semantic_ids = _pad_1d((row["semantic_ids"] for row in rows), 0)
    audio_mask = torch.zeros_like(semantic_ids, dtype=torch.bool)

    direct_targets = ["continuous_targets" in row for row in rows]
    posterior_targets = [
        "vae_mu" in row and "vae_logvar" in row for row in rows
    ]
    if all(direct_targets):
        target_keys = ("continuous_targets",)
    elif all(posterior_targets):
        target_keys = ("vae_mu", "vae_logvar")
    else:
        raise ValueError(
            "a batch must use either continuous_targets or vae_mu + vae_logvar"
        )

    first_values = rows[0][target_keys[0]]
    if not first_values or not first_values[0]:
        raise ValueError("continuous target arrays must be non-empty")
    latent_dim = len(first_values[0])
    padded_targets = {
        key: torch.zeros(
            semantic_ids.shape[0],
            semantic_ids.shape[1],
            latent_dim,
            dtype=torch.float32,
        )
        for key in target_keys
    }
    for batch_index, row in enumerate(rows):
        ids = row["semantic_ids"]
        mask = row.get("audio_mask", [True] * len(ids))
        tensors = {
            key: torch.tensor(row[key], dtype=torch.float32)
            for key in target_keys
        }
        for values in tensors.values():
            if values.ndim != 2 or values.shape[1] != latent_dim:
                raise ValueError("all continuous targets must have the same width")
        length = min(
            len(ids),
            len(mask),
            *(values.shape[0] for values in tensors.values()),
        )
        audio_mask[batch_index, :length] = torch.tensor(mask[:length], dtype=torch.bool)
        for key, values in tensors.items():
            padded_targets[key][batch_index, :length] = values[:length]
    output = {
        "semantic_ids": semantic_ids,
        "audio_mask": audio_mask,
        **padded_targets,
    }
    if text_ids is not None:
        text_attention_mask = torch.zeros_like(text_ids, dtype=torch.bool)
        for batch_index, row in enumerate(rows):
            supplied = row.get(
                "text_attention_mask",
                [True] * len(row["text_input_ids"]),
            )
            length = min(len(row["text_input_ids"]), len(supplied))
            text_attention_mask[batch_index, :length] = torch.tensor(
                supplied[:length],
                dtype=torch.bool,
            )
        output["text_input_ids"] = text_ids
        output["text_attention_mask"] = text_attention_mask
    return output


def materialize_continuous_targets(
    batch: dict[str, torch.Tensor],
    *,
    sample_posterior: bool = True,
) -> torch.Tensor:
    """Return direct targets or samples from a precomputed SSL-VAE posterior."""

    if "continuous_targets" in batch:
        return batch["continuous_targets"]
    if "vae_mu" not in batch or "vae_logvar" not in batch:
        raise KeyError(
            "batch needs continuous_targets or both vae_mu and vae_logvar"
        )
    mean = batch["vae_mu"]
    if not sample_posterior:
        return mean
    std = torch.exp(0.5 * batch["vae_logvar"])
    return mean + std * torch.randn_like(mean)
