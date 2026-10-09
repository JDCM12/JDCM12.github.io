"""Interfaces for the components intentionally abstracted in the public code."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class DiscreteSemanticTokenizer(Protocol):
    """Private or replaceable SSL tokenizer that produces semantic IDs."""

    def encode(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``semantic_ids`` and their valid-frame mask."""


@runtime_checkable
class SSLVAEEncoder(Protocol):
    """Private or replaceable SSL-VAE that supplies continuous targets."""

    def encode(
        self,
        audio: torch.Tensor,
        *,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return a target plus optional posterior mean/log-variance."""


@runtime_checkable
class AcousticRenderer(Protocol):
    """External DiT from SSL-VAE bottlenecks to 64-D LeVo VAE latents."""

    def render(
        self,
        semantic_latents: torch.Tensor,
        *,
        steps: int = 32,
    ) -> torch.Tensor:
        """Return the acoustic latent sequence consumed by a waveform VAE."""


@runtime_checkable
class LeVoVAEDecoder(Protocol):
    """Replaceable LeVo VAE decoder interface."""

    def decode(self, acoustic_latents: torch.Tensor) -> torch.Tensor:
        """Return waveform shaped ``[batch, channels, samples]``."""


class PrivateModulePlaceholder:
    """Explicit boundary for a module that is not part of this release."""

    def __init__(self, name: str) -> None:
        self.name = str(name)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.name} is intentionally not bundled. "
            "Provide a compatible adapter in your local environment."
        )
