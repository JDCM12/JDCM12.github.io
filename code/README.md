# Public standard joint pathway core

This directory is a deliberately self-contained implementation of the public
part of the paper's standard joint pathway. It keeps the semantic latent DiT
explicit and reproducible without releasing private audio infrastructure.

## What is included

- a Qwen-compatible causal backbone wrapper;
- the submitted v2 prompt/lyrics normalization and numbered section tags;
- the semantic-ID autoregressive head;
- the full 1-D semantic latent DiT (AdaLN-zero transformer blocks, timestep
  embedding, ID conditioning, optional LM readout conditioning);
- masked ID cross-entropy;
- masked flow-matching velocity loss;
- separate standalone DiT pretraining and Qwen+DiT joint semantic training
  entry points;
- autoregressive semantic-ID sampling;
- adapter protocols for the discrete SSL tokenizer, SSL-VAE target encoder,
  acoustic renderer, and LeVo VAE decoder;
- a small JSONL dataset/collator contract;
- a smoke test that uses a tiny toy causal backbone.

## What is intentionally not included

- model weights or checkpoints;
- private training/evaluation data;
- internal tokenizer or SSL-VAE implementations;
- the downstream acoustic DiT and LeVo VAE waveform decoder;
- cluster launchers, storage paths, credentials, or company libraries.

The missing pieces are not silently mocked as working models. They are
represented by `NotImplementedError` adapters so that a downstream user must
provide an implementation explicitly.

## Install

The code requires Python 3.10+ and the packages in
[`requirements.txt`](requirements.txt). A full Qwen run additionally needs a
compatible `transformers` installation and access to a Qwen checkpoint.

```bash
python3 -m pip install -r code/requirements.txt
```

## Smoke test

The smoke test does not download a model and only exercises tensor shapes,
losses, gradients, and the sampling path:

```bash
PYTHONPATH=code python3 -m unittest code/tests/test_smoke.py
```

## Training data contract

`JsonlJointCDataset` accepts one JSON object per line. Standalone DiT examples
need semantic IDs and continuous targets; joint semantic examples additionally
include tokenized text:

```json
{
  "text_input_ids": [11, 12, 13],
  "semantic_ids": [4, 8, 15],
  "continuous_targets": [[0.1, 0.2], [0.0, 0.3], [0.2, 0.4]],
  "audio_mask": [true, true, true]
}
```

Instead of `continuous_targets`, a row may contain precomputed SSL-VAE
posterior parameters:

```json
{
  "semantic_ids": [4, 8, 15],
  "vae_mu": [[0.1, 0.2], [0.0, 0.3], [0.2, 0.4]],
  "vae_logvar": [[-1.0, -1.2], [-0.9, -1.1], [-1.3, -0.8]],
  "audio_mask": [true, true, true]
}
```

The training entry points sample `mu + std * epsilon` by default, matching the
offline SSL-VAE-target training path. Pass `--no-sample-vae-target` to train on
the posterior mean. For standalone pretraining, `text_input_ids` may be
omitted. The target width is configurable (the paper variants use 48, 96, and
128 dimensions). The dataset is only a public interface; it does not prescribe
how private audio encoders produce these arrays.

## Training stages

### Standalone semantic DiT pretraining

Standalone pretraining trains the full semantic latent DiT with reference
semantic IDs and continuous SSL-VAE targets. It does not require the Qwen
backbone:

```bash
PYTHONPATH=code python3 -m jointc.warmup \
  --data path/to/semantic_latents.jsonl \
  --latent-dim 96 \
  --dit-hidden-size 1152 \
  --dit-depth 28 \
  --dit-num-heads 16 \
  --max-steps 100000 \
  --output warmup_dit.pt
```

### Joint semantic training

Joint semantic training loads the pretrained DiT, runs teacher-forced Qwen
hidden states through a latent readout, and optimizes:

```text
L = L_ID + fm_loss_weight * L_FM
```

The flow-matching gradient passes through the readout into the selected Qwen
layers. `--d-corrupt-prob` can replace some DiT ID conditions with the
classifier-free unconditional ID while retaining the LM-derived condition.
The submitted public reference uses the DiT-XL 1-D configuration
(width/depth/heads = 1152/28/16), all 28 Qwen layers, trainable semantic-ID
embeddings, bf16 compute, TF32, seed 1994, activation checkpointing, and
separate DiT/AR learning rates. The CLI defaults mirror those settings.

```bash
PYTHONPATH=code python3 -m jointc.train \
  --model-name-or-path Qwen/Qwen2.5-1.5B \
  --data path/to/jointc.jsonl \
  --warmup-checkpoint warmup_dit.pt \
  --ar-checkpoint discrete_lm.pt \
  --unfreeze-last-backbone-layers 28 \
  --dit-lr 5e-5 \
  --ar-lr 1e-5 \
  --max-steps 500000 \
  --output jointc.pt
```

The public package does not implement the private SSL tokenizer, SSL-VAE, or
LeVo VAE. They provide the `semantic_ids` and `continuous_targets` fields in
the JSONL contract and can be replaced by local adapters.

## Using a real Qwen backbone

```python
from transformers import AutoModel
from jointc.config import JointCConfig
from jointc.model import JointCModel

backbone = AutoModel.from_pretrained("Qwen/Qwen2.5-1.5B")
config = JointCConfig(
    hidden_size=backbone.config.hidden_size,
    semantic_vocab_size=32768,
    latent_dim=96,
    condition_dim=1152,
    dit_hidden_size=1152,
    dit_depth=28,
    dit_num_heads=16,
)
model = JointCModel(backbone, config)
model.unfreeze_last_backbone_layers(28)
```

The full semantic DiT is available as `model.dit`; it is a sequence-level
transformer, not an independent per-frame predictor. The downstream acoustic
DiT that maps semantic features to 64-D LeVo VAE acoustic latents, and the
waveform decoder, are deliberately injected:

```python
from jointc.adapters import AcousticRenderer, LeVoVAEDecoder

semantic_latent = model.sample_latent(
    semantic_ids=semantic_ids,
    cond_z=model.condition_for_ids(text_input_ids, semantic_ids),
    steps=32,
)
acoustic_latent = acoustic_renderer.render(semantic_latent, steps=32)
waveform = levo_vae_decoder.decode(acoustic_latent)
```

This separation is the release boundary: the Qwen/joint-semantic logic is
public, and the private semantic/audio modules remain replaceable components.

## Inference boundary

`jointc.infer` autoregressively samples semantic IDs and then integrates the
full DiT velocity field to produce SSL-VAE latents:

```bash
PYTHONPATH=code python3 -m jointc.infer \
  --model-name-or-path Qwen/Qwen2.5-1.5B \
  --checkpoint jointc.pt \
  --prompt "style prompt" \
  --lyrics "lyrics" \
  --output semantic_latent.pt
```

Waveform decoding stays behind the approved LeVo/acoustic adapter and is not
silently replaced by a toy decoder.
