from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - depends on the local environment
    torch = None
    nn = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this environment")
class JointCSmokeTest(unittest.TestCase):
    def test_forward_backward_and_sampling(self):
        from jointc.config import JointCConfig
        from jointc.data import collate_jointc, materialize_continuous_targets
        from jointc.dit import Id2LatentDiT
        from jointc.model import JointCModel
        from jointc.sampling import sample_semantic_ids
        from jointc.text import build_text_condition
        from jointc.train import _load_partial

        class Output:
            def __init__(self, hidden):
                self.last_hidden_state = hidden

        class TinyCausalBackbone(nn.Module):
            def __init__(self, vocab_size=64, hidden_size=32, layers=3):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            hidden_size, 4, batch_first=True
                        )
                        for _ in range(layers)
                    ]
                )
                self.norm = nn.LayerNorm(hidden_size)
                self.config = type("Config", (), {"hidden_size": hidden_size})()

            def get_input_embeddings(self):
                return self.embed_tokens

            def forward(
                self,
                inputs_embeds,
                attention_mask=None,
                use_cache=False,
                return_dict=True,
            ):
                hidden = inputs_embeds
                for layer in self.layers:
                    hidden = layer(hidden)
                return Output(self.norm(hidden))

        backbone = TinyCausalBackbone()
        config = JointCConfig(
            hidden_size=32,
            semantic_vocab_size=16,
            latent_dim=4,
            condition_dim=8,
            dit_hidden_size=32,
            dit_depth=2,
            dit_num_heads=4,
            unfreeze_last_backbone_layers=1,
        )
        model = JointCModel(backbone, config)
        self.assertNotIn("input_embeddings.weight", model.state_dict())
        self.assertIn("backbone.embed_tokens.weight", model.state_dict())
        batch = {
            "text_input_ids": torch.tensor([[1, 2, 3]]),
            "semantic_ids": torch.tensor([[4, 5, 6, 7]]),
            "audio_mask": torch.tensor([[True, True, True, True]]),
            "continuous_targets": torch.randn(1, 4, 4),
            "text_attention_mask": torch.tensor([[True, True, True]]),
        }
        output = model(**batch)
        self.assertEqual(tuple(output["condition"].shape), (1, 4, 8))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["flow_l1"]))
        self.assertTrue(torch.isfinite(output["z_abs"]))
        output["loss"].backward()
        ids = sample_semantic_ids(
            model,
            batch["text_input_ids"],
            max_new_tokens=4,
            seed=7,
        )
        self.assertEqual(ids.shape[0], 1)
        self.assertTrue(
            torch.equal(
                ids,
                sample_semantic_ids(
                    model,
                    batch["text_input_ids"],
                    max_new_tokens=4,
                    seed=7,
                ),
            )
        )
        condition = model.condition_for_ids(batch["text_input_ids"], ids)
        latent = model.sample_latent(
            ids,
            cond_z=condition,
            steps=2,
            seed=11,
        )
        self.assertEqual(latent.shape[0], 1)
        self.assertTrue(
            torch.equal(
                latent,
                model.sample_latent(
                    ids,
                    cond_z=condition,
                    steps=2,
                    seed=11,
                ),
            )
        )

        warmup = model.warmup_forward(
            semantic_ids=batch["semantic_ids"],
            audio_mask=batch["audio_mask"],
            continuous_targets=batch["continuous_targets"],
        )
        self.assertEqual(tuple(warmup["prediction"].shape), (1, 4, 4))

        dit = Id2LatentDiT(
            latent_dim=4,
            vocab_size=16,
            hidden_size=32,
            depth=2,
            num_heads=4,
            max_seq_len=8,
            z_cond_dim=8,
        )
        self.assertTrue(torch.equal(
            dit.pos_embed[0, 0, :16],
            torch.zeros_like(dit.pos_embed[0, 0, :16]),
        ))
        self.assertTrue(torch.equal(
            dit.pos_embed[0, 0, 16:],
            torch.ones_like(dit.pos_embed[0, 0, 16:]),
        ))
        x = torch.randn(1, 3, 4)
        ids_50hz = torch.tensor([[1, 2, 3, 4, 5, 6]])
        z = torch.randn(1, 3, 8)
        self.assertEqual(
            tuple(
                dit(
                    x,
                    torch.tensor([0.5]),
                    ids_50hz,
                    cond_z=z,
                    cond_is_50hz=True,
                ).shape
            ),
            (1, 3, 4),
        )

        posterior_batch = collate_jointc(
            [
                {
                    "text_input_ids": [0, 3],
                    "semantic_ids": [1, 2],
                    "vae_mu": [[0.1, 0.2], [0.3, 0.4]],
                    "vae_logvar": [[-1.0, -1.0], [-1.0, -1.0]],
                },
                {
                    "text_input_ids": [4],
                    "semantic_ids": [3],
                    "vae_mu": [[0.5, 0.6]],
                    "vae_logvar": [[-2.0, -2.0]],
                },
            ]
        )
        self.assertEqual(
            posterior_batch["text_attention_mask"].tolist(),
            [[True, True], [True, False]],
        )
        self.assertTrue(
            torch.equal(
                materialize_continuous_targets(
                    posterior_batch,
                    sample_posterior=False,
                ),
                posterior_batch["vae_mu"],
            )
        )
        self.assertEqual(
            build_text_condition(
                "Pop Rock",
                "[Verse]\nHello\n[Verse]\nWorld",
            ),
            (
                "[prompt_start] pop rock [prompt_end];"
                "[lyrics_start] [verse_1];hello;[verse_2];world [lyrics_end]"
            ),
        )

        # A warmup DiT has no z projection. Its common transformer weights
        # should still load directly into the JointC DiT.
        warmup_dit = Id2LatentDiT(
            latent_dim=4,
            vocab_size=16,
            hidden_size=32,
            depth=2,
            num_heads=4,
            max_seq_len=8,
            z_cond_dim=0,
        )
        with torch.no_grad():
            warmup_dit.x_embedder.weight.fill_(0.125)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "warmup.pt"
            torch.save({"model": warmup_dit.state_dict()}, checkpoint)
            _load_partial(model.dit, str(checkpoint))
        self.assertTrue(
            torch.allclose(
                model.dit.x_embedder.weight,
                torch.full_like(model.dit.x_embedder.weight, 0.125),
            )
        )
        self.assertEqual(
            tuple(
                dit(
                    x,
                    torch.tensor([0.5]),
                    torch.tensor([[1, 2, 3]]),
                    cond_z=z,
                    history_x=torch.randn_like(x),
                ).shape
            ),
            (1, 3, 4),
        )


if __name__ == "__main__":
    unittest.main()
