"""TC-VIS: the visual scene features (Week 11): the weight conversion, the frame mapping,
and, where the weights are installed, the encoder itself."""
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.analysis import visual as V

MODELS = Path(__file__).resolve().parents[2] / "models"


class VisualTest(unittest.TestCase):
    def test_to_frames_takes_the_nearest_earlier_sample(self):
        vis = {"time_s": np.array([0.0, 0.5, 1.0]), "embedding": np.eye(3, V.DIM, dtype=np.float32),
               "meta": {"fps": 2.0}}
        emb, valid = V.to_frames(vis, 60, 30.0)                  # 2 s at 30 fps
        self.assertEqual(emb.shape, (60, V.DIM))
        self.assertEqual(int(np.argmax(emb[14])), 0)
        self.assertEqual(int(np.argmax(emb[15])), 1)
        self.assertEqual(int(np.argmax(emb[45])), 2)
        self.assertTrue(valid[:45].all() and not valid[46:].any())

    def test_convert_state_dict_keys(self):
        import torch
        sd = {"embeddings.cls_token": torch.zeros(1, 1, 4), "embeddings.mask_token": torch.zeros(1, 4),
              "embeddings.position_embeddings": torch.zeros(1, 5, 4),
              "embeddings.patch_embeddings.projection.weight": torch.zeros(4, 3, 14, 14),
              "embeddings.patch_embeddings.projection.bias": torch.zeros(4),
              "layernorm.weight": torch.ones(4), "layernorm.bias": torch.zeros(4)}
        a = "encoder.layer.0."
        for x in ("weight", "bias"):
            for k in ("norm1", "norm2", "attention.output.dense", "mlp.fc1", "mlp.fc2"):
                sd[a + f"{k}.{x}"] = torch.zeros(4, 4) if x == "weight" else torch.zeros(4)
            for q, v in (("query", 1.0), ("key", 2.0), ("value", 3.0)):
                sd[a + f"attention.attention.{q}.{x}"] = torch.full((4, 4) if x == "weight" else (4,), v)
        sd[a + "layer_scale1.lambda1"], sd[a + "layer_scale2.lambda1"] = torch.ones(4), torch.ones(4)
        out = V.convert_state_dict(sd)
        self.assertEqual(out["blocks.0.attn.qkv.weight"].shape, (12, 4))
        self.assertEqual(out["blocks.0.attn.qkv.bias"].tolist(), [1.0] * 4 + [2.0] * 4 + [3.0] * 4)
        self.assertEqual(out["mask_token"].shape, (1, 4))
        self.assertIn("blocks.0.ls2.gamma", out)

    @unittest.skipUnless((MODELS / V.WEIGHTS).is_file(), "DINOv2 weights not installed")
    def test_encoder_is_deterministic_and_unit_length(self):
        m = V.load_encoder(MODELS)
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
        a, b = V.embed_batch(m, frames, "cpu"), V.embed_batch(m, frames, "cpu")
        self.assertEqual(a.shape, (2, V.DIM))
        np.testing.assert_allclose(a, b, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
