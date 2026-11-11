"""TC-EBC: Engine B v2 contrastive objective on window embeddings (Week 10)."""
import unittest

import numpy as np
import torch

from assemblyvision.tools.train_engine_b import supcon_loss, window_embeddings


class ContrastTest(unittest.TestCase):
    def test_windows_inside_one_phase_only(self):
        h = torch.arange(2 * 3 * 12, dtype=torch.float32).reshape(2, 3, 12)
        P = torch.tensor([[0] * 4 + [0, 0, 1, 1] + [1] * 4, [2] * 8 + [-1] * 4])
        M = torch.ones(2, 12)
        M[1, 6:] = 0
        z, lab = window_embeddings(h, P, M, 4)
        self.assertEqual(lab.tolist(), [0, 1, 2])                # mixed, other and padded windows are left out
        np.testing.assert_allclose(z.norm(dim=1).numpy(), 1.0, atol=1e-6)
        np.testing.assert_allclose(z[0].numpy(), torch.nn.functional.normalize(h[0, :, :4].mean(1), dim=0).numpy(),
                                   atol=1e-6)

    def test_loss_prefers_phase_clusters(self):
        lab = torch.tensor([0, 0, 1, 1])
        good = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1, 0.1], [0, 1], [0.1, 1]]), dim=1)
        bad = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [0, 1], [1, 0.1], [0.1, 1]]), dim=1)
        self.assertLess(float(supcon_loss(good, lab, 0.1)), float(supcon_loss(bad, lab, 0.1)))
        self.assertEqual(float(supcon_loss(good[:1], lab[:1], 0.1)), 0.0)
        self.assertEqual(float(supcon_loss(good[[0, 2]], torch.tensor([0, 1]), 0.1)), 0.0)    # no positive pair


if __name__ == "__main__":
    unittest.main()
