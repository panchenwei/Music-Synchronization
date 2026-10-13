import unittest

import numpy as np
import torch

from src.goal4_pretrain import interpolation, macro, normalization


class PretrainControlTests(unittest.TestCase):
    def test_interpolation_never_reads_hidden_answers(self):
        values = torch.tensor([0., 1., 999., 999., 4., 5.])[None, :, None].expand(1, 6, 58).clone()
        hidden = torch.zeros_like(values, dtype=torch.bool)
        hidden[:, 2:4] = True
        hidden[:, :, 25:] = True
        valid = torch.ones_like(hidden)
        a = interpolation(values, valid, hidden, torch.tensor([6]))
        changed = values.clone()
        changed[hidden] = -1e6
        b = interpolation(changed, valid, hidden, torch.tensor([6]))
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a[0, :, 0], torch.arange(6, dtype=torch.float32))
        self.assertTrue((a[:, :, 25:] == 0).all())

    def test_normalization_excludes_unobserved_imputations(self):
        x = np.ones((8, 58), np.float32)
        valid = np.ones_like(x, bool)
        valid[4:] = False
        a = normalization([(x, valid, {})])
        x[4:] = 1e6
        b = normalization([(x, valid, {})])
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right)

    def test_macro_equal_work_weight(self):
        rows = [{"piece_id": "a", "group": "score", "mse": 1.0}] * 20
        rows += [{"piece_id": "b", "group": "score", "mse": 3.0}]
        self.assertEqual(macro(rows), 2.0)


if __name__ == "__main__":
    unittest.main()
