import unittest

import numpy as np
import pandas as pd
import torch

from src.goal5_downstream import BoundaryTransformer, decode, map_starts, match_f1
from src.goal6_input_consistency import consistent_supervision
from src.goal7_small_bigru import SmallBiGRU


class DownstreamTests(unittest.TestCase):
    def test_label_coordinate_repeat_and_fraction(self):
        annotation = pd.DataFrame({"mc": [1, 2], "mc_onset": ["1/8", "1/2"], "phraseend": ["{", "}"]})
        path = pd.DataFrame({"mc": [1, 2, 1, 2], "visit": [1, 1, 2, 2], "start_qb": [0., 3., 6., 9.], "duration_qb": [3.] * 4})
        labels, coverage, rows = map_starts(annotation, path, 12, "synthetic")
        self.assertEqual([r["quarterbeat"] for r in rows if r["is_start"]], [.5, 6.5])
        self.assertEqual(np.flatnonzero(labels).tolist(), [6])
        self.assertFalse(coverage[0])
        self.assertTrue(coverage[11])

    def test_missing_label_coordinate_stops(self):
        annotation = pd.DataFrame({"mc": [1], "mc_onset": [np.nan], "phraseend": ["{"]})
        path = pd.DataFrame({"mc": [1], "visit": [1], "start_qb": [0.], "duration_qb": [3.]})
        with self.assertRaisesRegex(ValueError, "mc_onset"):
            map_starts(annotation, path, 3, "bad")

    def test_one_to_one_matching(self):
        self.assertEqual(match_f1([9, 10, 11], [10], 1), (1 / 3, 1.0, .5))
        self.assertEqual(match_f1([9, 10], [10, 11], 1)[2], 1.0)
        self.assertEqual(match_f1([9], [10], 0)[2], 0.0)

    def test_decoder_plateau_threshold_and_coverage(self):
        probability = np.array([.9, .2, .8, .8, .1, .7])
        coverage = np.array([False, True, True, True, True, True])
        self.assertEqual(decode(probability, coverage, .75).tolist(), [2])
        self.assertEqual(decode(probability, coverage, .65).tolist(), [2, 5])

    def test_model_variable_length(self):
        model = BoundaryTransformer().eval()
        x = torch.randn(2, 91, 58)
        hidden = torch.zeros_like(x, dtype=torch.bool)
        lengths = torch.tensor([91, 37])
        with torch.no_grad():
            out = model(x, hidden, hidden, lengths)
        self.assertEqual(out.shape, (2, 91))
        self.assertTrue(torch.isfinite(out).all())

    def test_goal6_uses_pretrain_visible_columns_only(self):
        x = torch.randn(1, 8, 58)
        valid = torch.ones_like(x, dtype=torch.bool)
        lengths = torch.tensor([8])
        labels = {"piece": (np.zeros(8, np.float32), np.ones(8, bool))}
        clean, hidden, missing, _, _ = consistent_supervision(
            x, valid, lengths, [{"piece_id": "piece"}], labels
        )
        self.assertTrue(hidden[:, :, 25:58].all())
        self.assertTrue(hidden[:, :, [7, 8, 9, 10, 11]].all())
        self.assertFalse(hidden[:, :, 0:7].any())
        self.assertTrue((clean[hidden] == 0).all())
        self.assertFalse(missing.any())

    def test_goal7_bigru_shape_padding_and_parameter_count(self):
        model = SmallBiGRU().eval()
        clean = torch.randn(2, 13, 58)
        hidden = torch.zeros_like(clean, dtype=torch.bool)
        missing = torch.zeros_like(clean, dtype=torch.bool)
        lengths = torch.tensor([13, 7])
        with torch.no_grad():
            output = model(clean, hidden, missing, lengths)
        self.assertEqual(output.shape, (2, 13))
        self.assertTrue(torch.isfinite(output).all())
        # 6005 model parameters plus two 58-D input-state tokens.
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 6121)


if __name__ == "__main__":
    unittest.main()
