import unittest

import numpy as np
import pandas as pd
import torch

from src.goal5_downstream import BoundaryTransformer, decode, map_starts, match_f1


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


if __name__ == "__main__":
    unittest.main()
