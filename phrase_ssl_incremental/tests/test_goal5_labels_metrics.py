import unittest

import numpy as np
import pandas as pd
import torch

from src.goal5_downstream import BoundaryTransformer, decode, map_starts, match_f1
from src.goal6_input_consistency import consistent_supervision
from src.goal7_small_bigru import SmallBiGRU
from src.goal8_novelty import novelty_features
from src.goal9_interval_decoder import decode as interval_decode, fit_prior
from src.goal10_nested_decoder_cv import build_nested_splits
from src.goal11_interval_ranking import interval_ranking_loss
from src.goal12_global_crf_smoke import path_log_partition, sequence_nll


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

    def test_goal8_novelty_detects_context_change_without_edge_artifacts(self):
        values = torch.zeros(1, 24, 58)
        values[:, 12:, 0:7] = 2
        valid = torch.ones_like(values, dtype=torch.bool)
        novelty, observed = novelty_features(values, valid, torch.tensor([24]))
        self.assertTrue((novelty[0, 12] > 0).all())
        self.assertEqual(int(novelty[0, :, 2].argmax()), 12)
        self.assertFalse(observed[0, :8, 2].any())
        self.assertFalse(observed[0, 17:, 2].any())

    def test_goal9_interval_prior_is_train_only_and_decoder_is_ordered(self):
        labels = {"a": (np.eye(1, 64, 8).ravel() + np.eye(1, 64, 24).ravel() + np.eye(1, 64, 40).ravel(), np.ones(64, bool))}
        prior = fit_prior(labels, ["a"])
        self.assertEqual(prior["works"], 1)
        self.assertEqual(prior["intervals"], 2)
        probability = np.full(64, 0.01)
        probability[[8, 24, 40]] = [0.9, 0.8, 0.85]
        prediction = interval_decode(probability, np.ones(64, bool), 0.5, prior)
        self.assertEqual(prediction.tolist(), sorted(prediction.tolist()))
        self.assertTrue(set(prediction).issubset({8, 24, 40}))

    def test_goal10_nested_splits_are_disjoint_and_outer_once(self):
        pieces = [f"piece_{index}" for index in range(38)]
        splits = build_nested_splits(pieces)
        self.assertEqual(len(splits), 5)
        outer = []
        for split in splits:
            self.assertFalse(set(split["train"]) & set(split["inner"]))
            self.assertFalse(set(split["train"]) & set(split["outer"]))
            self.assertFalse(set(split["inner"]) & set(split["outer"]))
            self.assertEqual(set(split["train"] + split["inner"] + split["outer"]), set(pieces))
            outer.extend(split["outer"])
        self.assertEqual(sorted(outer), sorted(pieces))

    def test_goal11_interval_ranking_prefers_true_next_boundary(self):
        labels = torch.zeros(1, 12)
        labels[0, [2, 8]] = 1
        coverage = torch.ones(1, 12, dtype=torch.bool)
        potential = torch.zeros(256)
        good = torch.zeros(1, 12, requires_grad=True)
        bad = torch.zeros(1, 12, requires_grad=True)
        good.data[0, 8] = 4
        bad.data[0, 7] = 4
        good_loss, count = interval_ranking_loss(good, labels, coverage, potential)
        bad_loss, _ = interval_ranking_loss(bad, labels, coverage, potential)
        self.assertEqual(count, 1)
        self.assertLess(float(good_loss.detach()), float(bad_loss.detach()))
        good_loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())

    def test_goal12_global_partition_matches_brute_force(self):
        unary = torch.tensor([0.2, -0.3, 0.7, -0.1, 0.4, -0.5], requires_grad=True)
        transition = torch.linspace(-0.2, 0.3, 4)
        scores = [unary.new_zeros(())]
        for mask in range(1, 1 << len(unary)):
            starts = [index for index in range(len(unary)) if mask & (1 << index)]
            gaps = [right - left for left, right in zip(starts[:-1], starts[1:])]
            if all(2 <= gap <= 4 for gap in gaps):
                scores.append(unary[starts].sum() + sum((transition[gap - 1] for gap in gaps), unary.new_zeros(())))
        expected = torch.logsumexp(torch.stack(scores), dim=0)
        actual = path_log_partition(unary, transition, 2, 4)
        self.assertLess(float((expected - actual).abs().detach()), 1e-6)
        labels = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.float32)
        loss = sequence_nll(unary, labels, transition, 2, 4)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(unary.grad).all())
        self.assertTrue(bool((unary.grad != 0).any()))


if __name__ == "__main__":
    unittest.main()
