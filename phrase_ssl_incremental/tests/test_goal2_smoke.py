import unittest

import numpy as np
import torch

from src.goal2_smoke import METADATA, ReconstructionModel, corrupt, reconstruction_loss, sample_spans, validity


class SmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.randn(2, 64, 58)
        self.valid = torch.ones_like(self.x, dtype=torch.bool)
        self.valid[:, :, METADATA] = False
        self.lengths = torch.tensor([64, 25])
        self.spans, self.records = sample_spans(self.lengths, torch.Generator().manual_seed(2))

    def test_contiguous_spans_inside_real_length(self):
        for row, intervals in enumerate(self.records):
            for start, stop in intervals:
                self.assertTrue(4 <= stop - start <= 8)
                self.assertLessEqual(stop, self.lengths[row])
                self.assertTrue(self.spans[row, start:stop].all())
                self.assertEqual(int(self.spans[row].sum()), stop - start)

    def test_padding_does_not_change_valid_predictions(self):
        net = ReconstructionModel().eval()
        args = corrupt(self.x, self.valid, self.spans, self.lengths)
        out = net(*args[:3], self.lengths)
        self.assertEqual(out.shape, (2, 64, 58))
        changed = self.x.clone()
        changed[1, 25:] = 1e6
        other = corrupt(changed, self.valid, self.spans, self.lengths)
        torch.testing.assert_close(out[1, :25], net(*other[:3], self.lengths)[1, :25], rtol=0, atol=0)
        self.assertTrue(args[3][1, 25:].all())

    def test_loss_gradient_only_masked_observed_nonpadding(self):
        self.valid[0, :, 0] = False
        prediction = torch.randn_like(self.x, requires_grad=True)
        loss, _, selected = reconstruction_loss(prediction, self.x, self.valid, self.spans, self.lengths)
        loss.backward()
        self.assertTrue((prediction.grad[~selected] == 0).all())
        self.assertFalse(selected[1, 25:].any())
        self.assertFalse(selected[:, :, METADATA].any())
        self.assertFalse(selected[0, :, 0].any())
        target = self.x.clone()
        target[~selected] = float("nan")
        torch.testing.assert_close(loss, reconstruction_loss(prediction, target, self.valid, self.spans, self.lengths)[0])

    def test_distinct_tokens_and_no_target_or_metadata_copy(self):
        self.valid[0, 0, 3] = False
        args = corrupt(self.x, self.valid, self.spans, self.lengths)
        self.assertTrue(args[2][0, 0, 3])
        net = ReconstructionModel()
        self.assertIsNot(net.mask_token, net.missing_token)
        altered = self.x.clone()
        altered[args[1] | ~self.valid] = 1234
        other = corrupt(altered, self.valid, self.spans, self.lengths)
        for a, b in zip(args, other):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(net(*args[:3], self.lengths), net(*other[:3], self.lengths), rtol=0, atol=0)

    def test_validity_rejects_imputed_raw_missing(self):
        values = np.ones((64, 58), np.float32)
        times = np.arange(64, dtype=float)
        dynamics = np.ones(64)
        times[10] = np.nan
        dynamics[20] = np.nan
        observed = validity(values, times, dynamics)
        self.assertFalse(observed[9, 0])
        self.assertFalse(observed[10, 0])
        self.assertFalse(observed[20, 3])
        self.assertFalse(observed[20, 4])
        self.assertFalse(observed[:, METADATA].any())
        self.assertFalse(observed[:, 25:30].any())


if __name__ == "__main__":
    unittest.main()
