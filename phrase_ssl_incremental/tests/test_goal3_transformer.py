import unittest

import numpy as np
import torch

from src.goal2_smoke import METADATA, corrupt
from src.goal3_transformer_smoke import FullPieceTransformer, bucket_batches, mask_spans, masked_loss


class TransformerSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.x = torch.randn(2, 73, 58)
        self.valid = torch.ones_like(self.x, dtype=torch.bool)
        self.valid[:, :, METADATA] = False
        self.lengths = torch.tensor([73, 39])
        self.spans, self.records = mask_spans(self.lengths, torch.Generator().manual_seed(42))

    def test_dynamic_lengths_and_contiguous_span_bounds(self):
        for row, records in enumerate(self.records):
            for start, stop in records:
                self.assertTrue(4 <= stop - start <= 8)
                self.assertLessEqual(stop, self.lengths[row])
                self.assertTrue(self.spans[row, start:stop].all())
            self.assertFalse(self.spans[row, self.lengths[row]:].any())
        model = FullPieceTransformer().eval()
        for length in (13, 64, 73, 129):
            x = torch.randn(1, length, 58)
            lens = torch.tensor([length])
            mask, _ = mask_spans(lens, torch.Generator().manual_seed(3))
            args = corrupt(x, torch.ones_like(x, dtype=torch.bool), mask, lens)
            self.assertEqual(model(*args[:3], lens).shape, (1, length, 58))

    def test_bucket_collation_preserves_whole_sequences(self):
        samples = [(np.full((n, 58), i, np.float32), np.ones((n, 58), bool), str(n)) for i, n in enumerate((337, 121, 288))]
        batches = bucket_batches(samples)
        self.assertEqual(sorted(int(n) for batch in batches for n in batch[2]), [121, 288, 337])
        self.assertEqual([tuple(batch[0].shape) for batch in batches], [(1, 121, 58), (2, 337, 58)])
        for x, valid, lens, ids in batches:
            for row, n in enumerate(lens):
                self.assertEqual(int(n), int(ids[row]))
                self.assertFalse(valid[row, n:].any())

    def test_padding_and_other_batch_member_invariance(self):
        model = FullPieceTransformer().eval()
        args = corrupt(self.x, self.valid, self.spans, self.lengths)
        output = model(*args[:3], self.lengths)
        changed = self.x.clone()
        changed[1, 39:] = 1e6
        other = corrupt(changed, self.valid, self.spans, self.lengths)
        torch.testing.assert_close(output[1, :39], model(*other[:3], self.lengths)[1, :39], atol=1e-6, rtol=1e-5)
        solo = [arg[1:2, :39] for arg in args[:3]]
        torch.testing.assert_close(output[1, :39], model(*solo, torch.tensor([39]))[0], atol=1e-6, rtol=1e-5)
        self.assertTrue((output[1, 39:] == 0).all())

    def test_masked_loss_and_gradient_selectivity(self):
        self.valid[0, :, 3] = False
        prediction = torch.randn_like(self.x, requires_grad=True)
        loss, _, selected = masked_loss(prediction, self.x, self.valid, self.spans, self.lengths)
        loss.backward()
        self.assertTrue((prediction.grad[~selected] == 0).all())
        self.assertFalse(selected[0, :, 3].any())
        self.assertFalse(selected[1, 39:].any())
        self.assertFalse(selected[:, :, METADATA].any())
        changed = self.x.clone()
        changed[~selected] = float("nan")
        torch.testing.assert_close(loss, masked_loss(prediction, changed, self.valid, self.spans, self.lengths)[0])

    def test_hidden_answers_and_position_metadata_cannot_be_copied(self):
        args = corrupt(self.x, self.valid, self.spans, self.lengths)
        changed = self.x.clone()
        changed[args[1] | ~self.valid] = -1e6
        other = corrupt(changed, self.valid, self.spans, self.lengths)
        for left, right in zip(args, other):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        model = FullPieceTransformer().eval()
        self.assertIsNot(model.mask_token, model.missing_token)
        torch.testing.assert_close(model(*args[:3], self.lengths), model(*other[:3], self.lengths), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
