"""Approximately parameter-matched BiGRU on the current 58-dimensional input."""
import torch
from torch import nn
from .recurrence_depth_models import make_model as cnn_model


class CurrentBiGRU(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        reference = cnn_model('C3', seed)
        self.input_projection = reference.input_projection
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1901)
            self.rnn = nn.GRU(32, 14, num_layers=1, batch_first=True, bidirectional=True)
            self.final_norm = nn.LayerNorm(28)
            self.output = nn.Linear(28, 1)
        self.dropout = nn.Dropout(.2)

    def forward(self, inputs, padding_mask=None):
        if inputs.ndim != 3 or inputs.shape[-1] != 58:
            raise ValueError('Expected [batch,time,58]')
        mask = torch.zeros(inputs.shape[:2], dtype=torch.bool, device=inputs.device) if padding_mask is None else padding_mask.bool()
        if mask.shape != inputs.shape[:2]:
            raise ValueError('Mask shape mismatch')
        lengths = (~mask).sum(1)
        expected = torch.arange(inputs.shape[1], device=inputs.device)[None] >= lengths[:, None]
        if (lengths == 0).any() or not torch.equal(mask, expected):
            raise ValueError('Only nonempty right-padded sequences are supported')
        h = self.input_projection(inputs.masked_fill(mask[..., None], 0)).masked_fill(mask[..., None], 0)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        z, _ = self.rnn(packed)
        z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=inputs.shape[1])
        return self.output(self.final_norm(self.dropout(z))).squeeze(-1).masked_fill(mask, 0)


def make_model(kind, seed):
    assert kind == 'G'
    return CurrentBiGRU(seed)
