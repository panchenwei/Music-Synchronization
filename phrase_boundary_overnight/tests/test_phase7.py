import pytest
import torch

from src.phase7_models import Phase7BoundaryModel, ResidualTemporalConv


@pytest.mark.parametrize("kind", ["A", "B", "C"])
def test_phase7_shape_and_padding_invariance(kind):
    torch.manual_seed(7)
    model = Phase7BoundaryModel(kind, dropout=0.0).eval()
    x = torch.randn(2, 11, 9)
    padded = torch.cat([x, torch.randn(2, 5, 9)], 1)
    mask = torch.zeros(2, 16, dtype=torch.bool); mask[:, 11:] = True
    changed = padded.clone(); changed[:, 11:] = 1000
    with torch.no_grad():
        short = model(x)
        long = model(padded, mask)[:, :11]
        altered = model(changed, mask)[:, :11]
    assert short.shape == (2, 11)
    assert torch.allclose(short, long, atol=1e-6)
    assert torch.allclose(long, altered, atol=1e-6)


def test_phase7_conv_preserves_length_and_rejects_full_mask():
    conv = ResidualTemporalConv(32, 5, 0.0)
    assert conv(torch.randn(3, 7, 32)).shape == (3, 7, 32)
    model = Phase7BoundaryModel("B")
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 9), torch.ones(1, 4, dtype=torch.bool))
