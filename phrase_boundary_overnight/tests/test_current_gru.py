import pytest
import torch
from src.current_gru_models import CurrentBiGRU
from src.recurrence_depth_models import make_model


def test_shape_padding_and_shared_projection():
    torch.set_num_threads(2)
    m = CurrentBiGRU(42).eval()
    assert sum(p.numel() for p in m.parameters()) == 6005
    c = make_model('C3', 42)
    torch.testing.assert_close(m.input_projection.weight, c.input_projection.weight, atol=0, rtol=0)
    x = torch.randn(2, 12, 58)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, 7:] = True
    out = m(x, mask)
    altered = x.clone(); altered[mask] = 1e6
    torch.testing.assert_close(out, m(altered, mask), atol=0, rtol=0)
    torch.testing.assert_close(out[0, :7], m(x[:1, :7])[0], atol=1e-6, rtol=1e-5)
    assert out.shape == (2, 12) and torch.all(out[mask] == 0)
    with pytest.raises(ValueError):
        m(x, torch.ones_like(mask))


def test_tiny_overfit_and_bidirectional_dependency():
    torch.set_num_threads(2)
    m = CurrentBiGRU(43)
    x = torch.randn(2, 12, 58)
    y = (x[:, :, 0] > 0).float()
    opt = torch.optim.Adam(m.parameters(), lr=.02)
    for _ in range(100):
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), y)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
    m.eval()
    assert ((m(x) > 0) == y.bool()).float().mean() > .99
    xx = x.clone().requires_grad_()
    m(xx)[:, 5].sum().backward()
    assert xx.grad[:, :5].abs().sum() > 0 and xx.grad[:, 6:].abs().sum() > 0
