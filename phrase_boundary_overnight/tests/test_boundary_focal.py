import torch
from torch.nn import functional as F
from src.boundary_focal import WeightedFocal


def test_gamma_zero_is_exact_weighted_bce_and_gradient():
    x = torch.tensor([-3., -.2, 0., .7, 4.], requires_grad=True)
    y = torch.tensor([0., 1., 0., 1., 1.])
    a = WeightedFocal(torch.tensor(10.), 0)(x, y)
    b = F.binary_cross_entropy_with_logits(x, y, pos_weight=torch.tensor(10.), reduction='none')
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(a.sum(), x)[0], torch.autograd.grad(b.sum(), x)[0], rtol=0, atol=0)


def test_matches_alpha_balanced_formula_without_double_weighting():
    x = torch.tensor([-4., -.2, 0., .7, 3.], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0., 1., 0., 1., 1.], dtype=torch.float64)
    w = 10.
    for gamma in (1, 2):
        p = x.sigmoid()
        pt = p*y + (1-p)*(1-y)
        reference = F.binary_cross_entropy_with_logits(x, y, reduction='none') * (1-pt)**gamma * (1+(w-1)*y)
        a = WeightedFocal(torch.tensor(w, dtype=x.dtype), gamma)(x, y)
        torch.testing.assert_close(a, reference)
        torch.testing.assert_close(torch.autograd.grad(a.sum(), x, retain_graph=True)[0], torch.autograd.grad(reference.sum(), x, retain_graph=True)[0])


def test_mask_and_extreme_logits_have_finite_gradients():
    x = torch.tensor([-1000., 1000., -1000., 1000., .3], requires_grad=True)
    y = torch.tensor([0., 1., 1., 0., 0.])
    mask = torch.tensor([1., 1., 1., 1., 0.])
    for gamma in (1, 2):
        loss = (WeightedFocal(torch.tensor(10.), gamma)(x, y)*mask).sum()/mask.sum()
        grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
        assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad[-1] == 0


def test_easy_examples_are_downweighted_without_inverting_class_weight():
    x = torch.tensor([-4., 4., .1, -.1])
    y = torch.tensor([0., 1., 0., 1.])
    base = WeightedFocal(torch.tensor(10.), 0)(x, y)
    focal = WeightedFocal(torch.tensor(10.), 2)(x, y)
    assert (focal/base)[0] < (focal/base)[2] and (focal/base)[1] < (focal/base)[3]
