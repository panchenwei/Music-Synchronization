import torch
from src.recurrence_width_models import make_model, widen


def test_initial_eval_matches_narrow_padding_and_rng():
    torch.set_num_threads(2)
    for seed in (42,43):
        old=make_model('O',seed);rng=torch.get_rng_state().clone();wide=widen(old)
        assert torch.equal(rng,torch.get_rng_state())
        assert sum(p.numel() for p in old.parameters())==3297
        assert sum(p.numel() for p in wide.parameters())==8641
        assert wide.frontend.convs[0].groups==64 and len(wide.blocks)==0
        x=torch.randn(2,31,58);mask=torch.zeros(2,31,dtype=torch.bool);mask[1,23:]=True
        old.eval();wide.eval()
        torch.testing.assert_close(old(x,padding_mask=mask),wide(x,padding_mask=mask),atol=2e-6,rtol=2e-6)
        a=wide(x,padding_mask=mask);x[mask]=99999
        torch.testing.assert_close(a,wide(x,padding_mask=mask),atol=0,rtol=0)
        assert torch.equal(a[mask],torch.zeros_like(a[mask]))


def test_dropout_breaks_duplicate_channel_symmetry_during_training():
    m=make_model('W',42);m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001)
    x=torch.randn(4,64,58);y=(torch.rand(4,64)<.1).float()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x),y);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    opt.step();w=m.input_projection.weight.detach()
    assert float((w[:32]-w[32:]).abs().max())>1e-6
