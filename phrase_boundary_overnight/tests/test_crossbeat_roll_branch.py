import torch
from src.crossbeat_roll_branch import CrossbeatBoundary, features_in_blocks
from src.score_roll_branch import RollBoundary
from src.diagnose_roll_invariance import example

torch.set_num_threads(2)


def test_parameters_rng_and_control_equivalence():
    a=RollBoundary('R',42).eval(); rng=torch.get_rng_state()
    b=CrossbeatBoundary(False,42).eval(); torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
    c=CrossbeatBoundary(True,42).eval()
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in c.parameters())==6121
    assert sum(p.numel() for p in c.parameters() if p.requires_grad)==3297
    for p,q in zip(a.parameters(),c.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    roll=torch.rand(2,8,2,128,4)
    torch.testing.assert_close(a.stem(roll),b.stem(roll),rtol=0,atol=0)


def test_crossbeat_information_not_erased():
    a=example([0,2,4,5,7,9,11,12]); b=example([12,11,9,7,5,4,2,0])
    old=CrossbeatBoundary(False,42).eval(); new=CrossbeatBoundary(True,42).eval()
    with torch.no_grad():
        assert float((old.stem(a)-old.stem(b)).abs().max())<1e-5
        assert float((new.stem(a)-new.stem(b)).abs().max())>1e-4
        # Whole-sequence pitch translation should remain insensitive away from pitch edges.
        shifted=example([12,14,16,17,19,21,23,24])
        torch.testing.assert_close(new.stem(a),new.stem(shifted),atol=2e-6,rtol=2e-6)


def test_masked_padding_and_chunk_halo():
    model=CrossbeatBoundary(True,42).eval(); roll=torch.rand(1,11,2,128,4)
    with torch.no_grad():
        whole=model.stem(roll)
        torch.testing.assert_close(whole,features_in_blocks(model,roll,3),atol=2e-6,rtol=2e-6)
        padded=torch.cat([roll,torch.full((1,3,2,128,4),1000.)],dim=1)
        mask=torch.tensor([[False]*11+[True]*3])
        output=model.stem(padded,mask)
        torch.testing.assert_close(whole,output[:,:11],atol=2e-6,rtol=2e-6)
        assert not output[:,11:].count_nonzero()


def test_core_training_has_finite_gradients():
    model=CrossbeatBoundary(True,42)
    x=torch.rand(2,8,58); roll=torch.rand(2,8,2,128,4)
    model(x,roll).square().mean().backward()
    grads=[p.grad for p in model.core.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0
    assert all(p.grad is None for p in model.stem.parameters())
