import copy
import numpy as np
import torch
from src.score_roll_branch import RollBoundary,RollSampler
from src.score_context_study import make_model
from src.phase7_models import CurvePieceBalancedSampler
from src.models import Normalizer


def test_initial_prediction_equals_old_and_random_stem_matches():
    torch.set_num_threads(2)
    a=RollBoundary('L',42).eval();b=RollBoundary('R',42).eval();old=make_model('P',42).eval()
    for x,y in zip(a.stem.parameters(),b.stem.parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    x=torch.randn(2,5,58);x[...,34:]=0
    roll=torch.rand(2,5,2,128,4)
    with torch.no_grad():
        torch.testing.assert_close(a(x,roll),old(x),rtol=0,atol=0)
        torch.testing.assert_close(a(x,roll),b(x,roll),rtol=0,atol=0)
    assert not any(p.requires_grad for p in b.stem.parameters())


def test_sampler_matches_old_rng_window_and_resume():
    n=70;rng=np.random.default_rng(44)
    item=dict(curves=rng.normal(size=(3,n,58)).astype(np.float32),labels=np.arange(n,dtype=np.float32),
              label_mask=np.ones(n,np.float32),piano_roll=np.broadcast_to(np.arange(n,dtype=np.float32)[:,None,None,None],(n,2,128,4)).copy())
    data={'a':item};norm=Normalizer(np.zeros(58),np.ones(58))
    a=RollSampler(data,norm,64,3,42);b=CurvePieceBalancedSampler(data,norm,64,3,42)
    for _ in range(3):
        aa=a.batch();bb=b.batch()
        for x,y in zip(aa[:4],bb):torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(aa[4][:,:,0,0,0],aa[1],rtol=0,atol=0)
    saved=copy.deepcopy(a.state());expected=a.batch();a.load_state(saved)
    for x,y in zip(expected,a.batch()):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_branch_gradient_and_padding_are_finite():
    model=RollBoundary('L',42).eval()
    with torch.no_grad():model.core.input_projection.weight[:,34:]=torch.linspace(-.05,.05,24)[None,:]
    x=torch.randn(1,6,58);roll=torch.rand(1,6,2,128,4);mask=torch.tensor([[False]*4+[True]*2])
    a=model(x,roll,padding_mask=mask)
    x2=x.clone();x2[:,4:]=1000;roll2=roll.clone();roll2[:,4:]=1000
    b=model(x2,roll2,padding_mask=mask)
    torch.testing.assert_close(a[:,:4],b[:,:4])
    a[:,:4].square().mean().backward()
    grads=[p.grad for p in model.stem.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>1e-6
