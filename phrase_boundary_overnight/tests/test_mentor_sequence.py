import copy
import numpy as np
import pytest
import torch
from src.mentor_sequence_models import make_model,ContextSampler
from src.models import Normalizer


def test_same_supervised_draws_with_full_context():
    data={str(i):dict(curves=np.random.default_rng(i).normal(size=(2,n,58)).astype('float32'),labels=(np.arange(n)%11==0).astype('float32'),label_mask=np.ones(n,'float32')) for i,n in enumerate((31,96,155))}
    norm=Normalizer(np.zeros(58),np.ones(58));a=ContextSampler(data,norm,42,False);b=ContextSampler(data,norm,42,True)
    for _ in range(3):
        x=a.batch();y=b.batch();assert a.state()==b.state()
        for i,(sa,sb) in enumerate(zip(a.last_selection,b.last_selection)):
            assert sa[:3]==sb[:3]
            n=sa[-1];offset=sb[-2]
            for j in (0,1,2):torch.testing.assert_close(x[j][i,:n],y[j][i,offset:offset+n],atol=0,rtol=0)
            assert x[2][i].sum()==y[2][i].sum()


def test_transformer_padding_and_long_context_effect():
    torch.set_num_threads(2);m=make_model('Tfull',42).eval();x=torch.randn(1,180,58)
    with torch.no_grad():
        a=m(x);pad=torch.arange(197)[None]>=180
        b=m(torch.cat([x,torch.randn(1,17,58)*100],dim=1),padding_mask=pad)
        torch.testing.assert_close(a,b[:,:180],atol=2e-6,rtol=2e-6)
        changed=x.clone();changed[:,130:]+=3
        assert (m(changed)[:,:30]-a[:,:30]).abs().max()>1e-5
        assert (m(x.flip(1)).flip(1)-a).abs().max()>1e-5
    m64=make_model('T64',42)
    for k,v in m.state_dict().items():assert torch.equal(v,m64.state_dict()[k])


def test_transformer_finite_grad_and_tiny_overfit():
    torch.set_num_threads(2);m=make_model('Tfull',42);x=torch.randn(2,48,58);y=(torch.arange(48)%8==0).float().repeat(2,1)
    opt=torch.optim.AdamW(m.parameters(),lr=.003);criterion=torch.nn.BCEWithLogitsLoss()
    losses=[]
    for _ in range(100):
        opt.zero_grad();loss=criterion(m(x),y);assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss.detach()))
    assert np.mean(losses[-10:])<np.mean(losses[:10])*.3


def test_resume_rng_optimizer_exact():
    torch.set_num_threads(2);m=make_model('Tfull',42);x=torch.randn(2,48,58);opt=torch.optim.AdamW(m.parameters(),lr=.001)
    def update():
        opt.zero_grad();loss=m(x).square().mean();loss.backward();opt.step();return float(loss.detach())
    update();weights=copy.deepcopy(m.state_dict());state=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state()
    loss=update();after=copy.deepcopy(m.state_dict());m.load_state_dict(weights);opt.load_state_dict(state);torch.set_rng_state(rng)
    assert update()==loss
    for k,v in after.items():assert torch.equal(v,m.state_dict()[k])
