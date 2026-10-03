import numpy as np
import torch
from src.phrase_end_targets import end_targets


def test_first_end_and_unknown_suffix():
    y,m,_=end_targets([0,5,10],[3,8,13],16)
    assert list(np.flatnonzero(y*m))==[3,8,13]
    assert m[0]==0 and m[14:].sum()==0


def test_ambiguous_and_interlocking():
    y,m,_=end_targets([0,4,8],[2.5,4,9],12)
    assert m[2]==m[3]==0 and y[4]*m[4]==1
    assert y[9]*m[9]==1


def test_empty_end_not_negative_dataset():
    y,m,_=end_targets([0,4],[],8)
    assert y.sum()==m.sum()==0


def test_shared_core_and_gradients():
    from src.phrase_end_auxiliary import DualBoundary
    from src.score_context_study import make_model
    torch.set_num_threads(2)
    base=make_model('P',42);dual=DualBoundary(42)
    for k,v in base.state_dict().items():torch.testing.assert_close(v,dual.core.state_dict()[k],rtol=0,atol=0)
    x=torch.randn(2,12,58);pad=torch.zeros(2,12,dtype=torch.bool);pad[0,9:]=True
    base.eval();dual.eval();a,b=dual(x,padding_mask=pad,both=True)
    torch.testing.assert_close(a,base(x,padding_mask=pad),rtol=0,atol=0)
    assert a.shape==b.shape==(2,12) and b[0,9:].abs().sum()==0
    (a.square().mean()+.25*b.square().mean()).backward()
    assert dual.end_head.weight.grad.abs().sum()>0 and dual.core.input_projection.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in dual.parameters() if p.grad is not None)


def test_sampler_pair_and_resume():
    from src.phrase_end_auxiliary import DualSampler
    from src.models import Normalizer
    n=10;x=np.zeros((2,n,58),np.float32)
    d={'p':dict(curves=x,labels=np.arange(n,dtype=np.float32)%2,label_mask=np.ones(n,np.float32),end_labels=np.arange(n,dtype=np.float32)%3==0,end_mask=np.ones(n,np.float32))}
    norm=Normalizer(np.zeros(58,np.float32),np.ones(58,np.float32));s=DualSampler(d,norm,42)
    state=s.state();a=s.batch();s.load_state(state);b=s.batch()
    for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=0,atol=0)
    assert len(a)==6 and a[4].shape==a[1].shape
