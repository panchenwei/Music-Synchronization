import numpy as np
import torch
from src.performance_pair_sampler import PairSampler,consistency_forward
from src.phase7_models import CurvePieceBalancedSampler
from src.models import Normalizer
from src.recurrence_depth_models import make_model

def test_primary_batch_and_resume_remain_identical():
    rng=np.random.default_rng(17);data={}
    for pid,n in [('a',90),('b',35)]:
        x=rng.normal(size=(3,n,58)).astype(np.float32)
        data[pid]=dict(curves=x,labels=np.zeros(n,np.float32),label_mask=np.ones(n,np.float32))
    norm=Normalizer(np.zeros(58),np.ones(58));a=PairSampler(data,norm,64,4,42);b=CurvePieceBalancedSampler(data,norm,64,4,42)
    for _ in range(3):
        for u,v in zip(a.batch(),b.batch()):torch.testing.assert_close(u,v,atol=0,rtol=0)
        for i,(pid,p,q,s) in enumerate(a.pair_keys):
            assert p!=q
            raw=data[pid]['curves'][q,s:s+64]
            np.testing.assert_array_equal(a.pair_batch[i,:len(raw)],raw)
    state=a.state();old=a.batch();pair=a.pair_batch.clone();a.load_state(state)
    for u,v in zip(old,a.batch()):torch.testing.assert_close(u,v,atol=0,rtol=0)
    torch.testing.assert_close(pair,a.pair_batch,atol=0,rtol=0)

def test_zero_weight_preserves_gradient_rng_and_nonzero_penalty_has_signal():
    torch.set_num_threads(2);a=make_model('C3',42).train();b=make_model('C3',42).train()
    x=torch.randn(2,64,58);paired=torch.randn_like(x);valid=torch.ones(2,64);mask=valid.clone();mask[:,50:]=0
    before=torch.get_rng_state();one=a(x);one.square().mean().backward();after=torch.get_rng_state()
    torch.set_rng_state(before);two=b(x);penalty=consistency_forward(b,two,paired,valid,mask)
    assert penalty>0 and torch.isfinite(penalty)
    (two.square().mean()+0*penalty).backward()
    torch.testing.assert_close(torch.get_rng_state(),after,atol=0,rtol=0)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
