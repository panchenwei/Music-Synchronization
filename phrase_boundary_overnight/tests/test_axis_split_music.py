import numpy as np
import torch
from src.axis_split_music import AxisBoundary, AxisStem, features_in_blocks
from src.recurrence_depth_models import make_model
from src.score_roll_branch import RollSampler
from src.phase7_models import CurvePieceBalancedSampler
from src.models import Normalizer


def test_initial_baseline_and_shared_modules():
    torch.set_num_threads(2)
    x=torch.randn(2,12,58); roll=torch.rand(2,12,2,128,4)
    a=AxisBoundary('S',42).eval(); b=AxisBoundary('J',42).eval(); old=make_model('C3',42).eval()
    torch.testing.assert_close(a(x,roll),old(x),atol=2e-6,rtol=0)
    torch.testing.assert_close(a(x,roll),b(x,roll),atol=2e-6,rtol=0)
    for part in ('input','norm','readout'):
        for k,v in getattr(a.stem,part).state_dict().items():
            torch.testing.assert_close(v,getattr(b.stem,part).state_dict()[k],atol=0,rtol=0)


def test_mask_and_chunk_equivalence():
    for mode in ('S','J'):
        m=AxisBoundary(mode,42).eval(); r=torch.rand(1,37,2,128,4)
        torch.testing.assert_close(features_in_blocks(m,r,9),m.stem(r),atol=2e-5,rtol=0)
        mask=torch.arange(37)[None]>=30; dirty=r.clone(); dirty[:,30:]=999
        torch.testing.assert_close(m.stem(r,mask),m.stem(dirty,mask),atol=0,rtol=0)
        torch.testing.assert_close(m.stem(r[:,:30]),m.stem(r,mask)[:,:30],atol=2e-5,rtol=0)


def test_stem_learns_and_sees_time_order():
    for mode in ('S','J'):
        m=AxisBoundary(mode,42); opt=torch.optim.AdamW(m.parameters(),lr=.001)
        x=torch.randn(2,10,58); r=torch.zeros(2,10,2,128,4)
        for b in range(10): r[:,b,:,55+b,b%4]=1
        initial=m.stem.input.weight.detach().clone()
        for _ in range(3):
            opt.zero_grad(); loss=(m(x,r)-1).square().mean(); loss.backward(); opt.step()
        assert not torch.equal(initial,m.stem.input.weight)
        m.eval(); assert (m.stem(r)-m.stem(r.flip(1)).flip(1)).abs().max()>1e-4


def test_sampler_keeps_original_rng_and_supervision():
    rng=np.random.default_rng(0); d={}
    for pid,n in (('a',31),('b',91)):
        d[pid]=dict(curves=rng.normal(size=(2,n,58)).astype('float32'),labels=np.zeros(n),label_mask=np.ones(n),piano_roll=np.zeros((n,2,128,4),np.float32))
    norm=Normalizer(np.zeros(58),np.ones(58)); a=RollSampler(d,norm,64,5,42); b=CurvePieceBalancedSampler(d,norm,64,5,42)
    for _ in range(3):
        for u,v in zip(a.batch()[:4],b.batch()): torch.testing.assert_close(u,v,atol=0,rtol=0)
        assert a.state()==b.state()
