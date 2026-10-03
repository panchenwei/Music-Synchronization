import torch
from src.ensemble_seed_replication import build_model
from src.score_context_study import make_model


def test_new_seed_common_core_initialization_and_frozen_stem():
    torch.set_num_threads(2)
    for seed in (44,45):
        b=build_model('B',seed).eval();n=build_model('N',seed).eval();r=build_model('R',seed).eval()
        old=make_model('P',seed).eval()
        for m in (b,n,r.core):
            for k,v in old.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
        assert not any(p.requires_grad for p in r.stem.parameters())
        x=torch.randn(1,4,58);x[...,34:]=0;roll=torch.rand(1,4,2,128,4)
        with torch.no_grad():
            torch.testing.assert_close(b(x),r(x,roll),rtol=0,atol=0)
            torch.testing.assert_close(b(x),n(x),rtol=0,atol=0)
