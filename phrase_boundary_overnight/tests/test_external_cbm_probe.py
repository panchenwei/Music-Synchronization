import itertools
import numpy as np
from src.external_cbm_probe import segment,band_scores


def test_scores_equal_explicit_kernel():
    x=np.random.default_rng(4).normal(size=(12,5));u=x/np.linalg.norm(x,axis=1,keepdims=True);s=u@u.T;scores=band_scores(x)
    for end in range(1,13):
        for length in range(1,end+1):
            idx=np.arange(length);k=((abs(idx[:,None]-idx)>0)&(abs(idx[:,None]-idx)<=7));expected=(s[end-length:end,end-length:end]*k).sum()/length
            assert abs(scores[end,length]-expected)<1e-12


def test_dynamic_program_equals_exhaustive():
    x=np.random.default_rng(7).normal(size=(9,4));scores=band_scores(x,7,5);cuts,value=segment(x,7,5);best=-np.inf
    for bits in itertools.product((False,True),repeat=8):
        path=np.r_[0,np.flatnonzero(bits)+1,9]
        if np.diff(path).max()>5:continue
        best=max(best,sum(scores[b,b-a] for a,b in zip(path[:-1],path[1:])))
    assert abs(value-best)<1e-12 and ((cuts>0)&(cuts<9)).all()


def test_clean_block_boundary():
    x=np.zeros((16,2));x[:8,0]=1;x[8:,1]=1;cuts,_=segment(x);np.testing.assert_array_equal(cuts,[8])
