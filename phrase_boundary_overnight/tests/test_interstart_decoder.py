import itertools
import numpy as np
from src.interstart_decoder import fit_prior,log_ratio,decode
from src.phase2_models import nms_probabilities


def example():
    y=np.zeros(100);y[[2,14,26,50,74]]=1
    return {'a':dict(labels=y,label_mask=np.ones(100))}


def test_prior_work_balance_tail_and_no_hard_min():
    a=example();p=fit_prior(a);d=np.arange(1,600)
    assert np.isfinite(log_ratio(p,d)).all()
    # Performance multiplicity is not used; each work's interval distribution has unit weight.
    a['a']['curves']=np.zeros((100,100,9));assert fit_prior(a)==p
    assert log_ratio(p,np.array([1,599])).min()>=np.log(.5)-1e-10


def test_zero_strength_exact_old_decoder():
    rng=np.random.default_rng(42);prior=fit_prior(example())
    for raw in (rng.random(64),np.ones(7)*.5,np.zeros(7)):
        for threshold in (.1,.5,.8):np.testing.assert_array_equal(decode(raw,threshold,prior,0),np.flatnonzero(nms_probabilities(raw)>=threshold))


def test_dp_matches_exhaustive_subsets():
    prior=fit_prior(example());raw=np.array([.65,.1,.62,.2,.75,.1,.49,.1,.56])
    indices=np.flatnonzero(nms_probabilities(raw)>0);threshold=.55;strength=.5
    def score(path):
        if not len(path):return 0.
        p=raw[path];return float((np.log(p/(1-p))-np.log(threshold/(1-threshold))).sum()+strength*log_ratio(prior,np.diff(path)).sum())
    paths=[np.array([v for v,keep in zip(indices,flags) if keep],int) for flags in itertools.product((0,1),repeat=len(indices))]
    path=decode(raw,threshold,prior,strength)
    assert np.isclose(score(path),max(score(x) for x in paths))
