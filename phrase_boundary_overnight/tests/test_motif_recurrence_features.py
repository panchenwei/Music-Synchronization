import numpy as np
from src.motif_recurrence_features import features,staff_sequences,stream_features,window_similarity


def test_order_vs_bag_with_same_contents():
    x=np.eye(3)[[0,1,2,2,1,0]]
    ordered=window_similarity(x,3,True);bag=window_similarity(x,3,False)
    assert abs(bag[0,3]-1)<1e-12 and abs(ordered[0,3]-1/3)<1e-12


def test_exact_repetition_and_start_contrast():
    x=np.eye(4)[[3,3,0,1,2,2,3,3,0,1,2,2,3,3]]
    a=stream_features(x,True,scales=(2,))
    assert a[2,0]==1 and a[2,3]==1
    assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()


def test_no_self_or_overlapping_span_match():
    a=stream_features(np.eye(6),True,scales=(2,))
    assert np.array_equal(a,np.zeros_like(a))  # Complete spans cannot be disjoint.


def test_global_transposition_preserves_recurrence():
    e=[(float(i),1.,60+(i%5),1.,1.,0.) for i in range(60)]+[(float(i),2.,40+(i%3),2.,1.,0.) for i in range(0,60,2)]
    shifted=[(q,d,p+7,s,v,t) for q,d,p,s,v,t in e]
    for ordered in (True,False):np.testing.assert_allclose(features(e,60,ordered),features(shifted,60,ordered),rtol=0,atol=1e-6)


def test_tied_sustain_has_no_extra_onset_and_subbeat_kept():
    x=staff_sequences([(0.,2.,60,1,1,0.),(.25,.5,64,2,1,0.)],3).reshape(2,3,4,24)
    assert x[0,0,0,0]>0 and x[0,1,:12].size>0
    assert x[0,1,:,:12].sum()==0 and x[0,1,:,12].sum()>0
    assert x[1,0,1,4]>0 and x[1,0,0,4]==0
