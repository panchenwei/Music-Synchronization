import numpy as np
import torch
from src.relation_interval_features import reference_voices,relation_features
from src.score_context_study import make_model

def example():return [(0,1,60,1,1),(0,.5,48,2,1),(.5,.5,50,2,1)]

def test_vertical_duration_and_horizontal_hand_calculation():
    events=example();m,b=reference_voices(events)
    x=relation_features(events,1,m,True);z=relation_features(events,1,b,False)
    assert m==(1,1) and b==(2,1)
    np.testing.assert_allclose(x[0,[0,2]],[.75,.25]);assert x[0,12:].sum()==0
    np.testing.assert_allclose(z[0,[0,10]],[.75,.25]);assert z[0,14]==1

def test_transposition_and_order_invariance():
    events=example()+[(1,.25,62,1,1),(1.25,.25,59,1,1),(1.5,.5,64,1,1)]
    for voice,high in (((1,1),True),((2,1),False)):
        x=relation_features(events,3,voice,high)
        np.testing.assert_array_equal(x,relation_features(list(reversed(events)),3,voice,high))
        for shift in (-7,-1,1,7,12):
            moved=[(o,d,p+shift,s,v) for o,d,p,s,v in events]
            np.testing.assert_array_equal(x,relation_features(moved,3,voice,high))

def test_silence_grace_and_missing_reference():
    x=relation_features([(0,0,70,1,1),(0,1,48,2,1)],3,(1,1),True)
    assert x.sum()==0
    z=relation_features([(0,.5,60,1,1),(1.5,.5,62,1,1)],3,(1,1),True)
    assert z[2].sum()==0 and z[1,14]==1

def test_multiple_attacks_in_one_beat_are_not_lost():
    e=[(0,.2,60,1,1),(.25,.2,62,1,1),(.5,.2,61,1,1),(.75,.2,60,1,1)]
    x=relation_features(e,1,(1,1),True)
    np.testing.assert_allclose(x[0,[14,23]],[1/3,2/3])

def test_zero_initial_added_block_and_parameters():
    m=make_model('P',42).eval();assert sum(p.numel() for p in m.parameters())==3297
    x=torch.randn(2,20,58);z=x.clone();z[:,:,34:]=0
    with torch.no_grad():torch.testing.assert_close(m(x),m(z),atol=1e-6,rtol=1e-5)
