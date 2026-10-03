import numpy as np
from src.predictive_information_features import Predictor,streams,fit_predictors,features


def test_predictor_context_and_no_online_fit():
    model=Predictor(3).fit([[0,1,0,1]*30,[2,2]*10]);before={k:v.copy() for k,v in model.counts.items()}
    assert model.distribution([0])[1]>model.distribution([0],True)[1]
    p=model.distribution([99,99,99]);assert np.isfinite(p).all() and abs(p.sum()-1)<1e-10
    for k,v in before.items():np.testing.assert_array_equal(v,model.counts[k])


def test_transposition_and_order_probe():
    notes=np.array([(i,1,p,1,1) for i,p in enumerate([60,62,64,62,60,67,60,62])],float)
    v=streams(notes);other=notes.copy();other[:,2]+=7;assert v==streams(other)
    predictors=fit_predictors([v]);a,_=features(v,predictors,9);b,_=features(streams(other),predictors,9)
    np.testing.assert_array_equal(a,b);assert a.shape==(9,14) and np.isfinite(a).all()
    reverse=notes.copy();reverse[:,2]=reverse[::-1,2];c,_=features(streams(reverse),predictors,9)
    assert not np.allclose(a,c)


def test_real_merged_event_schema():
    from src.note_relation_graph import merged_events
    notes=[(0.,1.,60,1,1),(1.,1.,60,1,1),(2.,1.,62,1,1),(3.,1.,64,1,1)]
    merged,count=merged_events(notes,[1,-1,None,None])
    assert count==1 and len(merged[0])==6
    assert streams(merged)==streams(np.asarray(merged)[:,:5])
    assert streams([])==[[],[]]
