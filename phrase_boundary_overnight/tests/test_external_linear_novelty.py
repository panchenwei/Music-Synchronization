import numpy as np
from src.external_linear_novelty import training_arrays,arrays


def test_weights_ignore_unknown_and_balance_works():
    data={'a':dict(piece_id='one',labels=np.array([0.,1.,1.]),label_mask=np.array([1.,1.,0.])),
          'b':dict(piece_id='one',labels=np.array([0.,1.]),label_mask=np.ones(2)),
          'c':dict(piece_id='two',labels=np.array([0.,1.]),label_mask=np.ones(2))}
    inputs={k:np.ones((len(v['labels']),30)) for k,v in data.items()}
    x,y,w,bw,ids,pos=training_arrays(data,inputs)
    assert len(y)==6 and sum(y)==3 and x.shape==(6,30)
    for pid in ('one','two'):assert np.isclose(sum(a for a,k in zip(bw,ids) if k==pid),1.)
    assert np.isclose(w.mean(),1.) and np.isclose(pos,1.)


def test_only_novelty_is_disabled():
    class Identity:
        def apply(self,x):return x.copy()
    features=np.arange(242,dtype=float).reshape(2,121);data={'x':{'features':features}}
    s=arrays(data,Identity(),'S')['x'];e=arrays(data,Identity(),'E')['x']
    assert s.shape==(2,30) and (s[:,28]==0).all()
    np.testing.assert_array_equal(s[:,:28],e[:,:28]);np.testing.assert_array_equal(s[:,-1],features[:,-1])
