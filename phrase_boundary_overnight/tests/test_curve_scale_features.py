import numpy as np
from src.curve_scale_features import contrast,transform,SCALES
from src.phase6_multiscale import segment_novelty


def test_direction_and_existing_absolute_contrast():
    y=np.r_[np.zeros(32),np.ones(32)];ok=np.ones(64,bool)
    for scale in SCALES:
        a,q=contrast(y,ok,scale);b,_=contrast(-y,ok,scale)
        assert a[32]==1 and q[32]==1
        np.testing.assert_allclose(a,-b,atol=1e-7)
        old,_=segment_novelty(y,ok,scale)
        np.testing.assert_allclose(abs(a[scale:65-scale]),old[scale:65-scale])
        assert not np.any(q[:scale]) and not np.any(q[65-scale:])


def test_missing_data_no_artificial_edges_or_last_tempo():
    a=np.ones(64);ok=np.ones(64,bool);ok[20:26]=False;a[20:26]=np.nan
    out,q=contrast(a,ok,4);assert np.max(abs(out))==0
    assert not q[24]
    curves=np.zeros((1,64,9),np.float32);curves[:,:,0]=5;curves[:,:,3]=2;curves[:,:,7:9]=1
    baseline=transform(curves);curves[0,-1,0]=999
    np.testing.assert_array_equal(transform(curves),baseline)
    assert np.max(abs(baseline[:,:,:2]))==0
    assert baseline.shape==(1,64,4,5)


def test_impulse_locality_and_scale_order():
    y=np.zeros(100);y[50]=1
    for s in SCALES:
        out,_=contrast(y,np.ones(100,bool),s)
        assert np.all(out[:51-s]==0) and np.all(out[51+s:]==0)
        assert out[50]>0 and out[51]<0
