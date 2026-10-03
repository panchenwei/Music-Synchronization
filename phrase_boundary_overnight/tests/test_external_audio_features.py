import numpy as np
from src.external_audio_features import score_features,aggregate_frames


def test_score_profile_keeps_pitch_and_rest():
    x=score_features(np.array([[0.,2.,60.],[0.,1.,48.],[3.,4.,67.]]),5)
    assert x.shape==(5,28) and np.isfinite(x).all()
    assert x[0,0]>0 and x[0,12]>0 and x[3,7]>0
    assert x[2].sum()==0 and x[4].sum()==0


def test_frame_time_mapping_and_missing():
    times=np.arange(0,6,.1);features=np.stack([times,np.ones_like(times)])
    out,known=aggregate_frames(times,features,np.array([0.,1.,2.,3.,np.nan,5.]))
    np.testing.assert_array_equal(known,[0,1,1,0,0,0]);assert abs(out[1,0]-.95)<1e-5 and out[1,1]==1
    assert np.isfinite(out).all()
