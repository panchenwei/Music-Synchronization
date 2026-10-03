import numpy as np
from src.voice_pool_features import features
from src.ordered_attack_features import features as old

def test_single_voice_identical_and_source_unchanged():
    e=np.array([[0,1,64,1,1],[0,1,40,2,1],[1,.5,42,2,1]],float);before=e.copy()
    v,r=features(e,3);np.testing.assert_array_equal(v,r);np.testing.assert_array_equal(e,before)
    np.testing.assert_array_equal(r,old(e,3)[0])

def test_other_voice_low_bass_changes_ordered_interval():
    e=np.array([[0,1,64,1,1],[0,1,48,2,1],[1,.5,52,2,1],[1.5,.5,54,2,1],[2,.5,56,2,1],[1,1,45,2,2]],float)
    v,r=features(e,4)
    assert np.isclose(r[1,28],4/48) and np.isclose(v[1,28],-3/48)
    assert v.shape==r.shape==(4,58) and np.isfinite(v).all()
    np.testing.assert_array_equal(v[1,:28],r[1,:28])
