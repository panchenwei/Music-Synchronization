import numpy as np
from src.external_novelty_trial import replace_features
from src.external_mert_trial import mask_modalities


def test_one_scalar_preserves_score_and_coverage():
    x=np.random.default_rng(42).normal(size=(64,121)).astype(np.float32);v=np.linspace(0,1,64);out=replace_features(x,v)
    np.testing.assert_array_equal(out[:,:28],x[:,:28]);np.testing.assert_array_equal(out[:,120],x[:,120]);np.testing.assert_allclose(out[:,28],v)
    assert np.count_nonzero(out[:,29:120])==0
    np.testing.assert_array_equal(mask_modalities(out,'S'),mask_modalities(x,'S'))
