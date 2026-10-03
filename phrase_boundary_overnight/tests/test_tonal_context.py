import numpy as np
import pytest
from src.tonal_context_features import MAJOR, MINOR, duration_histogram, key_correlations, tonic_weights, relative_profiles


def test_known_keys_and_official_weights():
    from music21.analysis.discrete import KrumhanslSchmuckler
    m=KrumhanslSchmuckler()
    np.testing.assert_allclose(MAJOR,m.getWeights('major'))
    np.testing.assert_allclose(MINOR,m.getWeights('minor'))
    for mode,profile in enumerate((MAJOR,MINOR)):
        for t in range(12):
            assert np.argmax(key_correlations(np.roll(profile,t))) == mode*12+t
            assert np.argmax(tonic_weights(np.roll(profile,t),'K')) == t


@pytest.mark.parametrize('kind',['K','S'])
def test_joint_transposition_mass_and_silence(kind):
    r=np.random.default_rng(5);x=r.random((17,24));h=r.random(12)
    a=relative_profiles(x,h,kind)
    for t in range(12):
        z=np.concatenate([np.roll(x[:,:12],t,axis=1),np.roll(x[:,12:],t,axis=1)],axis=1)
        np.testing.assert_allclose(relative_profiles(z,np.roll(h,t),kind),a,atol=2e-7)
    np.testing.assert_allclose(a[:,:12].sum(1),x[:,:12].sum(1),atol=1e-6)
    np.testing.assert_allclose(a[:,12:].sum(1),x[:,12:].sum(1),atol=1e-6)
    assert not relative_profiles(np.zeros((5,24)),np.zeros(12),kind).any()
    np.testing.assert_allclose(tonic_weights(np.ones(12),kind),np.ones(12)/12)


def test_duration_clipping_and_invalid():
    h=duration_histogram([(-1,2,60),(2,10,61),(0,0,62),(10,1,63)],4)
    np.testing.assert_array_equal(h[:4],[1,2,0,0])
    with pytest.raises(ValueError): tonic_weights(np.ones(12),'bad')
    with pytest.raises(ValueError): key_correlations(np.full(12,np.nan))
