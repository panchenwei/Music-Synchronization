import numpy as np
from src.score_novelty_features import change_curve, unit_rows, onset_rhythm, score_novelty


def test_constant_silence_and_edges():
    for x in (np.ones((30, 3)), np.zeros((30, 3))):
        v, a = change_curve(x, 3)
        np.testing.assert_allclose(v, 0, atol=1e-12)
        np.testing.assert_array_equal(np.flatnonzero(a), np.arange(3, 28))
    v, a = change_curve(np.ones((2, 4)), 3)
    assert not v.any() and not a.any()


def test_boundary_before_beat_and_ssm_equivalence():
    x = np.zeros((30, 2))
    x[:15, 0] = 1
    x[15:, 1] = 1
    v, _ = change_curve(x, 3)
    assert np.argmax(v) == 15 and np.isclose(v[15], .5)
    rng = np.random.default_rng(42)
    z = unit_rows(rng.normal(size=(40, 6)))
    scale = 6
    weights = np.exp(-.5*((np.arange(scale)+.5)/(scale/2.))**2)
    weights /= weights.sum()
    signed = np.r_[-weights[::-1], weights]
    kernel = np.outer(signed, signed) / 4
    a, _ = change_curve(z, scale)
    for b in range(scale, len(z)-scale+1):
        window = z[b-scale:b+scale]
        np.testing.assert_allclose(a[b], ((window @ window.T)*kernel).sum(), atol=1e-7)


def test_rhythm_and_pitch_permutation_invariance():
    events = [(2., .25, 60, 1, 1), (2.5, 2., 64, 1, 1), (3., 0., 65, 1, 1)]
    x = onset_rhythm(events, 60)
    assert x[2, 0] == .5 and x[2, 3] == .5 and np.isclose(x[2, 6], np.log(3))
    rng = np.random.default_rng(4)
    p = rng.random((60, 24))
    shifted = np.c_[np.roll(p[:, :12], 5, axis=1), np.roll(p[:, 12:], 5, axis=1)]
    a = score_novelty(p, events)
    b = score_novelty(shifted, list(reversed(events)))
    np.testing.assert_allclose(a, b, atol=1e-7)
    assert a.shape == (60, 24) and not a[:, 16:].any() and np.isfinite(a).all()
