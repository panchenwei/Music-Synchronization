import numpy as np
from src.motif_recurrence_features import normalize, stream_features
from src.transposed_recurrence import shift_chroma, window_cross_similarity, transposed_stream


def tokens(pitches):
    x = np.zeros((len(pitches), 4, 2, 12))
    for i, p in enumerate(pitches): x[i, :, :, p%12] = 1
    return normalize(x.reshape(len(pitches), 96))


def test_zero_shift_is_frozen_ordered_control():
    x = normalize(np.random.default_rng(11).random((40, 96)))
    np.testing.assert_array_equal(transposed_stream(x, (0,)), stream_features(x, True))


def test_constant_transposition_not_arbitrary_note_mapping():
    a = tokens([0, 2, 4]); good = tokens([5, 7, 9]); bad = tokens([5, 8, 10])
    def best(y):
        return max(np.mean(np.sum(a*shift_chroma(y, s), axis=1)) for s in range(12))
    assert np.isclose(best(good), 1.)
    assert best(bad) < .7
    # Picking a different shift for every beat would falsely make even bad equal one.
    assert np.allclose(np.max([np.sum(a*shift_chroma(bad, s), axis=1) for s in range(12)], axis=0), 1)


def test_transposition_invariance_availability_and_monotonicity():
    x = tokens(np.random.default_rng(7).integers(0, 12, 40))
    t = transposed_stream(x); o = stream_features(x, True)
    np.testing.assert_allclose(t, transposed_stream(shift_chroma(x, 5)), atol=1e-7, rtol=0)
    np.testing.assert_array_equal(t[:, 3::4], o[:, 3::4])
    assert np.isfinite(t).all() and (t >= o-1e-7).all() and (t <= 1).all()


def test_window_averages_before_maximizing_shift():
    x = tokens([0, 2, 4, 5, 8, 10])
    sims = [window_cross_similarity(x @ shift_chroma(x, s).T, 3)[0, 3] for s in range(12)]
    assert max(sims) < .7
