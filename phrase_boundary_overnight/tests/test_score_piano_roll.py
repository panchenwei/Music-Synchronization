import numpy as np
import pytest
from src.score_piano_roll import piano_roll


def test_exact_beat_anchor_and_half_open_duration():
    x = piano_roll([(1., 1., 60, 1, 1)], 3)
    assert x.shape == (3, 2, 128, 4)
    assert not x[0].any() and not x[2].any()
    np.testing.assert_array_equal(x[1,0,60], [1,1,1,1])
    np.testing.assert_array_equal(x[1,1,60], [1,0,0,0])


def test_fractional_union_and_repeated_onset_not_fake_energy():
    x = piano_roll([(.10,.15,60),(.20,.15,60),(.4,.05,60)],1)
    np.testing.assert_allclose(x[0,0,60], [.6,.6,0,0], atol=1e-6)
    np.testing.assert_array_equal(x[0,1,60], [1,1,0,0])
    assert abs(x[:,0].sum()/4-.30) < 1e-6
    duplicate = piano_roll([(.10,.15,60),(.20,.15,60),(.4,.05,60)]*2,1)
    np.testing.assert_array_equal(x,duplicate)


def test_clipping_is_not_onset_shifting_and_grace_is_excluded():
    x = piano_roll([(-.5,1.,60),(1.9,1.,62),(2.,1.,64),(.3,0.,67)],2)
    assert x[0,0,60].sum() == 2 and not x[:,1,60].any()
    assert abs(x[1,0,62].sum()-.4) < 1e-6 and x[1,1,62,3] == 1
    assert not x[:,:,64].any() and not x[:,:,67].any()


def test_pitch_shift_and_within_beat_order_are_preserved():
    events = [(0,.2,60),(.5,.3,64)]
    x = piano_roll(events,1)
    shifted = piano_roll([(o,d,p+3) for o,d,p in events],1)
    np.testing.assert_array_equal(np.roll(x,3,axis=2),shifted)
    reverse = piano_roll([(o,d,64 if p==60 else 60) for o,d,p in events],1)
    assert not np.array_equal(x,reverse)
    np.testing.assert_array_equal(x[:,1].sum(axis=2),reverse[:,1].sum(axis=2))


def test_invalid_inputs_fail_and_empty_music_is_zero():
    assert not piano_roll([],2).any()
    for event in [(0,1,128),(float('nan'),1,60),(0,1,60.5)]:
        with pytest.raises(ValueError):
            piano_roll([event],1)
