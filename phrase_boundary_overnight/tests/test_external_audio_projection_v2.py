import numpy as np
from src.external_audio_projection_v2 import sustain_with_one_tick


def test_one_tick_release_is_distinct_from_two_ticks_or_new_attack():
    events=[(0.,60,1.)]
    assert sustain_with_one_tick(events,np.array([[0.,3.-1/480,60]]),2.,480)
    assert not sustain_with_one_tick(events,np.array([[0.,3.-2/480,60]]),2.,480)
    assert not sustain_with_one_tick(events,np.array([[2.,3.,60]]),2.,480)
    assert not sustain_with_one_tick(events,np.array([[0.,3.,61]]),2.,480)
