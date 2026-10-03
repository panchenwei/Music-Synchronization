import numpy as np
import pytest
from src.external_audio_masks import build_masks,continuous_runs,snap_coordinate


def test_gaps_prefix_tail_and_adjacent_bar_merging():
    y,m,rid,covered,rows,runs=build_masks([(0,4),(4,8),(10,16)],[0,4,6,10,14],18)
    assert runs==[(0,8),(10,16)]
    np.testing.assert_array_equal(np.flatnonzero(y),[4,6,14])
    assert not m[[0,7,8,9,10,15,16,17]].any()
    assert m[1:7].all() and m[11:15].all()
    assert rid[9]==-1 and covered[10] and (y<=m).all()
    with pytest.raises(ValueError):continuous_runs([(0,4),(3,6)])


def test_ambiguous_half_positions_and_repeat_occurrences():
    y,m,*_=build_masks([(0,16)],[0,4.50000000001,8,12,14.5],18)
    assert not m[[4,5,14,15]].any()
    np.testing.assert_array_equal(np.flatnonzero(y),[8,12])
    assert snap_coordinate(4.50000000001)==4.5 and snap_coordinate(4.51)==4.51
    y2,m2,*_=build_masks([(0,16)],[0,4,8,12],18)
    np.testing.assert_array_equal(np.flatnonzero(y2),[4,8,12])
    # A separate run with only one reference has no known negatives.
    y3,m3,*_=build_masks([(0,4),(8,12)],[1,9],14)
    assert not y3.any() and not m3.any()
