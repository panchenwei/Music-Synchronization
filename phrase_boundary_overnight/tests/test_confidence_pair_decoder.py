import itertools
import numpy as np
import pytest
from src.confidence_pair_decoder import decode
from src.phase2_models import nms_probabilities


def test_missing_endpoint_and_no_zero_length():
    assert decode([.9, 0], [0, 0], .5, .5)[1] == []
    assert decode([.9], [.9], .5, .5)[1] == []
    assert decode([0, .9], [.9, 0], .5, .5)[1] == []


def test_gap_and_interlocking():
    assert decode([.9, 0, .9, 0, 0], [0, 0, .9, 0, .9], .5, .5)[1] == [(0, 2), (2, 4)]
    assert decode([.9, 0, 0, .9, 0], [0, .9, 0, 0, .9], .5, .5)[1] == [(0, 1), (3, 4)]


def test_competing_starts_and_original_nms():
    p, spans, _ = decode([.7, .9, 0, .8, 0], [0, 0, 0, 0, .9], .5, .5)
    assert spans == [(1, 4)]
    assert np.flatnonzero(p).tolist() == [1]


def test_bruteforce_optimality():
    rng = np.random.default_rng(43)
    for _ in range(30):
        start, end = rng.uniform(.1, .9, (2, 5))
        sp, ep = nms_probabilities(start) >= .5, nms_probabilities(end) >= .5
        best = 0.
        for actions in itertools.product(('', 'S', 'E', 'ES'), repeat=5):
            inside = False; total = 0.; valid = True
            for t, action in enumerate(actions):
                if action == 'S' and (inside or not sp[t]): valid = False; break
                if action == 'E' and (not inside or not ep[t]): valid = False; break
                if action == 'ES' and (not inside or not sp[t] or not ep[t]): valid = False; break
                if 'S' in action: total += np.log(start[t]/(1-start[t]))+1e-8
                if 'E' in action: total += np.log(end[t]/(1-end[t]))+1e-8
                if action == 'S': inside = True
                if action == 'E': inside = False
            if valid and not inside: best = max(best, total)
        scores, spans, score = decode(start, end, .5, .5)
        assert abs(score-best) < 1e-10
        assert set(np.flatnonzero(scores)) <= set(np.flatnonzero(sp))
        assert all(s < e and ep[e] for s, e in spans)


def test_bad_input():
    with pytest.raises(ValueError): decode([np.nan], [.9], .5, .5)
    with pytest.raises(ValueError): decode([.9], [.9], 0, .5)
