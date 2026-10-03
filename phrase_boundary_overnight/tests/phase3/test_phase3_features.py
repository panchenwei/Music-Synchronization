from __future__ import annotations

import numpy as np

from src.phase3_features import SCORE_CUE_NAMES, _relative_change, _safe_cosine_novelty


def test_phase3_feature_schema_is_compact_and_unique():
    assert len(SCORE_CUE_NAMES) == 16
    assert len(set(SCORE_CUE_NAMES)) == 16
    assert "piece_id" not in SCORE_CUE_NAMES
    assert "phraseend" not in SCORE_CUE_NAMES


def test_backward_change_has_no_last_to_first_wraparound():
    values = np.asarray([1.0, 3.0, 2.0])
    delta = np.diff(values, prepend=values[0])
    assert delta.tolist() == [0.0, 2.0, -1.0]
    assert delta[-1] == values[-1] - values[-2]


def test_cosine_novelty_and_missing_semantics():
    zero = np.zeros(12)
    value, valid = _safe_cosine_novelty(zero, zero)
    assert value == 0.0 and not valid
    a = np.eye(12)[0]
    b = np.eye(12)[1]
    value, valid = _safe_cosine_novelty(a, b)
    assert np.isclose(value, 1.0) and valid


def test_relative_change_is_symmetric_and_bounded_for_nonnegative_inputs():
    assert np.isclose(_relative_change(2.0, 6.0), _relative_change(6.0, 2.0))
    assert np.isclose(_relative_change(2.0, 2.0), 0.0)
    assert 0.0 <= _relative_change(0.0, 8.0) <= 1.0
