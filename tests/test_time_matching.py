"""Run directly: python tests/test_time_matching.py (no test framework needed)."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from valid import (
    _select_time_indices,
    _select_predicted_rows_by_time_dp,
    _select_gt_indices_by_time_dp,
)


def main():
    rng = np.random.default_rng(42)
    for m in range(1, 8):
        for n in range(m + 1):
            candidates = np.sort(rng.uniform(0, 10, m))
            targets = np.sort(rng.uniform(0, 10, n))
            indices = _select_time_indices(candidates, targets)
            optimum = min(
                np.sum((candidates[list(c)] - targets) ** 2)
                for c in combinations(range(m), n)
            )
            assert len(indices) == n
            assert np.all(np.diff(indices) > 0)
            assert np.isclose(np.sum((candidates[indices] - targets) ** 2), optimum)
            frame = pd.DataFrame({'predicted_performance_time_sec': candidates})
            selected, rule = _select_predicted_rows_by_time_dp(frame, targets)
            np.testing.assert_array_equal(selected.iloc[:, 0], candidates[indices])
            np.testing.assert_array_equal(_select_gt_indices_by_time_dp(targets, candidates), indices)
            assert rule == f'time_dp_drop_{m - n}'
    # Existing tie behavior prefers the later candidate.
    np.testing.assert_array_equal(_select_time_indices(np.array([1., 1.]), np.array([1.])), [1])
    try:
        _select_time_indices(np.array([1.]), np.array([1., 2.]))
    except ValueError:
        pass
    else:
        raise AssertionError('Impossible correspondence must fail')
    print('Time matching: exhaustive optimum, wrappers, ties and invalid size PASS')


if __name__ == '__main__':
    main()
