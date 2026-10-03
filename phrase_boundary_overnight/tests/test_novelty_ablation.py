import numpy as np
import pytest
from src.score_novelty_ablation import select_features


def test_novelty_values_and_availability_separate_without_mutation():
    x = np.arange(120, dtype=np.float32).reshape(5, 24)
    old = x.copy()
    values = select_features(x, 'V')
    flags = select_features(x, 'M')
    np.testing.assert_array_equal(x, old)
    np.testing.assert_array_equal(values[:, :12], x[:, :12])
    np.testing.assert_array_equal(flags[:, 12:16], x[:, 12:16])
    assert not values[:, 12:].any() and not flags[:, :12].any() and not flags[:, 16:].any()
    with pytest.raises(ValueError):
        select_features(x, 'UNKNOWN')
