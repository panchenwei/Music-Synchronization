import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from src.tabular_context_study import context, train_arrays
from src.models import Normalizer


def test_context_offsets_and_edges():
    x = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    np.testing.assert_array_equal(context(x, 0), x)
    z = context(x, 2)
    np.testing.assert_array_equal(z[0], [0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(z[2], [1, 2, 3, 4, 5, 6, 0, 0, 0, 0])
    np.testing.assert_array_equal(z[:, 4:6], x)


def test_weighting_and_label_mask():
    norm = Normalizer(np.zeros(2), np.ones(2))
    def item(n):
        return dict(curves=np.ones((n, 3, 2)), labels=np.array([1, 0, 1]), label_mask=np.array([1, 1, 0]))
    x, y, w, counts, ratio = train_arrays({'a': item(1), 'b': item(3)}, norm, 2)
    assert x.shape == (8, 10) and len(y) == 8 and ratio == 1
    assert np.isclose(w[:2].sum(), w[2:].sum())
    assert np.isclose(w.mean(), 1.) and all(r['base_weight_sum'] == 1 for r in counts)


def test_no_random_validation_and_resume(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] > 0).astype(int)
    def model(n, warm):
        return HistGradientBoostingClassifier(max_iter=n, max_leaf_nodes=7, early_stopping=False, warm_start=warm, random_state=42)
    with threadpool_limits(limits=2):
        a = model(3, True).fit(x, y)
        joblib.dump(a, tmp_path / 'model.joblib')
        a = joblib.load(tmp_path / 'model.joblib')
        a.set_params(max_iter=6)
        a.fit(x, y)
        b = model(6, False).fit(x, y)
        assert not a.do_early_stopping_ and a.n_iter_ == 6
        np.testing.assert_allclose(a.predict_proba(x), b.predict_proba(x), atol=1e-12)
