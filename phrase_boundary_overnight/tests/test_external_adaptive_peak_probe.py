import numpy as np
from src.external_adaptive_peak_probe import pick


def test_adaptive_peak_properties():
    for n in (1,2,8,100):
        assert not len(pick(np.ones(n))[0])
        assert not len(pick(np.zeros(n))[0])
    x=np.zeros(101);x[50]=1
    cut,smooth,threshold=pick(x)
    np.testing.assert_array_equal(cut,[50])
    np.testing.assert_array_equal(pick(7*x)[0],cut)
    np.testing.assert_array_equal(pick(x[::-1])[0],100-cut)
    assert smooth[50]>threshold[50]
    assert not len(pick(x,2)[0])
