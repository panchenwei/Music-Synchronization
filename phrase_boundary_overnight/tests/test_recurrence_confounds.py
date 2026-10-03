from itertools import permutations
import numpy as np
from src.recurrence_confounds import correspondence_average,control_features
from src.motif_recurrence_features import features,normalize


def test_exact_expectation_of_uniform_permuted_correspondence():
    x=normalize(np.array([[1,0,0],[0,1,0],[1,1,0],[0,1,1],[0,0,1],[1,0,1]],float))
    expected=np.mean([np.mean(np.sum(x[:3]*x[3:][list(p)],axis=1)) for p in permutations(range(3))])
    assert abs(correspondence_average(x,3)[0,3]-expected)<1e-12


def test_no_extra_normalization_of_repeated_window():
    x=np.eye(3)[[0,1,2,0,1,2]]
    assert abs(correspondence_average(x,3)[0,3]-1/3)<1e-12


def test_identical_availability_and_global_transposition():
    e=[(float(i),1.,60+i%5,1.,1.,0.) for i in range(60)]
    c=control_features(e,60);o=features(e,60,True)
    np.testing.assert_array_equal(c[:,3::4],o[:,3::4])
    shifted=[(q,d,p+3,s,v,t) for q,d,p,s,v,t in e]
    np.testing.assert_allclose(c,control_features(shifted,60),rtol=0,atol=1e-6)
    assert c.shape==(60,24) and np.isfinite(c).all() and ((c>=0)&(c<=1)).all()
