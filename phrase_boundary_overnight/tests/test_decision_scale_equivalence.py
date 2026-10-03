import numpy as np
from src.interstart_decoder import decode

def test_temperature_equals_rescaled_gap_potential():
    rng=np.random.default_rng(20260913)
    hist=np.exp(-.5*((np.arange(1,257)-18)/3)**2);hist/=hist.sum()
    prior={'mean':18.,'histogram':hist.tolist()}
    def sigmoid(x):return 1/(1+np.exp(-x))
    def logit(x):return np.log(x/(1-x))
    for temperature in (1.5,2.):
        for _ in range(20):
            p=rng.uniform(.05,.95,128);t=.6
            left=decode(sigmoid(logit(p)/temperature),float(sigmoid(logit(t)/temperature)),prior,.5)
            right=decode(p,t,prior,.5*temperature)
            np.testing.assert_array_equal(left,right)
