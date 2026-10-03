"""Pool notation voices within each staff, without claiming melody identification."""
import numpy as np
from .ordered_attack_features import features as fixed_features

def features(events,n):
    original=np.asarray(events).copy()
    pooled=original.copy()
    if len(pooled):pooled[:,4]=1
    v,_=fixed_features(pooled,n)
    r,_=fixed_features(original,n)
    return v,r
