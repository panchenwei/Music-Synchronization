import numpy as np
from src.error_review_30 import matches,decode,subtype

def test_one_truth_not_shared():
    pairs,fp,fn=matches([4,5,6],[5]);assert pairs==[(5,5)] and fp==[4,6] and fn==[]

def test_missing_raw_score():
    d=decode(np.array([.1,.2,.1]),np.array([0,1,0]),np.ones(3,bool),.5)
    assert subtype(d,'FN',1)=='below_threshold'

def test_masked_peak_suppresses_valid_candidate():
    d=decode(np.array([.9,.8,.1,.1]),np.array([0,0,1,0]),np.array([0,1,1,1],bool),.5)
    assert subtype(d,'FN',2)=='nms_removed'

def test_exact_first():
    pairs,fp,fn=matches([4,6],[5,6]);assert pairs==[(6,6),(4,5)] and not fp and not fn

def test_plateau_matches_only_once():
    d=decode(np.array([.8,.8,.8]),np.array([0,1,0]),np.ones(3,bool),.5)
    assert len(d['pairs'])==1 and len(d['fp'])==2
