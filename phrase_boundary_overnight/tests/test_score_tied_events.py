import numpy as np
from src.score_tied_events import attack_flags,tied_piano_roll
from src.score_piano_roll import piano_roll


def test_connected_chain_preserves_sounding_occupancy():
    e=[(0,1,60,1,1),(1,1,60,1,1),(2,1,60,1,1)]
    a,c=attack_flags(e,[1,0,-1]);np.testing.assert_array_equal(a,[1,0,0]);assert c==dict(connected=2,orphan=0,ambiguous=0)
    r,_=tied_piano_roll(e,[1,0,-1],3);np.testing.assert_array_equal(r[:,0],piano_roll(e,3)[:,0]);assert r[:,1].sum()==1


def test_orphan_and_other_voice_retained():
    e=[(0,1,60,1,1),(1,1,60,1,2),(2,1,62,1,1)]
    a,c=attack_flags(e,[1,-1,-1]);assert a.all() and c['orphan']==2


def test_ambiguous_and_repeated_attacks_not_merged():
    e=[(0,1,60,1,1),(0,1,60,1,1),(1,1,60,1,1),(2,1,60,1,1)]
    a,c=attack_flags(e,[1,1,-1,None]);assert a.all() and c['ambiguous']==1


def test_simultaneous_real_attack_survives_tied_channel_collapse():
    e=[(0,1,60,1,1),(1,1,60,1,1),(1,1,60,2,1),(2,0,61,1,1)]
    r,c=tied_piano_roll(e,[1,-1,None,None],3)
    assert c['connected']==1 and r[:,1].sum()==2 and r[1,1,60,0]==1
