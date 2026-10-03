import numpy as np
from src.tie_aware_roll import build
from src.score_piano_roll import piano_roll


def test_duration_occupancy_preserved_and_crossvoice_attack_removed():
    e=[(0.,1.,60,1,2),(1.,1.,60,1,1)]
    r,info=build(e,[1,-1],3);old=piano_roll(e,3)
    np.testing.assert_array_equal(r[:,0],old[:,0])
    assert r[0,1,60,0]==1 and r[1,1,60,0]==0
    assert info['verified_continuations']==1


def test_unresolved_continuation_not_silently_deleted():
    r,info=build([(1.,1.,60,1,1)],[-1],3)
    assert r[1,1,60,0]==1 and info['unresolved_continuations']==1


def test_another_real_attack_at_same_pitch_tick_survives():
    e=[(0.,1.,60,1,1),(1.,1.,60,1,1),(1.,.5,60,2,1)]
    r,info=build(e,[1,-1,None],3)
    assert info['verified_continuations']==1 and r[1,1,60,0]==1


def test_tie_chain_and_rearticulation():
    e=[(0.,1.,60,1,1),(1.,1.,60,1,1),(2.,1.,60,1,1),(3.,1.,60,1,1)]
    r,info=build(e,[1,0,-1,None],4)
    assert info['verified_continuations']==2
    np.testing.assert_array_equal(r[:,1,60,0],[1,0,0,1])


def test_one_predecessor_cannot_resolve_two_successors():
    e=[(0.,1.,60,1,1),(1.,1.,60,1,2),(1.,1.,60,1,3)]
    r,info=build(e,[1,-1,-1],3)
    assert info['verified_continuations']==0 and info['unresolved_continuations']==2
