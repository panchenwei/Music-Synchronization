from src.resolve_tie_audit import predecessors


def test_cross_voice_tie_is_not_a_new_attack():
    events=[(12.,1.,73,1,2),(13.,1/3,73,1,1)]
    assert predecessors(events,[1,-1],1)==('same_staff_cross_voice',[0])


def test_pitch_duration_and_right_tie_are_required():
    events=[(0.,1.,60,1,1),(1.,1.,60,1,1)]
    assert predecessors(events,[None,-1],1)==('unresolved',[])
    assert predecessors(events,[1,-1],1)==('same_staff_voice',[0])
    assert predecessors([(0.,.5,60,1,1),events[1]],[1,-1],1)==('unresolved',[])


def test_ambiguity_is_not_silently_resolved():
    events=[(0.,1.,60,1,1),(0.,1.,60,1,1),(1.,1.,60,1,1)]
    assert predecessors(events,[1,1,-1],2)==('same_staff_voice',[0,1])
