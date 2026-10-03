import pandas as pd
from src.prepare_external_score_probe import note_fingerprint


def test_fingerprint_transposition_and_voice_row_order():
    a=pd.DataFrame(dict(mc=[1,1,2],mc_onset=['0','1/4','0'],duration_qb=[1.,1.,2.],midi=[60,64,67]))
    b=a.iloc[::-1].copy();b['midi']+=3
    assert note_fingerprint(a)==note_fingerprint(b)
    b.iloc[0,b.columns.get_loc('midi')]+=1
    assert note_fingerprint(a)!=note_fingerprint(b)


def test_fingerprint_ignores_duplicate_noteheads_not_changed_timing():
    a=pd.DataFrame(dict(mc=[1,2],mc_onset=['0','0'],duration_qb=[1.,2.],midi=[60,67]))
    assert note_fingerprint(a)==note_fingerprint(pd.concat([a,a.iloc[:1]],ignore_index=True))
    b=a.copy();b['duration_qb']*=2
    assert note_fingerprint(a)!=note_fingerprint(b)
