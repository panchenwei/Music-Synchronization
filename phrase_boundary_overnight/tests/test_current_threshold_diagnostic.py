import pandas as pd
from src.current_threshold_diagnostic import select_threshold


def test_excluded_opus_does_not_choose_threshold():
    frame=pd.DataFrame([dict(opus=o,threshold=t,f1_tol1=f,f1_tol0=f,precision_tol1=f,recall_tol1=f) for o,t,f in [('a',.2,.8),('a',.8,.4),('b',.2,.1),('b',.8,.9)]])
    assert select_threshold(frame,'b')==.2
    frame.loc[frame.opus=='b','f1_tol1']=100;assert select_threshold(frame,'b')==.2
