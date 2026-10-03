import numpy as np
import pandas as pd
from src.score_local_coordinates import positioned_rows,local_events,local_labels


def test_missing_folded_position_uses_local_measure_offset():
    m=pd.DataFrame(dict(mc=[1,2],quarterbeats=[0,np.nan],duration_qb=[3,3],next=['2','-1']))
    notes=pd.DataFrame(dict(mc=[1,2],mc_onset=['1/4','1/2'],quarterbeats=[1,np.nan],duration_qb=[1,1],midi=[60,62],staff=[1,1],voice=[1,1],tied=[np.nan,np.nan]))
    e,t=local_events(notes,m,6);assert [x[0] for x in e]==[1,5] and t==[None,None]
    h=pd.DataFrame(dict(mc=[1,2],mc_onset=['0','1/4'],quarterbeats=[0,np.nan],phraseend=['I{','V{']))
    y,mask,*_=local_labels(h,m,6);assert y[4]==1 and y.sum()==1 and mask[4]==1


def test_repeat_visits_keep_each_occurrence():
    m=pd.DataFrame(dict(mc=[1,2],quarterbeats=[0,3],duration_qb=[3,3],next=['2','1, -1']))
    rows=pd.DataFrame(dict(mc=[1],mc_onset=['0']))
    positioned,mode,_=positioned_rows(rows,m,12)
    assert mode=='dcml_next_graph' and [r[0] for r in positioned]==[0,6]
