import numpy as np
import pandas as pd
import pytest
from src.fixed_ensemble_study import aligned_average


def frame(p):
    return pd.DataFrame(dict(piece_id=['x']*3,performance_id=['a']*3,beat=[0,1,2],label=[0,1,0],valid=[1,1,1],probability=p))


def test_align_before_probability_average():
    a=frame([.1,.8,.3]);b=frame([.5,.4,.1]).iloc[::-1]
    out=aligned_average([a,b]);np.testing.assert_allclose(out.probability,[.3,.6,.2])
    np.testing.assert_array_equal(a.probability,[.1,.8,.3])


def test_reject_metadata_duplicates_and_bad_probability():
    a=frame([.1,.8,.3])
    b=a.copy();b.loc[1,'label']=0
    with pytest.raises(ValueError):aligned_average([a,b])
    with pytest.raises(ValueError):aligned_average([pd.concat([a,a.iloc[:1]])])
    b=a.copy();b.loc[1,'probability']=float('nan')
    with pytest.raises(ValueError):aligned_average([b])
    with pytest.raises(ValueError):aligned_average([])
