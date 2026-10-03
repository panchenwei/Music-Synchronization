import numpy as np
from src.window_exposure_audit import exposure
from src.phase7_models import CurvePieceBalancedSampler
from src.models import Normalizer


def test_trace_matches_real_sampler_rng_and_valid_loss_counts():
    data={}
    for i,n in enumerate((17,83)):
        mask=np.ones(n,'float32');mask[::7]=0
        data[str(i)]=dict(curves=np.zeros((i+2,n,9),'float32'),labels=np.zeros(n,'float32'),label_mask=mask)
    sampler=CurvePieceBalancedSampler(data,Normalizer(np.zeros(9),np.ones(9)),64,32,42)
    observed=0
    for _ in range(7):observed+=float(sampler.batch()[2].sum())
    trace=exposure(data,42,7)
    assert trace['sampler']==sampler.state() and trace['valid_loss_beat_exposures']==observed
    assert trace['window_draws']==224 and trace['score_window_coverage']==1


def test_short_budget_is_not_claimed_to_cover_all_performances():
    data={'p':dict(curves=np.zeros((100,200,9),'float32'),labels=np.zeros(200,'float32'),label_mask=np.ones(200,'float32'))}
    x=exposure(data,1,1)
    assert x['unique_performance_windows']<=32 and x['performance_window_coverage']<1
