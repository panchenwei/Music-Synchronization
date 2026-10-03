import numpy as np
import torch
from src.audit_axis_auxiliary_information import diagnose
from src.axis_auxiliary_model import AuxiliaryBoundary
from src.models import Normalizer


def test_auxiliary_probe_only_uses_main_inference_output():
    torch.set_num_threads(2);rng=np.random.default_rng(4)
    roll=np.zeros((9,2,128,4),np.float32)
    for t in range(9):roll[t,:,60+t%4,t%4]=1
    data={'toy':dict(curves=rng.normal(size=(1,9,58)).astype(np.float32),labels=np.zeros(9),label_mask=np.ones(9),performance_ids=['a'],piano_roll=roll)}
    model=AuxiliaryBoundary('A',42).cuda()
    with torch.no_grad():model.core.input_projection.weight[:,58:].fill_(.1)
    outputs,stats=diagnose(model,data,Normalizer(np.zeros(58),np.ones(58)))
    np.testing.assert_array_equal(outputs['zero_embedding']['toy']['a'],outputs['silent_roll']['toy']['a'])
    assert np.isfinite(outputs['intact']['toy']['a']).all() and len(stats)==1
    with torch.no_grad():
        for p in model.auxiliary.parameters():p.fill_(999)
    changed,_=diagnose(model,data,Normalizer(np.zeros(58),np.ones(58)))
    np.testing.assert_array_equal(outputs['intact']['toy']['a'],changed['intact']['toy']['a'])
