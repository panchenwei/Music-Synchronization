import numpy as np
import pytest
import torch
from src.audit_axis_music_information import diagnose
from src.axis_split_music import AxisBoundary
from src.models import Normalizer


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Diagnostic runs on GPU')
def test_information_conditions_and_no_input_mutation():
    torch.set_num_threads(2)
    rng=np.random.default_rng(42);roll=np.zeros((8,2,128,4),np.float32)
    roll[:,0,60,:]=1;roll[::2,1,60,0]=1
    item=dict(curves=rng.normal(size=(2,8,58)).astype(np.float32),labels=np.zeros(8),label_mask=np.ones(8),performance_ids=['a','b'],piano_roll=roll)
    original=roll.copy();model=AxisBoundary('S',42).cuda()
    with torch.no_grad():
        weights=torch.randn((32,8),generator=torch.Generator().manual_seed(123))*.1
        model.core.input_projection.weight[:,58:].copy_(weights.cuda())
    outputs,features=diagnose(model,{'toy':item},Normalizer(np.zeros(58),np.ones(58)))
    assert set(outputs)=={'intact','zero_embedding','silent_roll'}
    for values in outputs.values():
        assert set(values['toy'])=={'a','b'}
        for p in values['toy'].values():assert p.shape==(8,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert len(features)==1 and features[0]['piece_id']=='toy'
    assert np.max(abs(outputs['intact']['toy']['a']-outputs['zero_embedding']['toy']['a']))>1e-5
    assert np.isfinite([v for k,v in features[0].items() if k!='piece_id']).all()
    np.testing.assert_array_equal(roll,original)
