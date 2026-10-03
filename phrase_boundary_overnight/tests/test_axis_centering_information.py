import numpy as np
import pytest
import torch
from src.audit_axis_music_information import diagnose
from src.axis_content_centering import ContentBoundary
from src.models import Normalizer


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU probe')
def test_centered_silence_equals_removing_embedding():
    torch.set_num_threads(2);rng=np.random.default_rng(4)
    roll=np.zeros((9,2,128,4),np.float32)
    for t in range(9):roll[t,:,60+t%4,t%4]=1
    data={'toy':dict(curves=rng.normal(size=(1,9,58)).astype(np.float32),labels=np.zeros(9),label_mask=np.ones(9),performance_ids=['a'],piano_roll=roll)}
    before=roll.copy()
    for kind in ('C','N'):
        model=ContentBoundary(kind,42).cuda()
        with torch.no_grad():model.core.input_projection.weight[:,58:].copy_(torch.randn(32,8,generator=torch.Generator().manual_seed(12)).cuda()*.1)
        outputs,stats=diagnose(model,data,Normalizer(np.zeros(58),np.ones(58)))
        np.testing.assert_array_equal(outputs['zero_embedding']['toy']['a'],outputs['silent_roll']['toy']['a'])
        assert np.isfinite(outputs['intact']['toy']['a']).all() and len(stats)==1
        np.testing.assert_array_equal(roll,before)
