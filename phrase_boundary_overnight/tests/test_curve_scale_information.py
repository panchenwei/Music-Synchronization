import numpy as np
import torch
from src.audit_curve_scale_information import intervene
from src.curve_scale_models import CurveScaleBoundary


def test_content_intervention_preserves_quality_old_inputs_and_source():
    torch.set_num_threads(2)
    x=np.random.default_rng(1).normal(size=(1,20,78)).astype(np.float32)
    data={'toy':{'curves':x}};before=x.copy()
    no=intervene(data,'zero_signed')['toy']['curves'];empty=intervene(data,'zero_all_new')['toy']['curves']
    np.testing.assert_array_equal(no[...,:58],x[...,:58]);np.testing.assert_array_equal(no[...,68:],x[...,68:])
    assert not np.any(no[...,58:68]) and not np.any(empty[...,58:])
    np.testing.assert_array_equal(x,before)
    for kind in ('F','S','J'):
        m=CurveScaleBoundary(kind,42).cuda().eval()
        with torch.no_grad():
            m.core.input_projection.weight[:,58:].fill_(.1)
            real=m(torch.from_numpy(x).cuda());removed=m(torch.from_numpy(empty).cuda())
        assert torch.isfinite(real).all() and torch.isfinite(removed).all()
        assert (real-removed).abs().max()>1e-7
