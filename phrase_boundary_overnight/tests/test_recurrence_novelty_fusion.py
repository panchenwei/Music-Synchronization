import numpy as np
import torch
from src.recurrence_novelty_fusion import extra_features, make_model


def test_only_novelty_values_change_not_availability_or_recurrence():
    rng=np.random.default_rng(4);o=rng.random((30,24)).astype('float32');n=rng.random((30,24)).astype('float32')
    a=extra_features(o,n,'A');f=extra_features(o,n,'F')
    assert a.shape==f.shape==(30,40)
    np.testing.assert_array_equal(a[:,:24],o);np.testing.assert_array_equal(f[:,:24],o)
    np.testing.assert_array_equal(a[:,36:],f[:,36:]);assert not a[:,24:36].any()
    np.testing.assert_array_equal(f[:,24:],n[:,:16])


def test_models_equal_capacity_initial_state_and_cpu_rng():
    a=make_model('A',42);state=torch.get_rng_state();f=make_model('F',42)
    torch.testing.assert_close(state,torch.get_rng_state(),atol=0,rtol=0)
    assert sum(p.numel() for p in a.parameters())==3809 and not a.frontend.wide
    for k,v in a.state_dict().items():torch.testing.assert_close(v,f.state_dict()[k],atol=0,rtol=0)


def test_initial_output_preserves_old_model_and_ignores_added_values():
    o=make_model('O',43).eval();f=make_model('F',43).eval()
    x=torch.randn(2,16,74)
    with torch.no_grad():torch.testing.assert_close(o(x[:,:,:58]),f(x),atol=1e-6,rtol=1e-6)
