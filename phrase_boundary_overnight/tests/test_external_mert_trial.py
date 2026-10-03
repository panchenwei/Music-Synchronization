import numpy as np
from src.external_mert_trial import mask_modalities
from src.external_audio_trial import make_model


def test_mert_masks_and_fixed_architecture():
    x=np.ones((2,64,121),np.float32);x[...,60:120]=0
    for kind in ('E','F'):np.testing.assert_array_equal(mask_modalities(x,kind),x)
    masked=mask_modalities(x,'S');assert (masked[...,28:120]==0).all() and (masked[...,:28]==1).all() and (masked[...,120]==1).all()
    assert x[...,28:60].sum()>0
    model=make_model(42);assert sum(p.numel() for p in model.parameters())==7937
