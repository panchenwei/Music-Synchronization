import numpy as np
import torch
from src.models import Normalizer
from src.recurrence_depth_models import make_model
from src.recurrence_transfer_core import transfer,shuffled_labels


def test_transfer_preserves_external_function_and_unseen_inputs():
    torch.set_num_threads(2)
    ext=make_model('C3',42).eval()
    with torch.no_grad():ext.input_projection.weight[:,34:].normal_(0,.05)
    target=make_model('C3',42).eval();before=target.input_projection.weight.detach().clone()
    en=Normalizer(np.r_[np.zeros(34),np.full(24,.3)].astype('float32'),np.r_[np.ones(34),np.full(24,.2)].astype('float32'))
    tn=Normalizer(np.r_[np.zeros(34),np.full(24,.6)].astype('float32'),np.r_[np.ones(34),np.full(24,.15)].astype('float32'))
    transfer(target,ext.state_dict(),en,tn)
    x=np.random.default_rng(9).random((2,30,58)).astype('float32');x[:,:,:34]=0
    torch.testing.assert_close(ext(torch.from_numpy(en.apply(x))),target(torch.from_numpy(tn.apply(x))),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(before[:,:34],target.input_projection.weight[:,:34],atol=0,rtol=0)


def test_shuffled_control_preserves_counts_and_unknown_mask():
    y=np.zeros(80,dtype='float32');y[::7]=1
    m=np.ones(80);m[:3]=0;m[-6:]=0
    z=shuffled_labels(y,m,'a')
    np.testing.assert_array_equal(z,shuffled_labels(y,m,'a'))
    np.testing.assert_array_equal(z[m==0],y[m==0])
    assert z[m>0].sum()==y[m>0].sum() and not np.array_equal(y,z)
