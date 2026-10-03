import numpy as np
import torch
from src.models import Normalizer
from src.recurrence_depth_models import make_model
from src.recurrence_transfer_core import transfer
from src.transfer_head_reset import transfer_without_head


def test_only_output_head_differs_without_random_draws():
    ext=make_model('C3',91)
    with torch.no_grad():
        ext.output.weight.fill_(.4);ext.output.bias.fill_(.9)
    norm=Normalizer(np.zeros(58,np.float32),np.ones(58,np.float32))
    a=make_model('C3',42);b=make_model('C3',42)
    original={k:v.clone() for k,v in b.output.state_dict().items()}
    transfer(a,ext.state_dict(),norm,norm)
    rng=torch.get_rng_state().clone()
    transfer_without_head(b,ext.state_dict(),norm,norm)
    torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
    for k,v in a.state_dict().items():
        if not k.startswith('output.'):
            torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    for k,v in original.items():
        torch.testing.assert_close(v,b.output.state_dict()[k],atol=0,rtol=0)
    assert not torch.equal(a.output.weight,b.output.weight)
