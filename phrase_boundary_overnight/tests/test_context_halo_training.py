import numpy as np
import torch
from src.context_halo_training import HaloSampler,HALO,make_model
from src.phase7_models import CurvePieceBalancedSampler
from src.models import Normalizer


def fixture():
    rng=np.random.default_rng(34)
    data={str(n):dict(curves=rng.normal(size=(2,n,58)).astype(np.float32),labels=(np.arange(n)%12==0).astype(np.float32),label_mask=np.ones(n,np.float32)) for n in (37,100,155)}
    return data,Normalizer(np.zeros(58,np.float32),np.ones(58,np.float32))


def test_same_rng_draws_and_supervised_content():
    data,norm=fixture();old=CurvePieceBalancedSampler(data,norm,64,8,42);new=HaloSampler(data,norm,64,8,42)
    for _ in range(7):
        a=old.batch();b=new.batch();assert old.state()==new.state()
        for x,y in zip(a,b):torch.testing.assert_close(x,y[:,HALO:HALO+64],atol=0,rtol=0)
        assert not b[2][:,:HALO].any() and not b[2][:,HALO+64:].any()


def test_full_context_for_every_supervised_position():
    torch.set_num_threads(2);data,norm=fixture();sampler=HaloSampler(data,norm,64,12,42)
    x,y,loss,valid=sampler.batch()
    for kind in ('HC3','HA2'):
        model=make_model(kind,42).eval()
        if kind=='HA2':
            with torch.no_grad():
                for layer in model.attention:layer.proj.weight.normal_(std=.1);layer.ffn[-1].weight.normal_(std=.1)
        out=model(x,padding_mask=~valid.bool())
        for i,(pid,perf,start) in enumerate(sampler.last_selection):
            full=model(torch.from_numpy(norm.apply(data[pid]['curves'][perf])).float()[None])[0]
            core=min(64,len(full)-start)
            torch.testing.assert_close(out[i,HALO:HALO+core],full[start:start+core],atol=4e-6,rtol=4e-6)
