import pytest
import torch
import numpy as np
from src.local_context_study import make_model, MaskedConv, predictions, max_match_count
from src.models import Normalizer
from src.evaluation import one_to_one_counts

@pytest.mark.parametrize('kind',['A9','B9','C9','A25','B25','C25','M25','L25'])
def test_shapes_padding_with_trained_norm_bias(kind):
    model=make_model(kind).eval()
    dim=9 if kind.endswith('9') else 25
    for module in model.modules():
        if isinstance(module,torch.nn.LayerNorm):module.bias.data.fill_(.13)
    x=torch.randn(2,13,dim); padded=torch.cat([x,torch.randn(2,6,dim)*100],dim=1)
    mask=torch.zeros(2,19,dtype=torch.bool);mask[:,13:]=True
    with torch.no_grad():
        a=model(x); b=model(padded,mask)
    assert a.shape==(2,13)
    assert torch.allclose(a,b[:,:13],atol=2e-6)
    assert (b[:,13:]==0).all()

@pytest.mark.parametrize('kind',['A25','B25','M25','L25'])
def test_inference_restores_mode(kind):
    model=make_model(kind).train()
    data={'p':{'curves':np.random.default_rng(1).normal(size=(1,9,25)).astype('float32'),'performance_ids':['x']}}
    norm=Normalizer(np.zeros(25),np.ones(25))
    predictions(model,data,norm,'cpu')
    assert model.training and all(m.training for m in model.modules())
    model.eval(); predictions(model,data,norm,'cpu'); assert not model.training

def test_shared_transformer_initialization():
    a=make_model('A25').state_dict()
    for kind in ['B25','M25','L25']:
        b=make_model(kind).state_dict()
        assert all(torch.equal(v,b[k]) for k,v in a.items())

def test_maximum_matching_counterexample():
    assert one_to_one_counts([1,2],[0,1],1).tp==1
    assert max_match_count([1,2],[0,1],1)==2

@pytest.mark.parametrize('kind',['B25','M25','L25'])
def test_gradients(kind):
    model=make_model(kind).train()
    y=model(torch.randn(2,16,25));y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
