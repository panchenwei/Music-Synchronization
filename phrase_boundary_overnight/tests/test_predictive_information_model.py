import torch
from src.predictive_information_model import make_model,make_base


def test_initial_core_and_rng_streams():
    for seed in (42,43):
        base=make_base('C3',seed);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
        models=[make_model(k,seed) for k in ('G','D','Z')]
        for model in models:
            assert sum(p.numel() for p in model.parameters())==6369
            for k,v in base.state_dict().items():
                actual=model.state_dict()[k]
                if k=='input_projection.weight':
                    assert torch.count_nonzero(actual[:,58:])==0;actual=actual[:,:58]
                torch.testing.assert_close(v,actual,atol=0,rtol=0)
        assert torch.equal(cpu,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))


def test_padding_gradient_and_checkpoint(tmp_path):
    torch.set_num_threads(2);model=make_model('G',42);x=torch.randn(2,64,72)
    mask=torch.zeros(2,64,dtype=torch.bool);mask[1,48:]=True;model.eval()
    y=model(x,padding_mask=mask);changed=x.clone();changed[mask]=999
    torch.testing.assert_close(y[~mask],model(changed,padding_mask=mask)[~mask],atol=0,rtol=0)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(y[~mask],(x[:,:,60]>0).float()[~mask]);loss.backward()
    assert torch.isfinite(loss) and model.input_projection.weight.grad[:,58:].abs().sum()>0
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
    clone=make_model('G',42).eval();clone.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(y,clone(x,padding_mask=mask),atol=0,rtol=0)
