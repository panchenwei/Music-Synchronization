import torch
from src.recurrence_dropout_models import make_model


def test_only_dropout_probability_changes_not_weights_initial_eval_or_rng():
    torch.set_num_threads(2)
    for seed in (42,43):
        a=make_model('C3',seed);rng=torch.get_rng_state().clone();b=make_model('D4',seed)
        assert torch.equal(rng,torch.get_rng_state())
        assert sum(p.numel() for p in b.parameters())==5921
        assert [layer.dropout.p for layer in a.frontend.layers]==[.2]*3
        assert [layer.dropout.p for layer in b.frontend.layers]==[.4]*3
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
        x=torch.randn(2,64,58);mask=torch.zeros(2,64,dtype=torch.bool);mask[1,41:]=True
        a.eval();b.eval();torch.testing.assert_close(a(x,padding_mask=mask),b(x,padding_mask=mask),atol=0,rtol=0)
        original=b(x,padding_mask=mask);x[mask]=9999
        torch.testing.assert_close(original,b(x,padding_mask=mask),atol=0,rtol=0)


def test_training_dropout_has_an_effect_and_finite_gradients():
    a=make_model('C3',42).train();b=make_model('D4',42).train()
    x=torch.randn(3,64,58);y=torch.rand(3,64)
    rng=torch.get_rng_state().clone();p=a(x);torch.set_rng_state(rng);q=b(x)
    assert not torch.equal(p,q)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(q,y);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
