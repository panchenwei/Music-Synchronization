import torch
from src.learned_boundary_contrast import make_model,original,side_features


def test_context_order_and_padding():
    h=torch.zeros(1,30,32);h[:,15:]=1;valid=torch.ones(1,30,dtype=torch.bool)
    g=side_features(h,valid,'G');assert g[0,:,0].argmax()==15 and g[0,15,0]==1
    assert side_features(h,valid,'Z').count_nonzero()==0
    valid[:,25:]=False;a=side_features(h,valid,'G');h[:,25:]=10000;torch.testing.assert_close(a,side_features(h,valid,'G'),atol=0,rtol=0)


def test_identity_rng_and_gradients():
    torch.set_num_threads(2);x=torch.randn(2,24,58)
    for seed in (42,43):
        ref=original('C3',seed);cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
        m=make_model('G',seed);torch.testing.assert_close(cpu,torch.get_rng_state(),atol=0,rtol=0)
        for a,b in zip(gpu,torch.cuda.get_rng_state_all()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        ref.eval();m.eval();torch.testing.assert_close(m(x),ref(x),atol=0,rtol=0)
        m.train();m(x).square().mean().backward();assert torch.isfinite(m.residual.weight.grad).all() and m.residual.weight.grad.abs().sum()>0


def test_tiny_overfit():
    torch.set_num_threads(2);x=torch.randn(2,32,58);y=(x[:,:,0]>.2).float();m=make_model('G',42);opt=torch.optim.AdamW(m.parameters(),lr=.01)
    m.eval();initial=torch.nn.functional.binary_cross_entropy_with_logits(m(x),y).item()
    for _ in range(100):
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x),y);loss.backward();opt.step()
    final=torch.nn.functional.binary_cross_entropy_with_logits(m(x),y).item();assert final<initial*.25
