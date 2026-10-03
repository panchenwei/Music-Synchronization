import torch
from src.recurrence_local_attention import make_model


def test_initial_baseline_padding_and_rng():
    torch.set_num_threads(2)
    a=make_model('C3',42).eval();state=torch.get_rng_state()
    x=torch.randn(2,80,58);mask=torch.zeros(2,80,dtype=torch.bool);mask[1,53:]=True
    for kind in ('A1','A2'):
        b=make_model(kind,42).eval();assert torch.equal(torch.get_rng_state(),state)
        torch.testing.assert_close(a(x,mask),b(x,mask),atol=1e-6,rtol=1e-6)
        dirty=x.clone();dirty[mask]=1e5
        torch.testing.assert_close(b(x,mask),b(dirty,mask),atol=0,rtol=0)
        assert (b(x,mask)[mask]==0).all()


def test_attention_learns_and_crop_interior_matches():
    torch.set_num_threads(2)
    for kind in ('A1','A2'):
        m=make_model(kind,42);opt=torch.optim.AdamW(m.parameters(),lr=.001)
        x=torch.randn(2,96,58);mask=torch.zeros(2,96,dtype=torch.bool);mask[1,40:]=True
        for step in range(2):
            opt.zero_grad();out=m(x,mask);out.square().mean().backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            if step==1:
                assert all(layer.qkv.weight.grad.abs().sum()>0 for layer in m.attention)
                assert all(layer.relative_bias.grad.abs().sum()>0 for layer in m.attention)
            opt.step()
        m.eval();full=m(x[:1]);crop=m(x[:1,16:80]);radius=6+8*len(m.attention)
        torch.testing.assert_close(full[:,16+radius:80-radius],crop[:,radius:64-radius],atol=2e-6,rtol=2e-6)
        # Masked padded keys and the CNN's post-normalization mask preserve all valid outputs.
        short=m(x[1:2,:40]);padded=m(x[1:2],mask[1:2])
        torch.testing.assert_close(short,padded[:,:40],atol=2e-6,rtol=2e-6)
