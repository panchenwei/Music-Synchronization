import torch
from src.axis_content_centering import ContentBoundary
from src.axis_split_music import features_in_blocks
from src.recurrence_depth_models import make_model


def test_blank_zero_and_chunk_equivalence():
    torch.set_num_threads(2)
    for kind in ('C','N'):
        m=ContentBoundary(kind,42).eval();r=torch.rand(1,29,2,128,4)
        assert torch.count_nonzero(m.stem(torch.zeros_like(r)))==0
        torch.testing.assert_close(features_in_blocks(m,r,7),m.stem(r),atol=5e-4,rtol=0)
        mask=torch.arange(29)[None]>=23;dirty=r.clone();dirty[:,23:]=999
        torch.testing.assert_close(m.stem(r,mask),m.stem(dirty,mask),atol=0,rtol=0)
        torch.testing.assert_close(m.stem(r[:,:23]),m.stem(r,mask)[:,:23],atol=5e-4,rtol=0)


def test_shared_weights_initial_baseline_and_learning():
    a=ContentBoundary('C',42);b=ContentBoundary('N',42)
    assert sum(p.numel() for p in a.parameters())==8497
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    x=torch.randn(2,12,58);r=torch.zeros(2,12,2,128,4)
    for t in range(12):r[:,t,:,60+t%3,t%4]=1
    for m in (a,b):
        m.eval();torch.testing.assert_close(m(x,r),make_model('C3',42).eval()(x),atol=2e-6,rtol=0)
        before=m.stem.base.input.weight.detach().clone();opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for _ in range(3):
            m.train();opt.zero_grad();loss=(m(x,r)-1).square().mean();assert torch.isfinite(loss)
            loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step()
        assert not torch.equal(before,m.stem.base.input.weight)
