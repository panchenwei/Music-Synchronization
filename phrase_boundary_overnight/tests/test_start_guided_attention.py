import torch
from src.start_guided_attention import make_model,attention_target
from src.recurrence_local_attention import make_model as old


def test_targets_respect_starts_and_unknown_holes():
    y=torch.tensor([[0,0,1,0,0,0,1,0]],dtype=torch.float32);known=torch.ones_like(y)
    a,_=attention_target(y,known,2,'G');d,_=attention_target(y,known,2,'D')
    assert a[0,1,3]==0 and a[0,2,1]==0 and d[0,1,3]>0
    torch.testing.assert_close(a.sum(-1),known)
    known[0,4]=0;a,_=attention_target(y,known,2,'G')
    assert a[0,4].sum()==0 and a[0,3,4]==0 and a[0,5,0]==0
    d1,_=attention_target(y,known,2,'D');d2,_=attention_target(1-y,known,2,'D')
    torch.testing.assert_close(d1,d2,atol=0,rtol=0)


def test_initial_a1_equivalence_padding_and_guidance_gradients():
    torch.set_num_threads(2)
    m=make_model('G',42).eval();ref=old('A1',42).eval()
    for k,v in ref.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
    x=torch.randn(2,25,58);mask=torch.zeros(2,25,dtype=torch.bool);mask[0,19:]=True
    torch.testing.assert_close(m(x,mask),ref(x,mask),atol=1e-6,rtol=1e-5)
    x2=x.clone();x2[mask]=10000
    torch.testing.assert_close(m(x,mask),m(x2,mask),atol=0,rtol=0)
    y=torch.zeros(2,25);y[:,8::8]=1
    m(x,mask);penalty=m.guidance(y,~mask,'G');assert torch.isfinite(penalty)
    penalty.backward();assert m.attention[0].qkv.weight.grad.abs().sum()>0
    before=m(x,mask).detach().clone();m.guidance(1-y,~mask,'G')
    torch.testing.assert_close(before,m(x,mask),atol=0,rtol=0)


def test_tiny_overfit_and_auxiliary_optimization():
    torch.set_num_threads(2);m=make_model('G',8);m.eval()
    x=torch.randn(2,24,58);y=torch.zeros(2,24);y[:,::6]=1;x[:,:,0]=2*y-1
    known=torch.ones_like(y);opt=torch.optim.Adam(m.parameters(),lr=.01)
    m(x);first=float(m.guidance(y,known,'G').detach())
    for _ in range(100):
        opt.zero_grad();z=m(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)+.1*m.guidance(y,known,'G');loss.backward();opt.step()
    z=m(x);assert ((z>0)==y.bool()).float().mean()>.98
    assert float(m.guidance(y,known,'G').detach())<first
