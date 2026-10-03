import torch
from src.auxiliary_gradient_projection import merge


def test_conflicting_and_separate_heads():
    a=[torch.tensor([1.,0.]),torch.tensor([3.]),None]
    b=[torch.tensor([-2.,2.]),None,torch.tensor([4.])]
    result,stats=merge(a,b,True)
    torch.testing.assert_close(result[0],torch.tensor([1.,2.]));assert stats['projected_dot']==0 and stats['applied']==1
    torch.testing.assert_close(result[1],a[1]);torch.testing.assert_close(result[2],b[2])
    control,_=merge(a,b,False);torch.testing.assert_close(control[0],torch.tensor([-1.,2.]))


def test_positive_and_zero():
    a=[torch.tensor([1.,0.])];b=[torch.tensor([2.,1.])]
    result,stats=merge(a,b,True);torch.testing.assert_close(result[0],a[0]+b[0]);assert stats['applied']==0
    result,stats=merge([torch.zeros(2)],b,True);torch.testing.assert_close(result[0],b[0])


def test_tiny_two_task_fit():
    from src.harmony_auxiliary import make_model
    torch.set_num_threads(2);torch.manual_seed(20260914);x=torch.randn(2,16,58);y=(x[...,0]>0).float();h=x[...,:7].argmax(-1)
    model=make_model('G',42).eval();params=list(model.parameters());opt=torch.optim.AdamW(params,lr=.01)
    def losses():
        a,b=model(x,both=True)
        return torch.nn.functional.binary_cross_entropy_with_logits(a,y),.25*torch.nn.functional.cross_entropy(b.reshape(-1,7),h.reshape(-1))
    before=float(sum(losses()).detach())
    for _ in range(100):
        main,aux=losses();gm=torch.autograd.grad(main,params,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(aux,params,allow_unused=True)
        gradients,_=merge(gm,ga,True);opt.zero_grad(set_to_none=True)
        for p,g in zip(params,gradients):p.grad=g
        torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
    assert float(sum(losses()).detach())<before*.2
