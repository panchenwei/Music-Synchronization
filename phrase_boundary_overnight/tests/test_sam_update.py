import copy
import pytest
import torch
from src.sam_update import step
from src.recurrence_depth_models import make_model


def fixture():
    torch.set_num_threads(2);m=make_model('C3',42)
    x=torch.randn(2,16,58);y=(torch.arange(16)%5==0).float().repeat(2,1)
    valid=torch.ones_like(y);valid[1,-3:]=0;mask=valid.clone();mask[:,0]=0
    return m,(x,y,mask,valid),torch.nn.BCEWithLogitsLoss(reduction='none')


def test_zero_is_exact_original_update():
    m,b,c=fixture();n=copy.deepcopy(m);a=torch.optim.AdamW(m.parameters(),lr=.001);o=torch.optim.AdamW(n.parameters(),lr=.001)
    state=torch.get_rng_state();x,y,mask,valid=b
    m.train();a.zero_grad(set_to_none=True);loss=(c(m(x,padding_mask=~valid.bool()),y)*mask).sum()/mask.sum();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);a.step();rng=torch.get_rng_state()
    torch.set_rng_state(state);result=step(n,o,b,c,0)
    assert result['loss']==float(loss.detach()) and torch.equal(rng,torch.get_rng_state())
    for p,q in zip(m.parameters(),n.parameters()):assert torch.equal(p,q)


def test_sam_radius_rng_and_restore_before_optimizer():
    m,b,c=fixture();initial=copy.deepcopy(m.state_dict());state=torch.get_rng_state()
    m.train();m(b[0],padding_mask=~b[3].bool());rng=torch.get_rng_state();torch.set_rng_state(state)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    def pre(_opt,_args,_kwargs):
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items())
    hook=opt.register_step_pre_hook(pre);r=step(m,opt,b,c,.05);hook.remove()
    assert abs(r['radius']-.05)<1e-5 and r['perturbed_loss']>r['loss']
    assert torch.equal(rng,torch.get_rng_state())
    assert any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items())


def test_sam_mask_and_tiny_overfit():
    m,b,c=fixture();n=copy.deepcopy(m);x,y,mask,valid=b;other=y.clone();other[mask==0]=1-other[mask==0]
    a=torch.optim.AdamW(m.parameters(),lr=.003);o=torch.optim.AdamW(n.parameters(),lr=.003);state=torch.get_rng_state()
    first=step(m,a,b,c,.05);torch.set_rng_state(state);second=step(n,o,(x,other,mask,valid),c,.05)
    assert first==second
    for p,q in zip(m.parameters(),n.parameters()):assert torch.equal(p,q)
    for _ in range(79):last=step(m,a,b,c,.05)
    assert last['loss']<first['loss']*.3


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_gpu_sam_repeated_restore_dropout_stream_and_radius():
    m,b,c=fixture();m=m.cuda();b=tuple(t.cuda() for t in b)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    # Populate AdamW moments, then prove an actual SAM update can be replayed.
    step(m,opt,b,c,.05)
    weights=copy.deepcopy(m.state_dict());optimizer=copy.deepcopy(opt.state_dict())
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state()
    m.train();m(b[0],padding_mask=~b[3].bool());expected_cuda=torch.cuda.get_rng_state()
    torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
    result=step(m,opt,b,c,.05);final=copy.deepcopy(m.state_dict());final_opt=copy.deepcopy(opt.state_dict())
    assert torch.equal(expected_cuda,torch.cuda.get_rng_state()) and abs(result['radius']-.05)<1e-5
    m.load_state_dict(weights);opt.load_state_dict(copy.deepcopy(optimizer));torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
    assert step(m,opt,b,c,.05)==result
    assert torch.equal(expected_cuda,torch.cuda.get_rng_state())
    for k,v in final.items():assert torch.equal(v,m.state_dict()[k])
    for i,d in final_opt['state'].items():
        for k,v in d.items():torch.testing.assert_close(v,opt.state_dict()['state'][i][k],atol=0,rtol=0)
