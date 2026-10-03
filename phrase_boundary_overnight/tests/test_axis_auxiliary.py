import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from src.axis_content_centering import ContentBoundary
from src.axis_auxiliary_model import AuxiliaryBoundary,AuxiliaryBCE
from src.axis_split_music import features_in_blocks


def inputs():
    x=torch.randn(2,12,58,device='cuda');r=torch.zeros(2,12,2,128,4,device='cuda')
    for t in range(12):r[:,t,:,60+t%3,t%4]=1
    mask=torch.arange(12,device='cuda')[None].expand(2,-1)>=10
    y=torch.zeros(2,12,device='cuda');y[:,[3,7]]=1
    return x,r,mask,y


def test_no_aux_control_updates_match_centered_gpu():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    c=ContentBoundary('C',42).cuda();a=AuxiliaryBoundary('A',42).cuda()
    assert sum(p.numel() for p in a.parameters())==8834
    x,r,mask,y=inputs();criterion=AuxiliaryBCE(reduction='none',pos_weight=torch.tensor(10.,device='cuda'))
    criterion.weight_auxiliary=0
    opts=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in (c,a)]
    for step in range(3):
        for m,opt in zip((c,a),opts):
            torch.manual_seed(900+step);m.train();opt.zero_grad()
            logits=m(x,r,padding_mask=mask)
            loss=(criterion(logits,y)*(~mask)).sum()/(~mask).sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        for key,value in c.state_dict().items():torch.testing.assert_close(value,a.state_dict()[key],atol=0,rtol=0)
    c.eval();a.eval();torch.testing.assert_close(c(x,r,mask),a(x,r,mask),atol=0,rtol=0)


def test_aux_supervision_reaches_stem_and_padding_is_masked():
    a=AuxiliaryBoundary('A',42).cuda();x,r,mask,y=inputs();a.train()
    main,aux=a(x,r,mask)
    criterion=AuxiliaryBCE(reduction='none',pos_weight=torch.tensor(10.,device='cuda'))
    actual=criterion((main,aux),y)
    bce=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(10.,device='cuda'))
    torch.testing.assert_close(actual,bce(main,y)+.25*bce(aux,y))
    loss=(actual*(~mask)).sum()/(~mask).sum();loss.backward()
    assert a.stem.base.input.weight.grad.abs().max()>0
    assert a.auxiliary[0].weight.grad.abs().max()>0
    assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
    a.eval();dirty=r.clone();dirty[:,10:]=999
    torch.testing.assert_close(a(x,r,mask),a(x,dirty,mask),atol=0,rtol=0)
    torch.testing.assert_close(features_in_blocks(a,r,5),a.stem(r),atol=5e-4,rtol=0)
    assert torch.count_nonzero(a.stem(torch.zeros_like(r)))==0
