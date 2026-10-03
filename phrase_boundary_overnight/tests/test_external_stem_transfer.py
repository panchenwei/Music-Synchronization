import numpy as np
import torch
from src.score_roll_branch import RollBoundary
from src.external_stem_transfer import transplant, external_split, external_dataset


def test_transfer_only_stem_and_freeze():
    a=RollBoundary('R',42);b=RollBoundary('R',42);source=RollBoundary('L',43)
    transplant(b,source.state_dict())
    for k,v in a.core.state_dict().items():torch.testing.assert_close(v,b.core.state_dict()[k],rtol=0,atol=0)
    for k,v in source.stem.state_dict().items():torch.testing.assert_close(v,b.stem.state_dict()[k],rtol=0,atol=0)
    assert sum(p.numel() for p in b.parameters())==6121
    assert sum(p.numel() for p in b.parameters() if p.requires_grad)==3297
    x=torch.randn(2,8,58);roll=torch.rand(2,8,2,128,4)
    a.eval();b.eval()
    torch.testing.assert_close(a(x,roll),b(x,roll),rtol=0,atol=0)
    b(x,roll).square().sum().backward()
    assert all(p.grad is None for p in b.stem.parameters())
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in b.core.parameters())


def test_external_split_and_no_positive_work_retained():
    df=external_split();assert len(df)==79
    train=external_dataset('train');assert len(train)==66
    item=train['grieg_lyric_pieces_op62n05']
    assert item['labels'][item['label_mask']>0].sum()==0 and item['label_mask'].sum()==73
    for item in train.values():
        assert np.count_nonzero(item['curves'])==0
        assert item['curves'].shape==(1,len(item['labels']),58)
        assert item['piano_roll'].shape==(len(item['labels']),2,128,4)
