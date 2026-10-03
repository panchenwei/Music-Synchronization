import numpy as np
import torch
from src.slice_energy_features import rest_before, voice_track, start_labels, tempo_hierarchy
from src.slice_energy_study import model_for

def test_rest_with_overlapping_voices():
    events=[(0,4,60,1,1),(0,1,48,2,1),(5,1,60,1,1)]
    r=rest_before(events,7)
    assert r[2]==0 and r[4]==0 and r[5]==1/8 and r[6]==0

def test_voice_does_not_jump_to_accompaniment():
    events=[(0,1,60,1,1),(1,1,62,1,1),(1,1,90,1,2),(3,1,59,1,1)]
    d,_,turn=voice_track(events,4,(1,1))
    np.testing.assert_array_equal(d,[0,2,0,-3])
    assert turn[3]==1

def test_slice_masks_and_tie():
    y,m,rows=start_labels([0,4,8.5,12],16)
    assert y[4]==y[12]==1 and y[0]==0
    assert m[0]==m[8]==m[9]==m[13]==0 and m[4]==1
    assert rows[2]['status']=='ambiguous_tie'

def test_missing_second_start_is_not_negative_work():
    _,mask,_=start_labels([3],16)
    assert mask.sum()==0

def test_hierarchy_constant_and_scale_invariance():
    a,relative,p=tempo_hierarchy(np.ones(31)*100)
    np.testing.assert_allclose(a[:,:5],0,atol=1e-7)
    np.testing.assert_allclose(relative,1)
    np.testing.assert_allclose(p,np.ones(31)/31)
    x=np.array([90,90,60,80,110]*7)
    np.testing.assert_allclose(tempo_hierarchy(x)[0],tempo_hierarchy(3*x)[0],atol=1e-6)
    assert np.isfinite(tempo_hierarchy(x)[0]).all()

def test_hierarchy_tail_is_not_zero_padded():
    x=np.r_[np.ones(24)*100,np.ones(2)*50]
    a,_,_=tempo_hierarchy(x)
    np.testing.assert_allclose(a[-2:,4],np.log(.5),atol=1e-6)

def test_expansion_starts_from_same_function():
    x=torch.randn(2,20,25)
    a=model_for(25,42).eval();b=model_for(31,42).eval()
    with torch.no_grad():
        torch.testing.assert_close(a(x),b(torch.cat([x,torch.randn(2,20,6)],dim=-1)))
    assert sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters())==192

def test_masked_padding_does_not_influence_valid_predictions():
    model=model_for(31,42).eval();x=torch.randn(2,20,31);mask=torch.zeros(2,20,dtype=torch.bool);mask[:,15:]=True
    with torch.no_grad():
        a=model(x,padding_mask=mask);x[:,15:]=999;b=model(x,padding_mask=mask)
    torch.testing.assert_close(a[:,:15],b[:,:15])
    assert (b[:,15:]==0).all()
