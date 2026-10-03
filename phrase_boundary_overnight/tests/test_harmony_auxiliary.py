import numpy as np
import pandas as pd
import torch
from src.harmony_auxiliary import targets,degree,shuffled,make_model,HarmonySampler
from src.recurrence_depth_models import make_model as cnn
from src.models import Normalizer


def test_conservative_targets():
    measures=pd.DataFrame([dict(mc=1,duration_qb=4,next='2'),dict(mc=2,duration_qb=4,next='-1')])
    rows=[dict(mc=1,mc_onset=0,duration_qb=1,numeral='I'),
          dict(mc=1,mc_onset='1/4',duration_qb=.5,numeral='V'),
          dict(mc=1,mc_onset='3/8',duration_qb=.5,numeral='I'),
          dict(mc=1,mc_onset='1/2',duration_qb=1,numeral='V',relativeroot='V'),
          dict(mc=1,mc_onset='3/4',duration_qb=8,numeral='ii')]
    frame=pd.DataFrame([dict(globalkey='C',localkey='I',**r) for r in rows])
    y,m,_,_,_=targets(frame,measures,8)
    np.testing.assert_array_equal(m,[1,0,0,1,0,0,0,0]);assert y[0]==0 and y[3]==1
    assert degree(pd.Series(dict(numeral='bII',globalkey='C',localkey='i')))==-1


def test_model_same_core_padding_and_gradients():
    torch.set_num_threads(2);a=make_model('G',42).eval();b=cnn('C3',42).eval()
    x=torch.randn(2,16,58);mask=torch.zeros(2,16,dtype=torch.bool);mask[0,12:]=True
    logits,aux=a(x,mask,True);torch.testing.assert_close(logits,b(x,mask),atol=0,rtol=0)
    changed=x.clone();changed[0,12:]=10000
    torch.testing.assert_close(a(changed,mask),logits,atol=0,rtol=0)
    assert sum(p.numel() for p in a.parameters())==6152
    for k,v in make_model('D',42).state_dict().items():torch.testing.assert_close(v,a.state_dict()[k],atol=0,rtol=0)
    loss=torch.nn.functional.cross_entropy(aux[~mask],torch.arange((~mask).sum())%7);loss.backward()
    assert a.harmony_head.weight.grad.abs().sum()>0 and a.core.input_projection.weight.grad.abs().sum()>0


def test_sampler_same_rng_targets_and_resume():
    v=dict(curves=np.random.default_rng(2).normal(size=(2,70,58)).astype(np.float32),labels=np.zeros(70,np.float32),label_mask=np.ones(70,np.float32),
           harmony_labels=np.arange(70)%7,harmony_mask=np.r_[np.ones(60),np.zeros(10)].astype(np.float32))
    norm=Normalizer(np.zeros(58),np.ones(58));g=HarmonySampler({'p':v},norm,42,'G');d=HarmonySampler({'p':v},norm,42,'D')
    a=g.batch();b=d.batch()
    for i in (0,1,2,3,5):assert torch.equal(a[i],b[i])
    assert not torch.equal(a[4],b[4]);state=g.state();c=g.batch();g.load_state(state)
    for x,y in zip(c,g.batch()):assert torch.equal(x,y)
    shuffled(v['harmony_labels'],v['harmony_mask'],'p',42)
