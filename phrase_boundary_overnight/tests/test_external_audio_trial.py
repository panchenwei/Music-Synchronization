import numpy as np
import torch
from src.external_audio_trial import make_model,mask_modalities,splits,Sampler,metrics
from src.models import Normalizer


def test_groups_and_modalities():
    sp=splits([str(i) for i in range(14)]);assert sum(len(v['test']) for v in sp.values())==14
    assert len({g for v in sp.values() for g in v['test']})==14
    x=np.ones((4,121),np.float32)
    for k in ('S','E','F'):
        z=mask_modalities(x,k);np.testing.assert_array_equal(z[:,:28],x[:,:28]);assert z[:,-1].all()
    assert mask_modalities(x,'S')[:,28:120].sum()==0 and mask_modalities(x,'E')[:,32:120].sum()==0


def test_sampler_model_and_metric():
    torch.set_num_threads(2);v=dict(features=np.ones((70,121),np.float32),labels=np.zeros(70,np.float32),label_mask=np.ones(70,np.float32),piece_id='p',performance_id='a',group='g')
    v['labels'][20]=1;data={'a':v};norm=Normalizer(np.zeros(121),np.ones(121));a=Sampler(data,norm,'S',42);b=Sampler(data,norm,'F',42)
    one=a.batch();two=b.batch()
    for i in (1,2,3):assert torch.equal(one[i],two[i])
    model=make_model(42);p=model(one[0],padding_mask=~one[3].bool());assert p.shape==(32,64)
    loss=(torch.nn.functional.binary_cross_entropy_with_logits(p,one[1],reduction='none')*one[2]).mean();loss.backward();assert torch.isfinite(loss)
    raw=np.zeros(70);raw[20]=.8;assert metrics({'a':raw},data,.5)[2]['macro_f1_tol1']==1
