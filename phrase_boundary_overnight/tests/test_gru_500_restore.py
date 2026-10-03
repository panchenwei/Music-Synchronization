import copy
import numpy as np
import torch
from src.current_gru_models import make_model
from src.recurrence_duration_core import restore,update
from src.phase7_models import CurvePieceBalancedSampler
from src.models import Normalizer

def test_gru_optimizer_sampler_and_dropout_resume_exact():
    torch.set_num_threads(2);n=71
    data={'p':dict(curves=np.random.default_rng(8).normal(size=(2,n,58)).astype('float32'),labels=(np.arange(n)%7==0).astype('float32'),label_mask=np.ones(n,'float32'))}
    norm=Normalizer(np.zeros(58,'float32'),np.ones(58,'float32'));m=make_model('G',42)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);sampler=CurvePieceBalancedSampler(data,norm,64,4,42);crit=torch.nn.BCEWithLogitsLoss(reduction='none');device=torch.device('cpu')
    update(m,opt,sampler,crit,device)
    saved=copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),rng=torch.get_rng_state(),cuda_rng=[]))
    expected=update(m,opt,sampler,crit,device);weights=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();samp=sampler.state();optim=copy.deepcopy(opt.state_dict())
    for _ in range(2):
        restore(saved,m,opt,sampler);assert update(m,opt,sampler,crit,device)==expected
        assert samp==sampler.state() and torch.equal(rng,torch.get_rng_state())
        for k,v in weights.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
        for i,values in optim['state'].items():
            for k,v in values.items():torch.testing.assert_close(v,opt.state_dict()['state'][i][k],atol=0,rtol=0)
