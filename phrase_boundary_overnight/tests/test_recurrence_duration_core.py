import copy
import numpy as np
import torch
from src.recurrence_depth_models import make_model
from src.recurrence_duration_core import restore, update, selected_step
from src.phase7_models import CurvePieceBalancedSampler
from src.models import Normalizer


def test_prefix_selection_keeps_old_best_and_first_tie():
    old=[dict(step=50,macro_f1_tol1=.5),dict(step=300,macro_f1_tol1=.4)]
    assert selected_step(old+[dict(step=350,macro_f1_tol1=.5)])==50
    assert selected_step(old+[dict(step=350,macro_f1_tol1=.51)])==350


def test_cpu_continuation_restores_optimizer_sampler_and_dropout_rng():
    torch.set_num_threads(2)
    n=71
    data={'p':dict(curves=np.random.default_rng(8).normal(size=(2,n,58)).astype('float32'),
                  labels=np.arange(n,dtype='float32')%7==0,label_mask=np.ones(n,'float32'))}
    norm=Normalizer(np.zeros(58,dtype='float32'),np.ones(58,dtype='float32'))
    m=make_model('C3',42);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    sampler=CurvePieceBalancedSampler(data,norm,64,32,42)
    crit=torch.nn.BCEWithLogitsLoss(reduction='none')
    update(m,opt,sampler,crit,torch.device('cpu'))
    saved=copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),
                             rng=torch.get_rng_state(),cuda_rng=[]))
    expected=update(m,opt,sampler,crit,torch.device('cpu'))
    weights=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();sample_state=sampler.state()
    restore(saved,m,opt,sampler)
    assert update(m,opt,sampler,crit,torch.device('cpu'))==expected
    assert sampler.state()==sample_state and torch.equal(torch.get_rng_state(),rng)
    for key,value in weights.items():torch.testing.assert_close(value,m.state_dict()[key],atol=0,rtol=0)
    restore(saved,m,opt,sampler)
    assert update(m,opt,sampler,crit,torch.device('cpu'))==expected
    for key,value in weights.items():torch.testing.assert_close(value,m.state_dict()[key],atol=0,rtol=0)
