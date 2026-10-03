"""Read-only SAM train-gap/directional probe versus frozen original probe."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from . import current_sharpness_probe as old
from .score_context_study import read
from .local_context_study import predictions,metrics
from .three_round_round2 import checkpoint_threshold

ROOT=old.ROOT;OUT=ROOT/'reports/sam_study';ART=ROOT/'artifacts/sam_study'


def main():
    # All training must have finished; this read-only probe may overlap the
    # separate decoder audit because neither process edits checkpoints.
    assert all((ART/'metrics'/f'R_seed{s}_fold{f}.json').exists() for f in (0,1) for s in (42,43))
    torch.set_num_threads(2);rows=[];gaps=[];hashes={}
    oldaudit=read(old.OUT/'audit.json');assert all(old.sha(p)==h for p,h in oldaudit['hashes'].items())
    for fold in (0,1):
        ids=old.split_ids(fold);tr=old.source.dataset(ids['train'],'C3');va=old.source.dataset(ids['validation'],'C3');norm=old.normalizer(tr)
        for seed in (42,43):
            for point in ('best','latest'):
                path=ART/'checkpoints'/f'R_seed{seed}_fold{fold}'/f'{point}.pt';hashes[str(path)]=old.sha(path)
                cp=torch.load(path,map_location='cpu',weights_only=False)
                np.testing.assert_array_equal(cp['mean'],norm.mean);np.testing.assert_array_equal(cp['std'],norm.std)
                model=old.source.make_model('C3',seed).cuda().eval();model.load_state_dict(cp['model'])
                th=checkpoint_threshold(cp)
                a=metrics(predictions(model,tr,norm,torch.device('cuda')),tr,th)[2]
                b=metrics(predictions(model,va,norm,torch.device('cuda')),va,th)[2]
                expected=next(h for h in cp['history'] if h['step']==cp['step'])
                assert abs(b['macro_f1_tol1']-expected['macro_f1_tol1'])<1e-10
                gaps.append(dict(fold=fold,seed=seed,point=point,step=cp['step'],train_f1=a['macro_f1_tol1'],dev_f1=b['macro_f1_tol1']))
                sampler=old.CurvePieceBalancedSampler(tr,norm,64,32,915);params=list(model.parameters())
                crit=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(old.positive_weight(tr,10),device='cuda'))
                for batch in range(8):
                    x,y,mask,valid=(a.cuda() for a in sampler.batch())
                    def loss():return (crit(model(x,padding_mask=~valid.bool()),y)*mask).sum()/mask.sum().clamp_min(1)
                    l0=loss();grads=torch.autograd.grad(l0,params);gn=torch.stack([g.square().sum() for g in grads]).sum().sqrt();originals=[p.detach().clone() for p in params]
                    for radius in (.01,.05):
                        losses=[]
                        try:
                            for sign in (1.,-1.):
                                with torch.no_grad():
                                    for p,v,g in zip(params,originals,grads):p.copy_(v+sign*radius*g/(gn+1e-12))
                                    losses.append(float(loss()))
                        finally:
                            with torch.no_grad():
                                for p,v in zip(params,originals):p.copy_(v)
                        rows.append(dict(kind='SAM',fold=fold,seed=seed,point=point,step=cp['step'],batch=batch,radius=radius,base_loss=float(l0.detach()),plus_loss=losses[0],increase=losses[0]-float(l0.detach()),symmetric_curvature=(losses[0]+losses[1]-2*float(l0.detach()))/radius**2,grad_norm=float(gn)))
                assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in cp['model'].items())
    df=pd.concat([pd.DataFrame(rows),pd.read_csv(old.OUT/'probe.csv').assign(kind='C3')],ignore_index=True)
    df.to_csv(OUT/'sharpness_probe.csv',index=False)
    means=df.groupby(['kind','point','radius'])[['base_loss','plus_loss','increase','symmetric_curvature','grad_norm']].mean();means.to_csv(OUT/'sharpness_means.csv')
    pd.DataFrame(gaps).to_csv(OUT/'train_gap.csv',index=False)
    assert all(old.sha(p)==h for p,h in hashes.items())
    old.write(OUT/'diagnostics_audit.json',dict(status='complete',new_models=8,new_train_probe_batches=64,source_baseline_probe_verified=True,checkpoints_unchanged=True,dropout_disabled=True,diagnostic_only=True,hashes=hashes))
    print(means.to_string())


if __name__=='__main__':main()
