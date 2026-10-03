"""Only 16 score statistics change; original model and targets remain frozen."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,hashlib,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.mentor_sequence_v2 import engine
from src.mentor_sequence_models import make_model,ContextSampler
from src.mentor_sequence_study import splits,digest_array
from src.run_recurrence_depth_study import dataset
from src.score_context_study import read,write,sha,normalizer
from src.audit_external_stem_transfer import checked_raw
from src.local_context_study import metrics,predictions
from src.interstart_decoder import fit_prior
from src import run_halo_decoder_composition as dec
BASE=Path(__file__).parent;OUT=BASE/'score_semantics_training';ART=ROOT/'artifacts/mentor_score_semantics_training_20260916';CACHE=ROOT/'artifacts/mentor_score_semantics_20260916/cache';OLD=ROOT/'artifacts/mentor_sequence_20260916_v2';COLS=engine.COLS

def guard(new=False):
    b=read(BASE/'BUDGET.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(b['observed_at_utc'].replace('Z','+00:00'))).total_seconds()
    if age>1800 or b['observed_used_percent']>=b['stop_new_runs_used_percent' if new else 'absolute_stop_used_percent']:raise TimeoutError('Budget stop or stale quota')
    seconds=sum(read(p)['seconds'] for p in ART.glob('*/metrics/*_fold*.json'));assert seconds<6000
    return seconds

def changed_data(data,arm):
    result={}
    for p,v in data.items():
        with np.load(CACHE/f'{p}.npz',allow_pickle=False) as z:score=z[arm].copy()
        new=dict(v);new['curves']=v['curves'].copy();new['curves'][...,9:25]=score[None];new['selected_score']=score;new['fixed_score']=score;result[p]=new
        np.testing.assert_array_equal(new['curves'][...,:9],v['curves'][...,:9]);np.testing.assert_array_equal(new['curves'][...,25:],v['curves'][...,25:])
    return result

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);guard(True)
    for p in (OUT,OUT/'decoder_state',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    for arm in ('L','T'):
        for p in (OUT/arm,ART/arm/'metrics',ART/arm/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    parent=read(BASE/'sequence_v2/sequence_contract.json');audit=read(BASE/'score_semantics_audit/contract.json');assert read(BASE/'score_semantics_audit/completion_audit.json')['status']=='complete'
    sources={**parent['sources'],**audit['sources'],**read(BASE/'score_semantics_audit/cache_hashes.json'),str(Path(__file__)):sha(Path(__file__)),str(OUT/'PROTOCOL.md'):sha(OUT/'PROTOCOL.md')}
    for seed in (42,43):
        for fold in (0,1):
            for p in [OLD/'metrics'/f'G64_seed{seed}_fold{fold}.json',OLD/'checkpoints'/f'G64_seed{seed}_fold{fold}'/'latest.pt']:sources[str(p)]=sha(p)
    p=BASE/'decoder_transfer/results.csv';sources[str(p)]=sha(p);assert all(sha(p)==h for p,h in sources.items())
    frame,_=splits();olddata=dataset(sorted(frame.piece_id.unique()),'C3');inputs={p:{k:digest_array(v[k]) for k in ('curves','labels','label_mask','performance_ids','pitch_profiles')} for p,v in olddata.items()};assert inputs==parent['inputs']
    arms={arm:changed_data(olddata,arm) for arm in ('L','T')};changed={arm:{p:digest_array(v['curves']) for p,v in data.items()} for arm,data in arms.items()}
    value=dict(sources=sources,original_inputs=inputs,new_curves=changed);contract=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    else:write(OUT/'contract.json',dict(contract=contract,**value))
    checks=[];outside=np.r_[0:9,25:58]
    for arm,data in arms.items():
        for fold in (0,1):
            part=frame[frame.fold==fold];ids=part[part.split=='train'].piece_id;tr={p:data[p] for p in ids};old={p:olddata[p] for p in ids};norm=normalizer(tr);ref=normalizer(old)
            np.testing.assert_array_equal(norm.mean[outside],ref.mean[outside]);np.testing.assert_array_equal(norm.std[outside],ref.std[outside])
            for seed in (42,43):
                a=ContextSampler(tr,norm,seed,False);b=ContextSampler(old,ref,seed,False);xa,ya,ma,va=a.batch();xb,yb,mb,vb=b.batch()
                for x,y in [(xa[...,outside],xb[...,outside]),(ya,yb),(ma,mb),(va,vb)]:torch.testing.assert_close(x,y,atol=0,rtol=0)
                assert a.last_selection==b.last_selection
            checks.append(dict(arm=arm,fold=fold,other42_normalizers_and_sampled_features_exact=True,labels_masks_sampler_identical=True))
    write(OUT/'preflight.json',dict(status='passed',original_audited_model_reused=True,rows=checks));engine.make_model=make_model;engine.guard=guard
    for arm,data in arms.items():
        engine.OUT=OUT/arm;engine.ART=ART/arm;write(OUT/'STATE.json',dict(status='running',arm=arm,pid=os.getpid()))
        for seed in (42,43):
            for fold in (0,1):
                part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};engine.train_one('G64',fold,seed,tr,va,contract)
    rows=[];diagnostics=[];cphashes={};dec.OUT=OUT/'decoder_state';dec.ART=ART/'decoder'
    for arm,data in arms.items():
        for fold in (0,1):
            part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};norm=normalizer(tr);prior=fit_prior(tr);assert prior==read(BASE/f'decoder_transfer/prior_fold{fold}.json')['prior']
            for seed in (42,43):
                guard();run=f'G64_seed{seed}_fold{fold}';meta=read(ART/arm/'metrics'/f'{run}.json');ref=read(OLD/'metrics'/f'{run}.json');cp=torch.load(ART/arm/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False);oldcp=torch.load(OLD/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False)
                assert cp['step']==600 and cp['sampler']==oldcp['sampler'] and meta['supervised_positions']==ref['supervised_positions'] and meta['params']==6005 and meta['pos_weight']==10
                np.testing.assert_array_equal(cp['mean'],norm.mean);np.testing.assert_array_equal(cp['std'],norm.std)
                for name in ('best.pt','latest.pt'):
                    p=ART/arm/'checkpoints'/run/name;cphashes[str(p)]=sha(p)
                raw=checked_raw(pd.read_csv(ART/arm/'metrics'/f'{run}_predictions.csv.gz'),va)
                for policy,strength in [('raw',0.),('B10',1.)]:
                    v=dec.evaluate(raw,va,meta['threshold'],prior,strength,arm+'_'+run+'_'+policy,contract);assert abs(v['raw_ap']-meta['raw_ap'])<1e-10
                    if policy=='raw':assert all(abs(v[c]-meta[c])<1e-10 for c in COLS)
                    rows.append(dict(arm=arm,seed=seed,fold=fold,policy=policy,**v))
                m=make_model('G64',seed).cuda();m.load_state_dict(cp['model']);endtr=metrics(predictions(m,tr,norm,torch.device('cuda')),tr,meta['threshold'])[2];endva=metrics(predictions(m,va,norm,torch.device('cuda')),va,meta['threshold'])[2]
                diagnostics.append(dict(arm=arm,seed=seed,fold=fold,best_step=meta['best_step'],best_train_f1=meta['train_at_dev_threshold']['macro_f1_tol1'],best_dev_f1=meta['macro_f1_tol1'],end_train_f1=endtr['macro_f1_tol1'],end_dev_f1=endva['macro_f1_tol1']))
                print('SEMANTIC_TRAIN_AUDITED',arm,run,flush=True)
    old=pd.read_csv(BASE/'decoder_transfer/results.csv');old=old[old.policy.isin(['raw','B10'])].assign(arm='old');f=pd.concat([pd.DataFrame(rows),old],ignore_index=True);f.to_csv(OUT/'results.csv',index=False);f.groupby(['arm','policy'])[COLS].mean().to_csv(OUT/'means.csv');pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False);comparisons=[]
    for arm,ref in [('L','old'),('T','old'),('T','L')]:
        for policy in ('raw','B10'):
            a=f[(f.arm==arm)&(f.policy==policy)].set_index(['seed','fold']);b=f[(f.arm==ref)&(f.policy==policy)].set_index(['seed','fold']);d=a[COLS]-b[COLS];v=d.mean();positive=int((d.macro_f1_tol1>0).sum());comparisons.append(dict(candidate=arm,reference=ref,policy=policy,positive_cells=positive,passed=bool(positive>=3 and v.macro_f1_tol1>=.015 and v.macro_f1_tol0>=0 and v.raw_ap>=-1e-12),**{c+'_delta':float(v[c]) for c in COLS}))
    write(OUT/'comparisons.json',comparisons);assert all(sha(p)==h for p,h in {**sources,**cphashes}.items());write(OUT/'checkpoint_hashes.json',cphashes)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,decoder_cells=16,source_hashes_unchanged=True,labels_masks_other42_columns_unchanged=True,samplers_and_exposures_equal=True,probability_gpu_checkpoint_replays=True,independent_test=False));write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None))

if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid()));raise
