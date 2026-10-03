"""Current U58 GRU positive-cost contrast with identical sampling and decoding."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,hashlib,math,time,copy,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
import torch
from src.mentor_sequence_v2 import engine
from src.mentor_sequence_models import make_model,ContextSampler
from src.phase6_models import positive_weight
from src.score_context_study import read,write,sha,normalizer
from src.mentor_sequence_study import digest_array
from src.audit_external_stem_transfer import checked_raw
from src import run_halo_decoder_composition as dec
from score_only_export_training import data_u
from interval_quality import intervals,score
OUT=BASE/'gru_cost';ART=ROOT/'artifacts/mentor_gru_cost_20260916';REF=ROOT/'artifacts/mentor_score_origin_20260916';COLS=engine.COLS

def guard(new=False):
    b=read(BASE/'BUDGET.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(b['observed_at_utc'].replace('Z','+00:00'))).total_seconds()
    if age>1800 or b['observed_used_percent']>=b['stop_new_runs_used_percent' if new else 'absolute_stop_used_percent']:raise TimeoutError('Budget stop/stale reading')
    elapsed=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*_fold*.json'));assert elapsed<1200;return elapsed

def preflight(tr):
    for w in (10.,math.sqrt(10)):
        x=torch.tensor([-2.,0.,2.,-3.],requires_grad=True);y=torch.tensor([0.,1.,1.,1.]);mask=torch.tensor([1.,1.,1.,0.]);a=(torch.nn.functional.binary_cross_entropy_with_logits(x,y,pos_weight=torch.tensor(w),reduction='none')*mask).sum()/mask.sum();b=(-(w*y*torch.nn.functional.logsigmoid(x)+(1-y)*torch.nn.functional.logsigmoid(-x))*mask).sum()/mask.sum();torch.testing.assert_close(a,b);a.backward();assert x.grad[-1]==0 and torch.isfinite(x.grad).all()
    norm=normalizer(tr);sampler=ContextSampler(tr,norm,42,False);x,y,mask,valid=[v.cuda() for v in sampler.batch()];m=make_model('G64',42).cuda();assert sum(p.numel() for p in m.parameters())==6005;opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    def update():
        opt.zero_grad(set_to_none=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(x,~valid.bool()),y,pos_weight=torch.tensor(math.sqrt(positive_weight(tr,10)),device='cuda'),reduction='none')*mask).sum()/mask.sum();assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(g);opt.step();return float(loss.detach())
    update();weights=copy.deepcopy(m.state_dict());ostate=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all();expected=update();after=copy.deepcopy(m.state_dict());m.load_state_dict(weights);opt.load_state_dict(copy.deepcopy(ostate));torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng);assert update()==expected and all(torch.equal(v,m.state_dict()[k]) for k,v in after.items());write(OUT/'preflight.json',dict(status='passed',weighted_loss_formula=True,mask_gradient_zero=True,gpu_update_replay_exact=True,params=6005))

def interval_eval(name,val):
    p=ART/'decoder'/f'{name}_positions.csv.gz';f=pd.read_csv(p);groups={(pid,str(perf)):g.beat.to_numpy(int) for (pid,perf),g in f.groupby(['piece_id','performance_id'])};rows=[]
    for pid,v in val.items():
        mask=v['label_mask'].astype(bool);truth=intervals(np.flatnonzero((v['labels']>.5)&mask),mask)
        for perf in v['performance_ids']:
            pred=intervals(groups.get((pid,str(perf)),[]),mask)
            for tol in (0,1):rows.append(dict(piece_id=pid,performance_id=str(perf),tolerance=tol,**score(pred,truth,tol)))
    z=pd.DataFrame(rows);z.to_csv(ART/'decoder'/f'{name}_interval_perfs.csv',index=False);work=z.groupby(['piece_id','tolerance']).mean(numeric_only=True);work.to_csv(ART/'decoder'/f'{name}_interval_works.csv');return {f'interval_{k}_tol{tol}':float(work.xs(tol,level='tolerance')[k].mean()) for tol in (0,1) for k in ('f1','precision','recall')}

def main():
    began=time.monotonic();guard(True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for p in (OUT,OUT/'decoder_state',ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    parent=read(BASE/'score_origin_training/contract.json');frame,data=data_u();inputs={p:{k:digest_array(v[k]) for k in ('curves','labels','label_mask','pitch_profiles')} for p,v in data.items()};assert inputs==parent['inputs'];sources=dict(parent['sources'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',BASE/'score_only_export_training.py',BASE/'interval_quality.py'):sources[str(p)]=sha(p)
    for seed in (42,43):
        for fold in (0,1):
            run=f'G64_seed{seed}_fold{fold}'
            for p in (REF/'metrics'/f'{run}.json',REF/'metrics'/f'{run}_predictions.csv.gz',REF/'checkpoints'/run/'latest.pt',BASE/f'score_origin_training/prior_fold{fold}.json'):sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items());value=dict(sources=sources,inputs=inputs);contract=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    else:write(OUT/'contract.json',dict(contract=contract,**value))
    preflight({p:data[p] for p in frame[(frame.fold==0)&(frame.split=='train')].piece_id});engine.OUT=OUT;engine.ART=ART;engine.guard=guard;engine.make_model=make_model;engine.positive_weight=lambda d,c:math.sqrt(positive_weight(d,c))
    for seed in (42,43):
        for fold in (0,1):
            part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};engine.train_one('G64',fold,seed,tr,va,contract)
    dec.OUT=OUT/'decoder_state';dec.ART=ART/'decoder';rows=[];audits=[]
    for seed in (42,43):
        for fold in (0,1):
            part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};run=f'G64_seed{seed}_fold{fold}';a=read(ART/'metrics'/f'{run}.json');b=read(REF/'metrics'/f'{run}.json');cp=torch.load(ART/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False);ref=torch.load(REF/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False);assert cp['sampler']==ref['sampler'] and a['supervised_positions']==b['supervised_positions'];np.testing.assert_array_equal(cp['mean'],ref['mean']);np.testing.assert_array_equal(cp['std'],ref['std']);assert abs(a['pos_weight']**2-b['pos_weight'])<1e-10;audits.append(dict(seed=seed,fold=fold,sampler_exposure_normalization_equal=True,old_weight=b['pos_weight'],new_weight=a['pos_weight'],gpu_replay_error=a['checkpoint_replay_error']))
            prior=read(BASE/f'score_origin_training/prior_fold{fold}.json')['frozen_reference_prior']
            for arm,root,meta in [('U',REF,b),('W',ART,a)]:
                raw=checked_raw(pd.read_csv(root/'metrics'/f'{run}_predictions.csv.gz'),va)
                for policy,strength in [('raw',0.),('B10',1.)]:
                    guard();name=f'{arm}_{run}_{policy}';r=dec.evaluate(raw,va,meta['threshold'],prior,strength,name,contract)
                    if policy=='raw':assert all(abs(r[k]-meta[k])<1e-10 for k in COLS)
                    rows.append(dict(arm=arm,seed=seed,fold=fold,policy=policy,**r,**interval_eval(name,va)))
            raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),va);dec.evaluate(raw,va,b['threshold'],prior,1.,f'W_reference_threshold_{run}',contract)
    f=pd.DataFrame(rows);f.to_csv(OUT/'results.csv',index=False);icols=[k for k in f if k.startswith('interval_')];f.groupby(['arm','policy'])[COLS+icols].mean().to_csv(OUT/'means.csv');pd.DataFrame(audits).to_csv(OUT/'training_audit.csv',index=False);a=f.query("arm=='W' and policy=='B10'").set_index(['seed','fold']);b=f.query("arm=='U' and policy=='B10'").set_index(['seed','fold']);delta=a[COLS+icols]-b[COLS+icols];delta.to_csv(OUT/'paired_deltas.csv');d=delta.mean();positive=int((delta.macro_f1_tol1>0).sum());ipositive=int((delta.interval_f1_tol1>0).sum());write(OUT/'decision.json',dict(boundary_gate=bool(d.macro_f1_tol1>=.015 and positive>=3 and d.macro_f1_tol0>=0 and d.raw_ap>=0 and d.interval_f1_tol1>=0),interval_exploratory_gate=bool(d.interval_f1_tol1>=.02 and ipositive>=3 and d.macro_f1_tol1>=-.005),positive_cells=positive,interval_positive_cells=ipositive,deltas={k:float(v) for k,v in d.items()},independent_test=False));assert all(sha(p)==h for p,h in sources.items());write(OUT/'completion_audit.json',dict(status='complete',new_training=4,reference_reused=4,decoder_cells=20,sampling_exposure_norm_equal=True,checkpoint_replays=True,source_hashes_unchanged=True,old19_extra10_Op50_predictions_not_read=True,seconds=time.monotonic()-began));write(OUT/'SEQUENCE_STATE.json',dict(status='complete',pid=None));print(f.groupby(['arm','policy'])[COLS+icols].mean().to_string(),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid()));raise
