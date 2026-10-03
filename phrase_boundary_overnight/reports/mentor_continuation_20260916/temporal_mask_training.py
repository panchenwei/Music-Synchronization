"""Fixed two-arm temporal masking experiment, original U58 reference reused."""
import os,sys,time,copy,json,hashlib,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
import torch
from src.mentor_sequence_v2 import engine
from src.mentor_sequence_models import make_model,ContextSampler
from src.score_context_study import read,write,sha,normalizer
from src.mentor_sequence_study import digest_array
from src.audit_external_stem_transfer import checked_raw
from src import run_halo_decoder_composition as dec
from score_only_export_training import data_u
from temporal_mask_sampler import MaskSampler,base_state
import gru_cost_study as cost
OUT=BASE/'temporal_masking';ART=ROOT/'artifacts/mentor_temporal_masking_20260916';REF=ROOT/'artifacts/mentor_score_origin_20260916';COLS=engine.COLS


def guard(new=False):
    b=read(BASE/'BUDGET.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(b['observed_at_utc'].replace('Z','+00:00'))).total_seconds()
    if age>1800 or b['observed_used_percent']>=b['stop_new_runs_used_percent' if new else 'absolute_stop_used_percent']:raise TimeoutError('Budget stop/stale quota')
    elapsed=sum(read(p)['seconds'] for p in ART.glob('*/metrics/*_fold*.json'))
    assert elapsed<1200
    return elapsed


def preflight(tr):
    norm=normalizer(tr);samplers={a:MaskSampler(tr,norm,42,False,arm=a) for a in ('D','B')};ref=ContextSampler(tr,norm,42,False);different=0
    before={p:digest_array(v['curves']) for p,v in tr.items()}
    for _ in range(25):
        z=ref.batch();batches={a:s.batch() for a,s in samplers.items()}
        for arm,s in samplers.items():
            x=batches[arm];assert base_state(s.state())==ref.state() and s.last_selection==ref.last_selection
            for i in (1,2,3):torch.testing.assert_close(x[i],z[i],atol=0,rtol=0)
            torch.testing.assert_close(x[0][~s.last_mask],z[0][~s.last_mask],atol=0,rtol=0)
            assert not x[0][s.last_mask].any() and not (s.last_mask&~x[3].bool()).any()
        a,b=samplers.values();assert torch.equal(a.last_mask.sum(1),b.last_mask.sum(1));different+=int((a.last_mask!=b.last_mask).any())
        assert a.state()['augmentation_rng']==b.state()['augmentation_rng']
    assert different and before=={p:digest_array(v['curves']) for p,v in tr.items()}
    s=samplers['B'];state=copy.deepcopy(s.state());expected=s.batch();s.load_state(state)
    for x,y in zip(expected,s.batch()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    m=make_model('G64',42).cuda();assert sum(p.numel() for p in m.parameters())==6005;opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    def update():
        x,y,mask,valid=[v.cuda() for v in s.batch()];opt.zero_grad(set_to_none=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(x,~valid.bool()),y,pos_weight=torch.tensor(10.,device='cuda'),reduction='none')*mask).sum()/mask.sum();assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(g);opt.step();return float(loss.detach())
    update();weights=copy.deepcopy(m.state_dict());ostate=copy.deepcopy(opt.state_dict());ss=copy.deepcopy(s.state());rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all();expected=update();after=copy.deepcopy(m.state_dict());m.load_state_dict(weights);opt.load_state_dict(copy.deepcopy(ostate));s.load_state(ss);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng);assert update()==expected and all(torch.equal(v,m.state_dict()[k]) for k,v in after.items())
    # Same unchanged architecture can fit a fixed tiny batch, no stochastic masking at evaluation.
    tiny=make_model('G64',42).eval();torch.manual_seed(7);x=torch.randn(2,8,58);y=torch.zeros(2,8);y[:,[2,5]]=1;opt=torch.optim.Adam(tiny.parameters(),lr=.02)
    for _ in range(100):opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(tiny(x),y);loss.backward();opt.step()
    assert torch.equal(tiny(x)>0,y.bool())
    write(OUT/'preflight.json',dict(status='passed',same_base_sampling=True,equal_mask_counts=True,unmasked_inputs_labels_padding_unchanged=True,input_arrays_unchanged=True,independent_rng_exact_resume=True,gpu_full_optimizer_sampler_replay_exact=True,tiny_fit_exact=True,params=6005))


def main():
    began=time.monotonic();guard(True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for p in (OUT,OUT/'decoder_state',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    for arm in ('D','B'):
        for p in (OUT/arm,ART/arm/'metrics',ART/arm/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    frame,data=data_u();parent=read(BASE/'score_origin_training/contract.json');inputs={p:{k:digest_array(v[k]) for k in ('curves','labels','label_mask','pitch_profiles')} for p,v in data.items()};assert inputs==parent['inputs'];sources=dict(parent['sources'])
    for p in (Path(__file__),BASE/'temporal_mask_sampler.py',OUT/'PROTOCOL.md',BASE/'score_only_export_training.py',BASE/'gru_cost_study.py',BASE/'interval_quality.py'):sources[str(p)]=sha(p)
    for seed in (42,43):
        for fold in (0,1):
            run=f'G64_seed{seed}_fold{fold}'
            for p in (REF/'metrics'/f'{run}.json',REF/'metrics'/f'{run}_predictions.csv.gz',REF/'checkpoints'/run/'latest.pt',BASE/f'score_origin_training/prior_fold{fold}.json'):sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items());value=dict(sources=sources,inputs=inputs);contract=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    else:write(OUT/'contract.json',dict(contract=contract,**value))
    preflight({p:data[p] for p in frame[(frame.fold==0)&(frame.split=='train')].piece_id});engine.guard=guard;engine.make_model=make_model
    for arm in ('D','B'):
        engine.OUT=OUT/arm;engine.ART=ART/arm;engine.ContextSampler=lambda *args,**kwargs:MaskSampler(*args,**kwargs,arm=arm)
        for seed in (42,43):
            for fold in (0,1):
                part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};engine.train_one('G64',fold,seed,tr,va,contract)
        write(OUT/arm/'SEQUENCE_STATE.json',dict(status='complete',pid=None))
    dec.OUT=OUT/'decoder_state';dec.ART=ART/'decoder';cost.ART=ART;rows=[];audits=[]
    for seed in (42,43):
        for fold in (0,1):
            part=frame[frame.fold==fold];va={p:data[p] for p in part[part.split=='validation'].piece_id};run=f'G64_seed{seed}_fold{fold}';rcp=torch.load(REF/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False);rm=read(REF/'metrics'/f'{run}.json');prior=read(BASE/f'score_origin_training/prior_fold{fold}.json')['frozen_reference_prior']
            for arm,root in [('U',REF),('D',ART/'D'),('B',ART/'B')]:
                meta=read(root/'metrics'/f'{run}.json');cp=torch.load(root/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False);assert base_state(cp['sampler'])==rcp['sampler'] and meta['supervised_positions']==rm['supervised_positions'] and meta['pos_weight']==rm['pos_weight'];np.testing.assert_array_equal(cp['mean'],rcp['mean']);np.testing.assert_array_equal(cp['std'],rcp['std']);audits.append(dict(arm=arm,seed=seed,fold=fold,sampling_exposure_norm_equal=True,gpu_replay_error=meta['checkpoint_replay_error'],train_f1=meta['train_at_dev_threshold']['macro_f1_tol1'],dev_raw_f1=meta['macro_f1_tol1']))
                raw=checked_raw(pd.read_csv(root/'metrics'/f'{run}_predictions.csv.gz'),va)
                for policy,strength in [('raw',0.),('B10',1.)]:
                    guard();name=f'{arm}_{run}_{policy}';r=dec.evaluate(raw,va,meta['threshold'],prior,strength,name,contract)
                    if policy=='raw':assert all(abs(r[k]-meta[k])<1e-10 for k in COLS)
                    rows.append(dict(arm=arm,seed=seed,fold=fold,policy=policy,**r,**cost.interval_eval(name,va)))
    f=pd.DataFrame(rows);icols=[k for k in f if k.startswith('interval_')];f.to_csv(OUT/'results.csv',index=False);means=f.groupby(['arm','policy'])[COLS+icols].mean();means.to_csv(OUT/'means.csv');pd.DataFrame(audits).to_csv(OUT/'training_audit.csv',index=False);decisions=[]
    for arm,control,gate in [('D','U',.015),('B','U',.015),('B','D',.005)]:
        a=f[(f.arm==arm)&(f.policy=='B10')].set_index(['seed','fold']);b=f[(f.arm==control)&(f.policy=='B10')].set_index(['seed','fold']);d=a[COLS+icols]-b[COLS+icols];d.to_csv(OUT/f'{arm}_minus_{control}.csv');v=d.mean();pos=int((d.macro_f1_tol1>0).sum());decisions.append(dict(arm=arm,control=control,positive_cells=pos,passed=bool(v.macro_f1_tol1>=gate and pos>=3 and v.macro_f1_tol0>=0 and v.raw_ap>=0 and v.interval_f1_tol1>=0),deltas={k:float(x) for k,x in v.items()}))
    candidates=[d['arm'] for d in decisions if d['control']=='U' and d['passed']];selected=max(candidates,key=lambda a:means.loc[(a,'B10'),'macro_f1_tol1']) if candidates else None
    write(OUT/'decision.json',dict(replication_candidate=selected,promote=False,replication_required=bool(selected),comparisons=decisions,independent_test=False));assert all(sha(p)==h for p,h in sources.items());write(OUT/'completion_audit.json',dict(status='complete',new_training=8,reference_reused=4,decode_cells=24,checkpoint_gpu_replays=True,sampling_labels_exposure_norm_equal=True,all_sources_unchanged=True,frozen_external_predictions_not_read=True,seconds=time.monotonic()-began));print(means.to_string(),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid()));raise
