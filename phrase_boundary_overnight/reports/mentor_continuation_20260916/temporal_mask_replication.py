"""New seeds and fixed ensemble confirmation for temporal input masking."""
import os,sys,time,json,hashlib,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
import torch
import temporal_mask_training as study
from temporal_mask_sampler import MaskSampler,base_state
from src.mentor_sequence_v2 import engine
from src.mentor_sequence_models import make_model
from src.score_context_study import read,write,sha
from src.mentor_sequence_study import digest_array
from src.audit_external_stem_transfer import checked_raw
from src.context_inference_audit_v2 import save_raw
from src import run_halo_decoder_composition as dec
from score_only_export_training import data_u
import gru_cost_study as cost
OUT=BASE/'temporal_mask_replication';ART=ROOT/'artifacts/mentor_temporal_mask_replication_20260916';OLDU=ROOT/'artifacts/mentor_score_origin_20260916';NEWU=ROOT/'artifacts/mentor_corrected_seed_ensemble_20260916';OLDB=ROOT/'artifacts/mentor_temporal_masking_20260916/B';COLS=engine.COLS


def source(arm,seed):return (OLDB if seed<44 else ART/'B') if arm=='B' else (OLDU if seed<44 else NEWU)


def main():
    began=time.monotonic();study.ART=ART;study.guard(True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert read(BASE/'temporal_masking/decision.json')['replication_candidate']=='B'
    for p in (OUT,OUT/'decoder_state',ART/'B/metrics',ART/'B/checkpoints',ART/'decoder',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    parent=read(BASE/'temporal_masking/contract.json');frame,data=data_u();inputs={p:{k:digest_array(v[k]) for k in ('curves','labels','label_mask','pitch_profiles')} for p,v in data.items()};assert inputs==parent['inputs'];sources=dict(parent['sources'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',BASE/'temporal_masking/decision.json',BASE/'corrected_seed_ensemble/export.json'):sources[str(p)]=sha(p)
    for arm in ('U','B'):
        for seed in (42,43,44,45):
            if arm=='B' and seed>=44:continue
            for fold in (0,1):
                run=f'G64_seed{seed}_fold{fold}'
                for p in (source(arm,seed)/'metrics'/f'{run}.json',source(arm,seed)/'metrics'/f'{run}_predictions.csv.gz',source(arm,seed)/'checkpoints'/run/'best.pt',source(arm,seed)/'checkpoints'/run/'latest.pt'):sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items());value=dict(sources=sources,inputs=inputs);contract=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    else:write(OUT/'contract.json',dict(contract=contract,**value))
    assert read(BASE/'temporal_masking/preflight.json')['status']=='passed';engine.OUT=OUT;engine.ART=ART/'B';engine.make_model=make_model;engine.guard=study.guard;engine.ContextSampler=lambda *args,**kwargs:MaskSampler(*args,**kwargs,arm='B')
    for seed in (44,45):
        for fold in (0,1):
            part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};engine.train_one('G64',fold,seed,tr,va,contract)
            assert sum(read(p)['seconds'] for p in (ART/'B/metrics').glob('*fold*.json'))<600
    dec.OUT=OUT/'decoder_state';dec.ART=ART/'decoder';cost.ART=ART;rows=[];works=[];audit=[];exports=[]
    def evaluate(raw,va,threshold,prior,name,arm,mode,seed,fold,policy):
        study.guard();r=dec.evaluate(raw,va,threshold,prior,1. if policy=='B10' else 0.,name,contract);rows.append(dict(arm=arm,mode=mode,seed=seed,fold=fold,policy=policy,**r,**cost.interval_eval(name,va)));w=pd.read_csv(ART/'decoder'/f'{name}_pieces.csv');z=pd.read_csv(ART/'decoder'/f'{name}_interval_works.csv').query('tolerance==1');merged=w.merge(z[['piece_id','f1','precision','recall']],on='piece_id');works.extend(dict(arm=arm,mode=mode,seed=seed,fold=fold,policy=policy,**v) for v in merged.to_dict('records'))
    for fold in (0,1):
        va={p:data[p] for p in frame[(frame.fold==fold)&(frame.split=='validation')].piece_id};prior=read(BASE/f'score_origin_training/prior_fold{fold}.json')['frozen_reference_prior']
        for arm in ('U','B'):
            raws=[];thresholds=[];components=[]
            for seed in (42,43,44,45):
                root=source(arm,seed);run=f'G64_seed{seed}_fold{fold}';meta=read(root/'metrics'/f'{run}.json');raw=checked_raw(pd.read_csv(root/'metrics'/f'{run}_predictions.csv.gz'),va);raws.append(raw);thresholds.append(meta['threshold']);cp=torch.load(root/'checkpoints'/run/'latest.pt',weights_only=False,map_location='cpu');ref=torch.load(source('U',seed)/'checkpoints'/run/'latest.pt',weights_only=False,map_location='cpu');assert cp['step']==600 and base_state(cp['sampler'])==ref['sampler'];np.testing.assert_array_equal(cp['mean'],ref['mean']);np.testing.assert_array_equal(cp['std'],ref['std']);assert meta['supervised_positions']==read(source('U',seed)/'metrics'/f'{run}.json')['supervised_positions']
                if arm=='B' and seed>=44:audit.append(dict(seed=seed,fold=fold,normalizer_sampler_exposure_equal=True,gpu_replay_error=meta['checkpoint_replay_error']))
                p=root/'checkpoints'/run/'best.pt';components.append(dict(seed=seed,checkpoint=str(p),sha256=sha(p),threshold=meta['threshold']))
                for policy in ('raw','B10'):evaluate(raw,va,meta['threshold'],prior,f'{arm}_{run}_{policy}',arm,'single',seed,fold,policy)
            mixed={p:{k:np.stack([r[p][k] for r in raws]).mean(0) for k in raws[0][p]} for p in va};threshold=float(np.mean(thresholds));pp=ART/'metrics'/f'{arm}_ensemble_fold{fold}.csv.gz';save_raw(mixed,pp,va);stored=checked_raw(pd.read_csv(pp),va)
            for policy in ('raw','B10'):evaluate(stored,va,threshold,prior,f'{arm}_ensemble_fold{fold}_{policy}',arm,'ensemble',-1,fold,policy)
            if arm=='B':exports.append(dict(fold=fold,components=components,threshold=threshold,prior=prior,input_version='U58',training_augmentation='4beat_contiguous_zero_normalized',inference_augmentation=False,independent_validation=False))
    f=pd.DataFrame(rows);f.to_csv(OUT/'results.csv',index=False);icols=[k for k in f if k.startswith('interval_')];means=f.groupby(['arm','mode','policy'])[COLS+icols].mean();means.to_csv(OUT/'means.csv');w=pd.DataFrame(works);w.to_csv(OUT/'work_metrics.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'training_audit.csv',index=False);np.testing.assert_allclose(means.loc[('U','ensemble','B10'),['macro_f1_tol1','interval_f1_tol1']],[.8148731767906621,.5913700914268428],atol=1e-12,rtol=0)
    z=f.query("mode=='single' and seed>=44 and policy=='B10'");nd=z[z.arm=='B'].set_index(['seed','fold'])[COLS+icols]-z[z.arm=='U'].set_index(['seed','fold'])[COLS+icols];nd.to_csv(OUT/'new_seed_deltas.csv');nv=nd.mean();npas=bool(nv.macro_f1_tol1>0 and (nd.macro_f1_tol1>0).sum()>=3 and nv.macro_f1_tol0>=0 and nv.raw_ap>=0 and nv.interval_f1_tol1>=0)
    sub=w.query("mode=='ensemble' and policy=='B10'");d=sub[sub.arm=='B'].set_index('piece_id')[['f1_tol1','f1_tol0','raw_ap','f1']]-sub[sub.arm=='U'].set_index('piece_id')[['f1_tol1','f1_tol0','raw_ap','f1']];d['opus']=d.index.str.extract(r'op(\d+)',expand=False);d.to_csv(OUT/'ensemble_work_deltas.csv');g=d.groupby('opus').mean();g.to_csv(OUT/'ensemble_opus_deltas.csv');v=d.mean(numeric_only=True);passed=bool(npas and v.f1_tol1>=.01 and (g.f1_tol1>0).sum()>=3 and v.f1_tol0>=0 and v.raw_ap>=0 and v.f1>=0)
    write(OUT/'decision.json',dict(new_seed_replication_passed=npas,new_seed_positive_cells=int((nd.macro_f1_tol1>0).sum()),new_seed_deltas=nv.to_dict(),ensemble_positive_opus=int((g.f1_tol1>0).sum()),ensemble_deltas=v.to_dict(),promote=passed,independent_test=False));write(OUT/'export.json',dict(folds=exports,contract=contract,candidate_only_not_promoted=not passed));assert all(sha(p)==h for p,h in sources.items());write(OUT/'completion_audit.json',dict(status='complete',new_training=4,decode_cells=40,U_ensemble_reference_exact=True,checkpoint_gpu_replays=True,normalizer_sampler_exposure_equal=True,sources_unchanged=True,frozen_external_predictions_not_read=True,seconds=time.monotonic()-began));write(OUT/'SEQUENCE_STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid()));raise
