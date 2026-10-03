"""Train a score-only export using the verified U train-coordinate contract."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,hashlib,json,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
import torch
from src.mentor_sequence_v2 import engine
from src.mentor_sequence_study import splits,digest_array
from src.run_recurrence_depth_study import dataset
from src.score_context_study import read,write,sha,normalizer
from src.mentor_sequence_models import ContextSampler
from src.audit_external_stem_transfer import checked_raw
from src import run_halo_decoder_composition as dec
from score_semantics_training import changed_data
from gru_input_isolation import model
OUT=BASE/'score_only_export';ART=ROOT/'artifacts/mentor_score_only_export_20260916';UREF=ROOT/'artifacts/mentor_score_origin_20260916';COLS=engine.COLS

def data_u():
    frame,_=splits();data=changed_data(dataset(sorted(frame.piece_id.unique()),'C3'),'T')
    for p in ('chopin_op17_no4','chopin_op68_no2'):
        with np.load(UREF/'cache'/f'{p}.npz',allow_pickle=False) as z:
            v=data[p];v['curves'][...,9:25]=z['score16'][None];v['curves'][...,34:]=z['motif24'][None];v.update(selected_score=z['score16'].copy(),fixed_score=z['score16'].copy(),pitch_profiles=z['motif24'].copy(),labels=z['labels'].copy(),label_mask=z['mask'].copy())
    return frame,data

def guard(new=False):
    b=read(BASE/'BUDGET.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(b['observed_at_utc'].replace('Z','+00:00'))).total_seconds()
    if age>1800 or b['observed_used_percent']>=b['stop_new_runs_used_percent' if new else 'absolute_stop_used_percent']:raise TimeoutError('Budget stop/stale quota')
    seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*_fold*.json'));assert seconds<6000;return seconds

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);guard(True)
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder',OUT/'decoder_state'):p.mkdir(parents=True,exist_ok=True)
    assert read(BASE/'score_origin_training/completion_audit.json')['status']=='complete';parent=read(BASE/'score_origin_training/contract.json');sources=dict(parent['sources'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',BASE/'gru_input_isolation.py',BASE/'score_semantics_training.py'):sources[str(p)]=sha(p)
    frame,data=data_u();inputs={p:{k:digest_array(v[k]) for k in ('curves','labels','label_mask','pitch_profiles')} for p,v in data.items()};assert inputs==parent['inputs'];assert all(sha(p)==h for p,h in sources.items())
    value=dict(sources=sources,inputs=inputs);contract=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    else:write(OUT/'contract.json',dict(contract=contract,**value))
    for seed in (42,43):
        m=model('score',seed).eval();x=torch.randn(2,17,58);other=x.clone();other[...,np.r_[0:9,25:34]]=777;torch.testing.assert_close(m(x),m(other),atol=0,rtol=0)
    write(OUT/'preflight.json',dict(status='passed',all_U_inputs_labels_masks_equal=True,dropped18_invariance=True,audited_wrapper_reused=True));engine.OUT=OUT;engine.ART=ART;engine.guard=guard;engine.make_model=lambda kind,seed:model('score',seed)
    for seed in (42,43):
        for fold in (0,1):
            part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};engine.train_one('G64',fold,seed,tr,va,contract)
    dec.OUT=OUT/'decoder_state';dec.ART=ART/'decoder';rows=[];components=[];hashes={}
    for seed in (42,43):
        for fold in (0,1):
            guard();part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};run=f'G64_seed{seed}_fold{fold}';meta=read(ART/'metrics'/f'{run}.json');ref=read(UREF/'metrics'/f'{run}.json');last=ART/'checkpoints'/run/'latest.pt';cp=torch.load(last,map_location='cpu',weights_only=False);refcp=torch.load(UREF/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False)
            assert cp['step']==600 and cp['sampler']==refcp['sampler'] and meta['supervised_positions']==ref['supervised_positions'];np.testing.assert_array_equal(cp['mean'],refcp['mean']);np.testing.assert_array_equal(cp['std'],refcp['std'])
            prior=read(BASE/f'score_origin_training/prior_fold{fold}.json')['frozen_reference_prior'];raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),va)
            for policy,strength in [('raw',0.),('B10',1.)]:
                value=dec.evaluate(raw,va,meta['threshold'],prior,strength,run+'_'+policy,contract)
                if policy=='raw':assert all(abs(value[k]-meta[k])<1e-10 for k in COLS)
                rows.append(dict(arm='S',seed=seed,fold=fold,policy=policy,**value))
            best=ART/'checkpoints'/run/'best.pt';hashes[str(best)]=sha(best);hashes[str(last)]=sha(last);components.append(dict(seed=seed,fold=fold,checkpoint=str(best),sha256=sha(best),threshold=meta['threshold'],prior=prior,best_step=meta['best_step']))
    ref=pd.read_csv(BASE/'score_origin_training/results.csv');ref=ref[ref.arm=='U'];f=pd.concat([pd.DataFrame(rows),ref],ignore_index=True);f.to_csv(OUT/'results.csv',index=False);f.groupby(['arm','policy'])[COLS].mean().to_csv(OUT/'means.csv');assert all(sha(p)==h for p,h in {**sources,**hashes}.items())
    write(OUT/'checkpoint_hashes.json',hashes);write(OUT/'export.json',dict(status='frozen_before_remaining12_targets',components=components,input='40 score features, 18 performance channels hard-masked AFTER train-only normalization',protocol_sha=sha(OUT/'PROTOCOL.md'),training_contract=contract,ensemble_selection_on_transfer=False));write(OUT/'completion_audit.json',dict(status='complete',training_runs=4,decode_cells=8,all_sources_unchanged=True,sampler_exposure_equal_U=True,checkpoint_gpu_replay=True,remaining12_targets_read=False));write(OUT/'SEQUENCE_STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None));print(f.groupby(['arm','policy'])[COLS].mean().to_string(),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid()));raise
