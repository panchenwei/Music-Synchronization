"""New, source-metadata-derived coordinate contract for two train works only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json,hashlib,traceback
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
import torch
from src.mentor_sequence_v2 import engine
from src.mentor_sequence_models import make_model,ContextSampler
from src.mentor_sequence_study import splits,digest_array
from src.run_recurrence_depth_study import dataset
from src.score_context_study import read,write,sha,normalizer
from src.slice_energy_study import DCML
from src.data import discover_dcml_pieces
from src.score_local_coordinates import local_events,local_labels,positioned_rows
from src.note_relation_graph import merged_events
from src.slice_energy_features import start_labels
from src.motif_recurrence_features import features
from src.interstart_decoder import fit_prior
from src.audit_external_stem_transfer import checked_raw
from src import run_halo_decoder_composition as dec
from score_semantics_audit import cues_from_events
from score_semantics_training import changed_data
OUT=BASE/'score_origin_training';ART=ROOT/'artifacts/mentor_score_origin_20260916';REF=ROOT/'artifacts/mentor_score_semantics_training_20260916/T';COLS=engine.COLS
IDS=['chopin_op17_no4','chopin_op68_no2']

def guard(new=False):
    b=read(BASE/'BUDGET.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(b['observed_at_utc'].replace('Z','+00:00'))).total_seconds()
    if age>1800 or b['observed_used_percent']>=b['stop_new_runs_used_percent' if new else 'absolute_stop_used_percent']:raise TimeoutError('Budget stop or stale quota')
    seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*_fold*.json'));assert seconds<6000;return seconds

def build(old,sources):
    data={p:dict(v) for p,v in old.items()};pieces=discover_dcml_pieces(DCML);origins=pd.read_csv(BASE/'score_only_compatibility/grid_bridge/all43.csv').set_index('piece_id');changes=[]
    for pid in IDS:
        o=origins.loc[pid];assert o.passed;offset=float(Fraction(str(o.origin_qb)));assert offset>0;v=old[pid];n=len(v['labels']);piece=pieces[pid]
        for p in (piece.notes_path,piece.measures_path,piece.harmony_path):sources[str(p)]=sha(p)
        notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t');harm=pd.read_csv(piece.harmony_path,sep='\t')
        yy,mm,*_=local_labels(harm,measures,n);np.testing.assert_array_equal(yy,v['labels']);np.testing.assert_array_equal(mm,v['label_mask'])
        events,ties=local_events(notes,measures,n);merged,_=merged_events(events,ties);shifted=[(e[0]-offset,*e[1:]) for e in merged if e[0]-offset<n and e[0]+e[1]-offset>0]
        bn=v['beat_number'];mn=v['measure_number'];score=cues_from_events(piece,bn,mn,shifted);motif=features(shifted,n,True)
        selected=harm[harm.phraseend.fillna('').astype(str).str.contains('{',regex=False)];rows,_,_=positioned_rows(selected,measures,n);targets=[q-offset for q,*_ in rows];labels,mask,mapping=start_labels(targets,n)
        w=data[pid];w['curves']=v['curves'].copy();w['curves'][...,9:25]=score[None];w['curves'][...,34:]=motif[None];w.update(selected_score=score,fixed_score=score,pitch_profiles=motif,labels=labels,label_mask=mask)
        outside=np.r_[0:9,25:34];np.testing.assert_array_equal(w['curves'][...,outside],v['curves'][...,outside]);assert np.isfinite(w['curves']).all()
        dest=ART/'cache'/f'{pid}.npz';values=dict(score16=score,motif24=motif,labels=labels,mask=mask,events=np.asarray(shifted),origin=np.array(offset))
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                for k,x in values.items():np.testing.assert_array_equal(z[k],x)
        else:np.savez_compressed(dest,**values)
        sources[str(dest)]=sha(dest);pd.DataFrame(mapping).to_csv(OUT/f'{pid}_target_mapping.csv',index=False)
        changes.append(dict(piece_id=pid,origin_qb=offset,label_changed_beats=int((labels!=yy).sum()),mask_changed_beats=int((mask!=mm).sum()),old_positive=int((yy*mm).sum()),new_positive=int((labels*mask).sum()),score16_changed_beats=int(np.any(abs(score-v['selected_score'])>1e-6,axis=1).sum()),motif_changed_beats=int(np.any(abs(motif-v['pitch_profiles'])>1e-6,axis=1).sum())))
    pd.DataFrame(changes).to_csv(OUT/'input_changes.csv',index=False);return data

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);guard(True)
    for p in (OUT,OUT/'decoder_state',ART/'cache',ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(BASE/'score_semantics_training/completion_audit.json')['status']=='complete';parent=read(BASE/'score_semantics_training/contract.json');sources=dict(parent['sources'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',BASE/'score_grid_bridge_audit.py',BASE/'score_only_compatibility/grid_bridge/source_hashes.json',BASE/'score_semantics_training.py'):sources[str(p)]=sha(p)
    sources.update(read(BASE/'score_only_compatibility/grid_bridge/source_hashes.json'))
    frame,_=splits();old=changed_data(dataset(sorted(frame.piece_id.unique()),'C3'),'T');data=build(old,sources)
    assert all(sha(p)==h for p,h in sources.items());value=dict(sources=sources,inputs={p:{k:digest_array(v[k]) for k in ('curves','labels','label_mask','pitch_profiles')} for p,v in data.items()});contract=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    else:write(OUT/'contract.json',dict(contract=contract,**value))
    checks=[]
    for fold in (0,1):
        part=frame[frame.fold==fold];ids=part[part.split=='train'].piece_id;vids=part[part.split=='validation'].piece_id;assert set(IDS)<=set(ids) and not(set(IDS)&set(vids))
        for pid in set(data)-set(IDS):
            for k in ('curves','labels','label_mask','pitch_profiles'):np.testing.assert_array_equal(data[pid][k],old[pid][k])
        tr={p:data[p] for p in ids};ref={p:old[p] for p in ids};a=normalizer(tr);b=normalizer(ref);outside=np.r_[0:9,25:34];np.testing.assert_array_equal(a.mean[outside],b.mean[outside]);np.testing.assert_array_equal(a.std[outside],b.std[outside])
        for seed in (42,43):
            sa=ContextSampler(tr,a,seed,False);sb=ContextSampler(ref,b,seed,False)
            for _ in range(25):
                xa,ya,ma,va=sa.batch();xb,yb,mb,vb=sb.batch();assert sa.last_selection==sb.last_selection;torch.testing.assert_close(xa[...,outside],xb[...,outside],atol=0,rtol=0);torch.testing.assert_close(va,vb,atol=0,rtol=0)
        prior=fit_prior(ref);write(OUT/f'prior_fold{fold}.json',dict(frozen_reference_prior=prior,new_training_prior=fit_prior(tr),new_prior_evaluated=False));checks.append(dict(fold=fold,validation_and_22_works_exact=True,performance18_and_norm_exact=True,sampler_selections_exact=True))
    write(OUT/'preflight.json',dict(status='passed',checks=checks));engine.OUT=OUT;engine.ART=ART;engine.guard=guard;engine.make_model=make_model
    for seed in (42,43):
        for fold in (0,1):
            part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};engine.train_one('G64',fold,seed,tr,va,contract)
    rows=[];audits=[];dec.OUT=OUT/'decoder_state';dec.ART=ART/'decoder';hashes={}
    for seed in (42,43):
        for fold in (0,1):
            guard();part=frame[frame.fold==fold];tr={p:data[p] for p in part[part.split=='train'].piece_id};va={p:data[p] for p in part[part.split=='validation'].piece_id};run=f'G64_seed{seed}_fold{fold}';meta=read(ART/'metrics'/f'{run}.json');ref=read(REF/'metrics'/f'{run}.json');cp=torch.load(ART/'checkpoints'/run/'latest.pt',weights_only=False,map_location='cpu');refcp=torch.load(REF/'checkpoints'/run/'latest.pt',weights_only=False,map_location='cpu');assert cp['step']==600 and cp['sampler']==refcp['sampler'] and meta['params']==6005
            norm=normalizer(tr);np.testing.assert_array_equal(cp['mean'],norm.mean);np.testing.assert_array_equal(cp['std'],norm.std);raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),va);prior=read(OUT/f'prior_fold{fold}.json')['frozen_reference_prior']
            for policy,strength in [('raw',0.),('B10',1.)]:
                result=dec.evaluate(raw,va,meta['threshold'],prior,strength,run+'_'+policy,contract)
                if policy=='raw':assert all(abs(result[k]-meta[k])<1e-10 for k in COLS)
                rows.append(dict(arm='U',seed=seed,fold=fold,policy=policy,**result))
            audits.append(dict(seed=seed,fold=fold,old_exposure=ref['supervised_positions'],new_exposure=meta['supervised_positions'],old_pos_weight=ref['pos_weight'],new_pos_weight=meta['pos_weight'],best_step=meta['best_step'],gpu_replay_error=meta['checkpoint_replay_error']))
            for name in ('best.pt','latest.pt'):
                p=ART/'checkpoints'/run/name;hashes[str(p)]=sha(p)
    reference=pd.read_csv(BASE/'score_semantics_training/results.csv');reference=reference[reference.arm=='T'];f=pd.concat([pd.DataFrame(rows),reference],ignore_index=True);f.to_csv(OUT/'results.csv',index=False);f.groupby(['arm','policy'])[COLS].mean().to_csv(OUT/'means.csv');pd.DataFrame(audits).to_csv(OUT/'audit.csv',index=False);comparisons=[]
    for policy in ('raw','B10'):
        a=f[(f.arm=='U')&(f.policy==policy)].set_index(['seed','fold']);b=f[(f.arm=='T')&(f.policy==policy)].set_index(['seed','fold']);d=a[COLS]-b[COLS];v=d.mean();pos=int((d.macro_f1_tol1>0).sum());comparisons.append(dict(policy=policy,positive_cells=pos,passed=bool(pos>=3 and v.macro_f1_tol1>=.015 and v.macro_f1_tol0>=0 and v.raw_ap>=0),**{k+'_delta':float(v[k]) for k in COLS}))
    assert all(sha(p)==h for p,h in {**sources,**hashes}.items());write(OUT/'checkpoint_hashes.json',hashes);write(OUT/'comparisons.json',comparisons);write(OUT/'completion_audit.json',dict(status='complete',training_runs=4,decode_cells=8,old_sources_and_cache_unchanged=True,validation_unchanged=True,train_origin_selected_from_metadata_not_f1=True,probabilities_gpu_replayed=True,independent_test=False));write(OUT/'SEQUENCE_STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None));print(f.groupby(['arm','policy'])[COLS].mean().to_string(),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid()));raise
