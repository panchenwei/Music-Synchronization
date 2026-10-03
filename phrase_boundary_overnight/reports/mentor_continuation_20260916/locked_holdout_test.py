"""Source-gated historical holdout evaluation, frozen model, no fitting."""
import os,sys,time,json,hashlib,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
import torch
from src.score_context_study import read,write,sha
from src.mentor_sequence_study import splits
from src.data import discover_dcml_pieces
from src.slice_energy_study import DCML
from src.score_local_coordinates import positioned_rows
from src.slice_energy_features import start_labels
from src.mentor_sequence_models import make_model
from src.models import Normalizer
from src.local_context_study import predictions,max_match_count
from src.context_inference_audit_v2 import save_raw
from src.audit_external_stem_transfer import checked_raw
from src import run_halo_decoder_composition as dec
from score_repeat_bridge import curve_export
from score_grid_bridge_audit import bridge
from score_only_features_export import extract
from score_only_export_training import data_u
import gru_cost_study as cost
OUT=BASE/'locked_holdout_test';ART=ROOT/'artifacts/locked_holdout_test_20260916'
EXPORTS={'U':BASE/'corrected_seed_ensemble/export.json','B':BASE/'temporal_mask_replication/export.json'}


def fingerprint(notes):
    f=pd.read_csv(notes,sep='\t');cols=['mc','mc_onset','duration_qb','midi','staff','voice'];return hashlib.sha256(f[cols].fillna('').astype(str).sort_values(cols).to_csv(index=False).encode()).hexdigest()


def features(pid,row,piece):
    beat,c,e,ids=curve_export(row);grid,meta=bridge(piece,beat)
    if not meta['passed']:return None,meta
    x,details=extract(piece,len(beat),float(Fraction(meta['origin_qb'])));np.testing.assert_allclose(x['beat_number'],beat.beat_number.to_numpy(),atol=1e-8,rtol=0);curves=np.concatenate([c,np.broadcast_to(x['score16'],(*c.shape[:2],16)),e,np.broadcast_to(x['motif24'],(*c.shape[:2],24))],axis=-1).astype(np.float32);assert np.isfinite(curves).all();assert curves.shape[-1]==58
    return dict(curves=curves,performance_ids=ids,pitch_profiles=x['motif24']),meta


def prepare():
    began=time.monotonic();OUT.mkdir(exist_ok=True);(ART/'cache').mkdir(parents=True,exist_ok=True);assert not (OUT/'contract.json').exists(),'Frozen preparation already exists';frame,ids=splits();training=set(frame.piece_id);assert len(ids)==19 and not training.intersection(ids);assert not set(frame.opus.astype(str)).intersection({p.split('_op')[1].split('_')[0] for p in ids})
    pieces=discover_dcml_pieces(DCML);manifest=pd.read_csv(ROOT/'manifests/piece_manifest.csv').set_index('piece_id');sources={str(p):sha(p) for p in (Path(__file__),OUT/'PROTOCOL.md',BASE/'score_grid_bridge_audit.py',BASE/'score_only_features_export.py',BASE/'score_repeat_bridge.py',BASE/'score_semantics_audit.py',BASE/'interval_quality.py',BASE/'gru_cost_study.py',ROOT/'manifests/piece_manifest.csv')}
    sources.update({str(p):sha(p) for p in (ROOT/'src').glob('*.py')})
    for path in EXPORTS.values():
        sources[str(path)]=sha(path)
        for fold in read(path)['folds']:
            for c in fold['components']:assert sha(c['checkpoint'])==c['sha256'];sources[c['checkpoint']]=c['sha256']
    # Already verified input interface, check anchor parity without requiring its broader geometry to pass the stricter test gate.
    from infer_corrected_candidate import build_features
    sources[str(BASE/'infer_corrected_candidate.py')]=sha(BASE/'infer_corrected_candidate.py');_,known=data_u();parity=[]
    for pid,v in known.items():
        item,_,_=build_features(pid);np.testing.assert_array_equal(item['curves'],v['curves']);parity.append(dict(piece_id=pid,all58_exact=True));print('PARITY',pid,flush=True)
    pd.DataFrame(parity).to_csv(OUT/'known24_parity.csv',index=False)
    train_files={sha(p) for pid in training for p in (pieces[pid].notes_path,pieces[pid].measures_path,pieces[pid].harmony_path)};train_fp={fingerprint(pieces[pid].notes_path) for pid in training};inventory=[];kept={};meta_by_id={}
    for pid in ids:
        assert time.monotonic()-began<600;piece=pieces[pid];r=manifest.loc[pid]
        for p in (Path(r.beat_time_path),Path(r.beat_dyn_path),piece.notes_path,piece.measures_path,piece.harmony_path):sources[str(p)]=sha(p)
        try:
            assert fingerprint(piece.notes_path) not in train_fp,'duplicate note sequence';assert not any(sha(p) in train_files for p in (piece.notes_path,piece.measures_path,piece.harmony_path)),'duplicate source file';item,meta=features(pid,r,piece);ok=item is not None;reason='' if ok else 'source_geometry_gate_failed';meta_by_id[pid]=meta
        except Exception as exc:ok=False;reason=repr(exc);meta={}
        inventory.append(dict(piece_id=pid,reason=reason,**{**meta,'passed':ok}));print('SOURCE_GATE',pid,ok,flush=True)
        if ok:kept[pid]=item
    pd.DataFrame(inventory).to_csv(OUT/'admission_all19.csv',index=False);assert len(kept)>=2
    # Cohort fixed before reading targets or model inference. Do not remove any admitted work based on labels/scores.
    write(OUT/'cohort_locked.json',dict(admitted=sorted(kept),excluded=[r['piece_id'] for r in inventory if not r['passed']],historical_holdout_not_blind=True,primary='B_fold0_B10',features_derived_without_harmonies=True))
    label_rows=[];coverage=[]
    for pid,item in kept.items():
        n=item['curves'].shape[1];piece=pieces[pid];h=pd.read_csv(piece.harmony_path,sep='\t');m=pd.read_csv(piece.measures_path,sep='\t');selected=h[h.phraseend.fillna('').astype(str).str.contains('{',regex=False)];positions,_,_=positioned_rows(selected,m,n);origin=float(Fraction(meta_by_id[pid]['origin_qb']));targets=[q-origin for q,*_ in positions];y,mask,mapping=start_labels(targets,n);item.update(labels=y,label_mask=mask)
        for row in mapping:label_rows.append(dict(piece_id=pid,**row))
        assert mask.sum()>0 and (y*mask).sum()>0,'Admitted work has no scored boundaries; stop entire evaluation'
        coverage.append(dict(piece_id=pid,opus=pid.split('_op')[1].split('_')[0],performances=len(item['performance_ids']),beats=n,scored_beats=int(mask.sum()),source_starts=len(set(targets)),source_noninitial_in_range=sum(not r['first_start'] for r in mapping),scored_starts=int((y*mask).sum()),ambiguous_noninitial=sum(r['status']=='ambiguous_tie' and not r['first_start'] for r in mapping),origin_qb=origin))
        path=ART/'cache'/f'{pid}.npz';assert not path.exists();np.savez_compressed(path,**item);sources[str(path)]=sha(path)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False);pd.DataFrame(label_rows).to_csv(OUT/'label_mapping.csv',index=False)
    for p in (OUT/'admission_all19.csv',OUT/'cohort_locked.json',OUT/'coverage.csv',OUT/'label_mapping.csv'):sources[str(p)]=sha(p)
    assert all(sha(p)==s for p,s in sources.items());contract=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest();write(OUT/'contract.json',dict(contract=contract,sources=sources,primary='B_fold0_B10',threshold_search=False));write(OUT/'preparation_audit.json',dict(status='complete',source_works=19,admitted=len(kept),known24_parity=True,current_train_dev_ids_and_opuses_disjoint=True,source_hash_and_note_fingerprint_no_overlap=True,historical_use_not_erased=True,no_model_inference=True,seconds=time.monotonic()-began))


def evaluate():
    began=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert read(OUT/'preparation_audit.json')['status']=='complete';contract=read(OUT/'contract.json');assert all(sha(p)==s for p,s in contract['sources'].items());ids=read(OUT/'cohort_locked.json')['admitted'];data={}
    for pid in ids:
        with np.load(ART/'cache'/f'{pid}.npz',allow_pickle=False) as z:data[pid]={k:z[k].copy() for k in z.files}
    for p in (ART/'metrics',ART/'decoder',OUT/'decoder_state'):p.mkdir(exist_ok=True)
    dec.OUT=OUT/'decoder_state';dec.ART=ART/'decoder';cost.ART=ART;rows=[];replays=[];hashes={};sensitivity=[]
    for arm in ('B','U'):
        for spec in read(EXPORTS[arm])['folds']:
            raws=[]
            for c in spec['components']:
                assert time.monotonic()-began<900;cp=torch.load(c['checkpoint'],map_location='cpu',weights_only=False);model=make_model('G64',c['seed']).cuda();model.load_state_dict(cp['model']);norm=Normalizer(cp['mean'],cp['std']);raw=predictions(model,data,norm,torch.device('cuda'));raws.append(raw)
                pid=ids[0];again=predictions(model,{pid:data[pid]},norm,torch.device('cuda'))[pid];error=max(float(abs(again[k]-raw[pid][k]).max()) for k in again);assert error<1e-7;replays.append(dict(arm=arm,fold=spec['fold'],seed=c['seed'],max_gpu_error=error))
            mean={p:{k:np.stack([r[p][k] for r in raws]).mean(0) for k in raws[0][p]} for p in ids};name=f'{arm}_fold{spec["fold"]}';path=ART/'metrics'/f'{name}.csv.gz';save_raw(mean,path,data);stored=checked_raw(pd.read_csv(path),data);hashes[str(path)]=sha(path)
            for policy,strength in [('raw',0.),('B10',1.)]:
                run=name+'_'+policy;r=dec.evaluate(stored,data,spec['threshold'],spec['prior'],strength,run,contract['contract']);rows.append(dict(arm=arm,fold=spec['fold'],policy=policy,**r,**cost.interval_eval(run,data)));positions=pd.read_csv(ART/'decoder'/f'{run}_positions.csv.gz');groups={(p,str(k)):g.beat.to_numpy(int) for (p,k),g in positions.groupby(['piece_id','performance_id'])}
                for pid,v in data.items():
                    mask=v['label_mask'].astype(bool);truth=np.flatnonzero((v['labels']>.5)&mask)
                    for perf in v['performance_ids']:
                        p=groups.get((pid,str(perf)),np.array([],int));p=p[mask[p]];tp=max_match_count(p,truth,1);binary=np.zeros(len(mask),bool);binary[p]=True;sensitivity.append(dict(run=run,piece_id=pid,performance_id=str(perf),maxmatch_f1=2*tp/max(len(p)+len(truth),1),beat_accuracy=float(np.mean(binary[mask]==v['labels'][mask].astype(bool))),all_negative_accuracy=1-len(truth)/int(mask.sum())))
    f=pd.DataFrame(rows);f.to_csv(OUT/'results.csv',index=False);pd.DataFrame(replays).to_csv(OUT/'gpu_replays.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'matching_and_accuracy.csv',index=False)
    primary=pd.read_csv(ART/'decoder/B_fold0_B10_pieces.csv').set_index('piece_id');primary['opus']=primary.index.str.extract(r'op(\d+)',expand=False);groups={k:g for k,g in primary.groupby('opus')};rng=np.random.default_rng(20260916);names=sorted(groups);draws=[]
    for _ in range(2000):draws.append(pd.concat([groups[k] for k in rng.choice(names,len(names),replace=True)]).f1_tol1.mean())
    primary.to_csv(OUT/'primary_per_work.csv');primary.groupby('opus').mean(numeric_only=True).to_csv(OUT/'primary_per_opus.csv');pf=pd.read_csv(ART/'decoder/B_fold0_B10_performances.csv');counts={k:int(pf[k].sum()) for k in ('tp_tol1','fp_tol1','fn_tol1','tp_tol0','fp_tol0','fn_tol0')};tp,fp,fn=(counts[k] for k in ('tp_tol1','fp_tol1','fn_tol1'));sens=pd.DataFrame(sensitivity).query("run=='B_fold0_B10'").groupby('piece_id').mean(numeric_only=True);assert (sens.maxmatch_f1-primary.f1_tol1>=-1e-12).all();coverage=pd.read_csv(OUT/'coverage.csv')
    write(OUT/'primary_summary.json',dict(primary='B_fold0_B10',works=len(ids),opus_groups=len(names),performances=int(coverage.performances.sum()),scored_starts_per_work_total=int(coverage.scored_starts.sum()),macro_f1=float(primary.f1_tol1.mean()),macro_precision=float(primary.precision_tol1.mean()),macro_recall=float(primary.recall_tol1.mean()),exact_f1=float(primary.f1_tol0.mean()),cluster_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),event_counts_across_performances=counts,micro_f1=2*tp/(2*tp+fp+fn),maxmatch_macro_f1=float(sens.maxmatch_f1.mean()),beat_accuracy=float(sens.beat_accuracy.mean()),all_negative_accuracy=float(sens.all_negative_accuracy.mean()),strict_blind_test=False,not_training_or_dev_for_current_model=True,historical_selection_bias_possible=True,exclusions_before_predictions=True))
    assert all(sha(p)==s for p,s in {**contract['sources'],**hashes}.items());write(OUT/'completion_audit.json',dict(status='complete',new_training=0,retuned_parameters=False,ensemble_cells=8,component_gpu_replays=len(replays),primary_prespecified=True,all_sources_checkpoints_unchanged=True,probability_and_event_counts_replayed=True,strict_blind_test=False,seconds=time.monotonic()-began));write(OUT/'decoder_state/STATE.json',dict(status='complete',pid=None));print(read(OUT/'primary_summary.json'));print(f[['arm','fold','policy','macro_f1_tol1','macro_f1_tol0','interval_f1_tol1']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);args=p.parse_args();prepare() if args.stage=='prepare' else evaluate()
