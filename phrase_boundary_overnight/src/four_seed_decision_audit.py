"""Eight frozen C3 runs: coverage messages versus relative decoder weight."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import run_recurrence_depth_study as base
from . import run_c3_seed_replication as extra
from . import run_recurrence_mean_control as prev
from . import run_recurrence_message_probe as links
from . import run_halo_decoder_composition as dec
from .recurrence_message_probe import mix_probabilities
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .local_context_study import metrics
from .interstart_decoder import fit_prior
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import save_raw,error

OUT=ROOT/'reports/four_seed_decision_audit';ART=ROOT/'artifacts/four_seed_decision_audit'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']
CONFIGS=(('B05','B',.5),('B075','B',.75),('B10','B',1.),('M','M',.5),('G','G',.5),('S','S',.5))

def bootstrap(pieces):
    manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv',dtype={'opus':str})
    labels=manifest[manifest.split=='validation'][['fold','piece_id','opus']].drop_duplicates()
    p=pieces.groupby(['condition','fold','piece_id']).f1_tol1.mean().reset_index().merge(labels,on=['fold','piece_id'],validate='many_to_one')
    rows=[]
    for candidate,control in [('M','B05'),('M','S'),('M','B075'),('M','B10'),('G','B05'),('B075','B05'),('B10','B05')]:
        a=p[p.condition==candidate].set_index(['fold','piece_id','opus']).f1_tol1
        b=p[p.condition==control].set_index(['fold','piece_id','opus']).f1_tol1;d=(a-b).reset_index(name='delta')
        rng=np.random.default_rng(20260913);samples=[]
        for _ in range(5000):
            fold_means=[]
            for f in (0,1):
                v=d[d.fold==f];groups=[g.delta.to_numpy() for _,g in v.groupby('opus')]
                picked=rng.integers(0,len(groups),len(groups))
                fold_means.append(np.concatenate([groups[i] for i in picked]).mean())
            samples.append(np.mean(fold_means))
        lo,hi=np.quantile(samples,[.025,.975])
        rows.append(dict(candidate=candidate,control=control,estimate=float(d.groupby('fold').delta.mean().mean()),ci95_low=float(lo),ci95_high=float(hi),bootstrap_repeats=5000,opus_groups=int(d.opus.nunique()),works=int(d.piece_id.nunique()),method='within-fold opus cluster bootstrap; conditional on selected trained models; no correction for repeated development selection'))
    pd.DataFrame(rows).to_csv(OUT/'cluster_bootstrap.csv',index=False)

def main():
    began=time.monotonic()
    for p in (OUT,ART,ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(prev.OUT/'completion_audit.json')['status']=='complete' and read(extra.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(prev.OUT/'contract.json')['hashes']);hashes.update(read(extra.OUT/'contract.json')['hashes'])
    files=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_decision_scale_equivalence.py',prev.OUT/'all_runs.csv',extra.ART/'decoder/summary.csv',ROOT/'artifacts/interstart_decoder_study/summary.csv']
    for f in (0,1):
        for s in (42,43,44,45):
            source=base.ART if s<44 else extra.ART
            files.extend(source/'metrics'/f'C3_seed{s}_fold{f}{ext}' for ext in ('.json','_predictions.csv.gz'))
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if (OUT/'completion_audit.json').exists():assert read(OUT/'completion_audit.json')['status']=='complete';print('ALREADY COMPLETE');return
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    old=pd.concat([pd.read_csv(extra.ART/'decoder/summary.csv'),pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv')],ignore_index=True)
    rows=[];checks=[];piece_rows=[]
    for f in (0,1):
        ids=split_ids(f);train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');prior=fit_prior(train)
        for s in (42,43,44,45):
            source=base.ART if s<44 else extra.ART;run=f'C3_seed{s}_fold{f}';meta=read(source/'metrics'/f'{run}.json')
            original=checked_raw(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'),val)
            assert max(abs(metrics(original,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
            transformed={}
            for kind in ('B','M','G','S'):
                raw={}
                for pid,pp in original.items():
                    if kind=='B':raw[pid]={k:v.copy() for k,v in pp.items()};continue
                    graph_path=(links.ART if kind=='G' else prev.ART)/'graphs'/f'{pid}.npz'
                    with np.load(graph_path,allow_pickle=False) as z:g=z[kind]
                    raw[pid]={k:mix_probabilities(v,g) for k,v in pp.items()}
                path=ART/f'{kind}_seed{s}_fold{f}_predictions.csv.gz';save_raw(raw,path,val)
                transformed[kind]=checked_raw(pd.read_csv(path),val);assert error(raw,transformed[kind])<1e-12
                checks.append(dict(path=str(path),sha256=sha(path)))
            for condition,kind,strength in CONFIGS:
                assert time.monotonic()-began<1200
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                name=f'{condition}_seed{s}_fold{f}'
                r=dec.evaluate(transformed[kind],val,meta['threshold'],prior,strength,name,digest)
                if condition=='B05':
                    reference=old[(old.fold==f)&(old.seed==s)&(old.strength==.5)].iloc[0]
                    assert max(abs(r[c]-reference[c]) for c in COLS)<1e-10
                rows.append(dict(condition=condition,kind=kind,fold=f,seed=s,**r))
                p=pd.read_csv(dec.ART/f'{name}_pieces.csv');p['condition']=condition;p['fold']=f;p['seed']=s;piece_rows.append(p)
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False)
    frame=pd.DataFrame(rows);assert len(frame)==48 and len(checks)==32
    subset_means=[]
    for name,seeds in [('all_four',[42,43,44,45]),('original_two',[42,43]),('additional_two',[44,45])]:
        m=frame[frame.seed.isin(seeds)].groupby('condition')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean().reset_index();m['subset']=name;subset_means.append(m)
    means=pd.concat(subset_means);means.to_csv(OUT/'means.csv',index=False)
    comparisons=[]
    for candidate,control in [('M','B05'),('M','G'),('M','S'),('M','B075'),('M','B10'),('G','B05'),('S','B05'),('B075','B05'),('B10','B05')]:
        a=frame[frame.condition==candidate].set_index(['fold','seed']);b=frame[frame.condition==control].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        d.to_csv(OUT/f'{candidate}_minus_{control}.csv')
        extra_gain=float(d.reset_index().query('seed >= 44').macro_f1_tol1.mean())
        passed=d.macro_f1_tol1.mean()>=.015 and (d.macro_f1_tol1>0).sum()>=6 and extra_gain>0 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=-1e-12
        comparisons.append(dict(candidate=candidate,control=control,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),additional_seed_delta=extra_gain,numerical_gate=bool(passed)))
    write(OUT/'comparisons.json',comparisons)
    pieces=pd.concat(piece_rows,ignore_index=True);pieces.to_csv(OUT/'all_piece_metrics.csv',index=False);bootstrap(pieces)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(c['path'])==c['sha256'] for c in checks)
    write(OUT/'probability_hashes.json',checks)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=0,gpu_replays_this_round=0,old_metrics_recomputed=8,old_decoder_recomputed=8,decoder_cells=48,saved_predictions_reloaded=32,source_hashes_unchanged=True,test_used=False,bootstrap_is_blind_test=False,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None));print('COMPLETE\n'+means.to_string(index=False),flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
