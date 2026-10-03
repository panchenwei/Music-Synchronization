"""Frozen-score graph probability diagnosis, development only, no training."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .recurrence_message_probe import build_graph,random_control,mix_probabilities
from .score_context_study import ROOT,read,write,sha,GRID
from .three_round_round2 import split_ids
from .local_context_study import metrics
from .phase2_models import choose_single_threshold
from .interstart_decoder import fit_prior
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import save_raw,error

OUT=ROOT/'reports/recurrence_message_probe';ART=ROOT/'artifacts/recurrence_message_probe'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']

def prepare():
    for p in (OUT,ART,ART/'graphs',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/relative_position_study/completion_audit.json')['status']=='complete'
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    files=[Path(__file__),ROOT/'src/recurrence_message_probe.py',ROOT/'tests/test_recurrence_message_probe.py',OUT/'PROTOCOL.md',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/context_inference_audit_v2.py']
    val_ids=sorted(set(p for f in (0,1) for p in split_ids(f)['validation']))
    assert len(val_ids)==19
    for pid in val_ids:
        source=ROOT/'artifacts/note_relation_study/cache'/f'{pid}.npz';files.append(source)
        with np.load(source,allow_pickle=False) as z:g=build_graph(z['note_events'],int(z['n_beats']))
        r=random_control(g,pid);np.testing.assert_array_equal(np.sort(g,axis=1),np.sort(r,axis=1))
        target=ART/'graphs'/f'{pid}.npz'
        if target.exists():
            with np.load(target,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['G'],g);np.testing.assert_array_equal(z['R'],r)
        else:np.savez_compressed(target,G=g,R=r)
        files.append(target)
    for f in (0,1):
        for s in (42,43):
            files.extend(base.ART/'metrics'/f'C3_seed{s}_fold{f}{ext}' for ext in ('.json','_predictions.csv.gz'))
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    return digest,hashes

def graph_diagnostic(val):
    rows=[]
    for pid,v in val.items():
        mask=v['label_mask'].astype(bool);y=v['labels'].astype(float);positive=(y>0)&mask
        with np.load(ART/'graphs'/f'{pid}.npz',allow_pickle=False) as z:
            for kind in ('G','R'):
                g=z[kind];covered=g.sum(1)>0
                # Only diagnostic uses known labels; no labels enter graph construction or messages.
                mass=g@mask.astype(float);target=(g@(y*mask))/np.maximum(mass,1e-12)
                eligible=positive&covered&(mass>0)
                a,b=np.nonzero(g)
                rows.append(dict(piece_id=pid,kind=kind,beats=len(y),edges=len(a),coverage=float(covered.mean()),positive_count=int(positive.sum()),covered_positive_count=int((positive&covered).sum()),diagnostic_positive_count=int(eligible.sum()),neighbor_start_rate_given_start=float(target[eligible].mean()) if eligible.any() else np.nan,prevalence=float(y[mask].mean()),mean_distance=float(abs(a-b).mean()) if len(a) else 0))
    return rows

def main():
    began=time.monotonic();digest,hashes=prepare()
    if (OUT/'completion_audit.json').exists():
        assert read(OUT/'completion_audit.json')['status']=='complete';print('ALREADY COMPLETE');return
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    rows=[];checks=[];diagnostics=[];thresholds=[]
    for f in (0,1):
        ids=split_ids(f);train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');prior=fit_prior(train)
        diagnostics.extend(graph_diagnostic(val))
        for s in (42,43):
            run=f'C3_seed{s}_fold{f}';meta=read(base.ART/'metrics'/f'{run}.json')
            original=checked_raw(pd.read_csv(base.ART/'metrics'/f'{run}_predictions.csv.gz'),val)
            assert max(abs(metrics(original,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
            for kind in ('B','G','R'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                assert time.monotonic()-began<1200
                raw={}
                for pid,pp in original.items():
                    if kind=='B':raw[pid]={k:v.copy() for k,v in pp.items()};continue
                    with np.load(ART/'graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z[kind]
                    raw[pid]={k:mix_probabilities(v,g) for k,v in pp.items()}
                name=f'{kind}_seed{s}_fold{f}';path=ART/f'{name}_predictions.csv.gz'
                save_raw(raw,path,val);reloaded=checked_raw(pd.read_csv(path),val);assert error(raw,reloaded)<1e-12
                adapted,_=choose_single_threshold(reloaded,val,GRID)
                if kind=='B':assert adapted==meta['threshold'] and error(original,reloaded)<1e-12
                checks.append(dict(run=name,reloaded=True,probability_change=error(raw,original),sha256=sha(path)))
                thresholds.append(dict(kind=kind,fold=f,seed=s,old=meta['threshold'],adapted=adapted))
                for policy,threshold in [('locked',meta['threshold']),('adapted',adapted)]:
                    for strength in (0.,.5):
                        r=dec.evaluate(reloaded,val,threshold,prior,strength,f'{name}_{policy}_lambda{strength:g}',digest)
                        if kind=='B' and strength==0:assert max(abs(r[c]-meta[c]) for c in COLS)<1e-10
                        rows.append(dict(kind=kind,fold=f,seed=s,policy=policy,**r))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False)
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
                pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False)
    frame=pd.DataFrame(rows);assert len(frame)==48
    means=frame.groupby(['kind','policy','strength'])[COLS].mean();means.to_csv(OUT/'means.csv')
    pd.DataFrame(diagnostics).to_csv(OUT/'graph_diagnostics.csv',index=False)
    comparisons=[]
    for other in ('B','R'):
        for policy in ('locked','adapted'):
            for strength in (0.,.5):
                index=(frame.policy==policy)&(frame.strength==strength)
                a=frame[index&(frame.kind=='G')].set_index(['fold','seed'])
                b=frame[index&(frame.kind==other)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
                d.to_csv(OUT/f'G_minus_{other}_{policy}_lambda{strength:g}.csv')
                passed=d.macro_f1_tol1.mean()>=.015 and (d.macro_f1_tol1>0).sum()>=3 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0
                comparisons.append(dict(candidate='G',control=other,policy=policy,strength=strength,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(passed)))
    write(OUT/'comparisons.json',comparisons)
    rebuilt=0
    for path in sorted((ART/'graphs').glob('*.npz')):
        with np.load(ROOT/'artifacts/note_relation_study/cache'/path.name,allow_pickle=False) as z:
            recomputed=build_graph(z['note_events'],int(z['n_beats']))
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['G'],recomputed)
            np.testing.assert_array_equal(z['R'],random_control(recomputed,path.stem))
        rebuilt+=1
    assert rebuilt==19
    assert all(sha(p)==h for p,h in hashes.items())
    assert all(sha(ART/f"{c['run']}_predictions.csv.gz")==c['sha256'] for c in checks)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=0,decoder_cells=48,old_metrics_recomputed=4,saved_predictions_reloaded=12,graphs_recomputed=19,source_hashes_unchanged=True,random_row_statistics_equal=True,labels_used_in_graphs=False,test_used=False,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None))
    print('COMPLETE\n'+means.to_string(),flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
