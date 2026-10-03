"""Decompose the frozen random-message gain before any new model fitting."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import run_recurrence_message_probe as prev
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .recurrence_mean_control import build_mean_graph
from .recurrence_message_probe import mix_probabilities
from .score_context_study import ROOT,read,write,sha,GRID
from .three_round_round2 import split_ids
from .phase2_models import choose_single_threshold
from .local_context_study import metrics
from .interstart_decoder import fit_prior
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import save_raw,error

OUT=ROOT/'reports/recurrence_mean_control';ART=ROOT/'artifacts/recurrence_mean_control'
KINDS=('M','U','S');COLS=prev.COLS

def main():
    began=time.monotonic()
    for p in (OUT,ART,ART/'graphs',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(prev.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(prev.OUT/'contract.json')['hashes'])
    files=[Path(__file__),ROOT/'src/recurrence_mean_control.py',ROOT/'tests/test_recurrence_mean_control.py',OUT/'PROTOCOL.md',prev.ART/'summary.csv']
    graph_paths=sorted((prev.ART/'graphs').glob('*.npz'));assert len(graph_paths)==19
    coverage=[]
    for p in graph_paths:
        with np.load(p,allow_pickle=False) as z:g=z['G']
        result={kind:build_mean_graph(g,p.stem,kind) for kind in KINDS}
        path=ART/'graphs'/p.name
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                for k,v in result.items():np.testing.assert_array_equal(z[k],v)
        else:np.savez_compressed(path,**result)
        files.append(path)
        for k,v in result.items():coverage.append(dict(piece_id=p.stem,kind=k,covered=int((v.sum(1)>0).sum()),beats=len(v)))
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if (OUT/'completion_audit.json').exists():assert read(OUT/'completion_audit.json')['status']=='complete';print('ALREADY COMPLETE');return
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder';rows=[];checks=[];thresholds=[]
    for f in (0,1):
        ids=split_ids(f);train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');prior=fit_prior(train)
        for s in (42,43):
            oldrun=f'C3_seed{s}_fold{f}';meta=read(base.ART/'metrics'/f'{oldrun}.json')
            original=checked_raw(pd.read_csv(base.ART/'metrics'/f'{oldrun}_predictions.csv.gz'),val)
            assert max(abs(metrics(original,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
            for kind in KINDS:
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                assert time.monotonic()-began<1200
                raw={}
                for pid,pp in original.items():
                    with np.load(ART/'graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z[kind]
                    raw[pid]={k:mix_probabilities(v,g) for k,v in pp.items()}
                name=f'{kind}_seed{s}_fold{f}';path=ART/f'{name}_predictions.csv.gz'
                save_raw(raw,path,val);reloaded=checked_raw(pd.read_csv(path),val);assert error(raw,reloaded)<1e-12
                checks.append(dict(run=name,reloaded=True,sha256=sha(path)))
                adapted,_=choose_single_threshold(reloaded,val,GRID);thresholds.append(dict(kind=kind,fold=f,seed=s,old=meta['threshold'],adapted=adapted))
                for policy,threshold in [('locked',meta['threshold']),('adapted',adapted)]:
                    for strength in (0.,.5):
                        r=dec.evaluate(reloaded,val,threshold,prior,strength,f'{name}_{policy}_lambda{strength:g}',digest)
                        rows.append(dict(kind=kind,fold=f,seed=s,policy=policy,**r))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False)
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
                pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False)
    frame=pd.DataFrame(rows);assert len(frame)==48
    combined=pd.concat([frame,pd.read_csv(prev.ART/'summary.csv')],ignore_index=True);combined.to_csv(OUT/'all_runs.csv',index=False)
    means=combined.groupby(['kind','policy','strength'])[COLS].mean();means.to_csv(OUT/'means.csv')
    comparisons=[]
    for other in ('B','R','G','U','S'):
        for policy in ('locked','adapted'):
            for strength in (0.,.5):
                idx=(combined.policy==policy)&(combined.strength==strength)
                a=combined[idx&(combined.kind=='M')].set_index(['fold','seed']);b=combined[idx&(combined.kind==other)].set_index(['fold','seed'])
                d=a[COLS]-b[COLS];d.to_csv(OUT/f'M_minus_{other}_{policy}_lambda{strength:g}.csv')
                comparisons.append(dict(control=other,policy=policy,strength=strength,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum())))
    write(OUT/'comparisons.json',comparisons)
    for p in graph_paths:
        with np.load(p,allow_pickle=False) as z:original=z['G']
        with np.load(ART/'graphs'/p.name,allow_pickle=False) as z:
            for kind in KINDS:np.testing.assert_array_equal(z[kind],build_mean_graph(original,p.stem,kind))
    assert all(sha(p)==h for p,h in hashes.items())
    assert all(sha(ART/f"{c['run']}_predictions.csv.gz")==c['sha256'] for c in checks)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=0,decoder_cells=48,old_metrics_recomputed=4,saved_predictions_reloaded=12,graphs_recomputed=57,source_hashes_unchanged=True,labels_used_in_graphs=False,test_used=False,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None));print('COMPLETE\n'+means.to_string(),flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
