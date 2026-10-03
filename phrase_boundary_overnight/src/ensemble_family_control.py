"""No fitting: equal-cost homogeneous and cross-family probability averages."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .score_context_study import ROOT,read,write,sha,GRID
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .three_round_round2 import split_ids
from .local_context_study import metrics
from .phase2_models import choose_single_threshold
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import save_raw,error
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities

OUT=ROOT/'reports/ensemble_family_control';ART=ROOT/'artifacts/ensemble_family_control'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def source(kind,seed):
    if kind=='C3':return ROOT/'artifacts'/('recurrence_depth_study' if seed<44 else 'c3_seed_replication')
    return ROOT/'artifacts'/('current_gru_study' if seed<44 else 'current_gru_replication')


def main():
    start=time.monotonic()
    for p in (OUT,ART,ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes={}
    for study in ('current_gru_study','current_gru_replication'):
        assert read(ROOT/'reports'/study/'completion_audit.json')['status']=='complete'
        hashes.update(read(ROOT/'reports'/study/'contract.json')['hashes'])
    for f in (0,1):
        for s in (42,43,44,45):
            for kind in ('C3','G'):
                for suffix in ('.json','_predictions.csv.gz'):
                    p=source(kind,s)/'metrics'/f'{kind}_seed{s}_fold{f}{suffix}';hashes[str(p)]=sha(p)
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/interstart_decoder.py',ROOT/'src/run_halo_decoder_composition.py'):
        hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if (OUT/'completion_audit.json').exists():print('ALREADY COMPLETE');return
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    rows=[];checks=[];dependence=[]
    for f in (0,1):
        ids=split_ids(f);train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');prior=fit_prior(train)
        for a,b in ((42,43),(44,45)):
            originals={}
            for kind in ('C3','G'):
                for seed in (a,b):
                    run=f'{kind}_seed{seed}_fold{f}';src=source(kind,seed)
                    meta=read(src/'metrics'/f'{run}.json');raw=checked_raw(pd.read_csv(src/'metrics'/f'{run}_predictions.csv.gz'),val)
                    assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                    originals[(kind,seed)]=raw
            for family,lkey,rkey in [('CC',('C3',a),('C3',b)),('GG',('G',a),('G',b)),('CGa',('C3',a),('G',b)),('CGb',('C3',b),('G',a))]:
                assert time.monotonic()-start<1200
                left,right=originals[lkey],originals[rkey]
                raw={p:{q:.5*(x+right[p][q]) for q,x in pp.items()} for p,pp in left.items()}
                threshold,_=choose_single_threshold(raw,val,GRID);run=f'{family}_pair{a}-{b}_fold{f}'
                path=ART/f'{run}_predictions.csv.gz';save_raw(raw,path,val)
                assert error(raw,checked_raw(pd.read_csv(path),val))<1e-12
                checks.append(dict(run=run,probability_sha256=sha(path),threshold=threshold))
                adjusted={}
                for pid,pp in raw.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[pid]={q:mix_probabilities(x,g) for q,x in pp.items()}
                    mask=val[pid]['label_mask']>.5
                    correlations=[float(np.corrcoef(left[pid][q][mask],right[pid][q][mask])[0,1]) for q in pp]
                    dependence.append(dict(fold=f,pair=f'{a}-{b}',family=family,piece_id=pid,mean_probability_correlation=float(np.mean(correlations))))
                for policy,x,strength in [('raw',raw,0.),('M10',adjusted,1.)]:
                    score=dec.evaluate(x,val,threshold,prior,strength,f'{run}_{policy}',digest)
                    if policy=='raw':assert max(abs(score[c]-metrics(raw,val,threshold)[2][c]) for c in COLS)<1e-10
                    rows.append(dict(family=family,pair=f'{a}-{b}',fold=f,policy=policy,**score))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False)
                pd.DataFrame(checks).to_csv(OUT/'probability_audit.csv',index=False)
    df=pd.DataFrame(rows);assert len(df)==32 and len(checks)==16
    # The two heterogeneous assignments share both trained model sets. Average
    # within seed-pair, not as if CGa/CGb were independent experiments.
    cg=df[df.family.isin(['CGa','CGb'])].groupby(['pair','fold','policy'])[COLS].mean().reset_index();cg['family']='CGmean'
    allrows=pd.concat([df,cg],ignore_index=True);allrows.to_csv(OUT/'all_runs.csv',index=False)
    means=allrows.groupby(['family','policy'])[COLS].mean();means.to_csv(OUT/'means.csv')
    comp=[]
    for ref in ('CC','GG'):
        a=allrows[(allrows.family=='CGmean')&(allrows.policy=='M10')].set_index(['pair','fold'])
        b=allrows[(allrows.family==ref)&(allrows.policy=='M10')].set_index(['pair','fold']);d=a[COLS]-b[COLS]
        comp.append(dict(reference=ref,**d.mean().to_dict(),positive_pair_fold_cells=int((d.macro_f1_tol1>0).sum()),independent_works_not_seed_pairs=True))
    write(OUT/'comparisons.json',comp);pd.DataFrame(dependence).to_csv(OUT/'probability_dependence.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    assert all(sha(ART/f"{r['run']}_predictions.csv.gz")==r['probability_sha256'] for r in checks)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=0,old_model_metrics_recomputed=16,decoder_cells=32,
        saved_probabilities_reloaded=16,source_hashes_unchanged=True,test_used=False,seconds=time.monotonic()-start))
    print(means.to_string(),flush=True);print(comp,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
