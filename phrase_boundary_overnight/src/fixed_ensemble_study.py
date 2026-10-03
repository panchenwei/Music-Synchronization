"""No-training, fixed-weight ensemble with key-aligned probability and metric replay."""
import hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .score_novelty_study import ROOT,read,write,sha,split_ids,dataset
from .local_context_study import metrics
from .phase2_models import choose_single_threshold,nms_probabilities
from .evaluation import one_to_one_counts

OUT=ROOT/'reports/fixed_ensemble_study';ART=ROOT/'artifacts/fixed_ensemble_study'
KEY=['piece_id','performance_id','beat'];META=KEY+['label','valid']
GRID=np.arange(.1,.91,.05).round(2).tolist();MEMBERS={'NR':('N','R'),'BNR':('B','N','R')}


def aligned_average(frames):
    ordered=[f.sort_values(KEY).reset_index(drop=True) for f in frames]
    if not ordered:raise ValueError('No members')
    for f in ordered:
        if f.duplicated(KEY).any():raise ValueError('Duplicate prediction key')
        if not f[META].equals(ordered[0][META]):raise ValueError('Prediction metadata mismatch')
        if not np.isfinite(f.probability).all() or not f.probability.between(0,1).all():raise ValueError('Invalid probability')
    out=ordered[0].copy()
    out['probability']=np.stack([f.probability.to_numpy() for f in ordered]).mean(0)
    return out


def raw_from_frame(frame,data):
    assert set(frame.piece_id)==set(data)
    raw={}
    for (pid,perf),g in frame.groupby(['piece_id','performance_id']):
        g=g.sort_values('beat')
        np.testing.assert_array_equal(g.beat,np.arange(len(data[pid]['labels'])))
        np.testing.assert_array_equal(g.label,data[pid]['labels'])
        np.testing.assert_array_equal(g.valid,data[pid]['label_mask'])
        raw.setdefault(pid,{})[str(perf)]=g.probability.to_numpy()
    for pid in data:assert set(raw[pid])==set(data[pid]['performance_ids'].astype(str))
    return raw


def matched_truth(prob,label,valid,threshold):
    pred=np.flatnonzero((nms_probabilities(prob)>=threshold)&valid)
    truth=np.flatnonzero((label>.5)&valid)
    used_p=set();used_t=set()
    for _,p,t in sorted((abs(int(p)-int(t)),int(p),int(t)) for p in pred for t in truth if abs(p-t)<=1):
        if p not in used_p and t not in used_t:used_p.add(p);used_t.add(t)
    assert len(used_t)==one_to_one_counts(pred,truth,1).tp
    return used_t,set(truth)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(ART/'metrics').mkdir(parents=True,exist_ok=True)
    sources={};hashes={};tables={}
    for study in ('score_novelty_study','score_roll_study'):
        assert read(ROOT/'reports'/study/'completion_audit.json')['status']=='complete'
        parent=read(ROOT/'reports'/study/'contract.json')
        assert all(sha(p)==h for p,h in parent['hashes'].items())
        hashes.update(parent['hashes'])
        tables[study]=pd.read_csv(ROOT/'artifacts'/study/'summary.csv')
    for kind in ('B','N','R'):
        study='score_roll_study' if kind=='R' else 'score_novelty_study'
        for fold in (0,1):
            for seed in (42,43):
                run=f'{kind}_seed{seed}_fold{fold}';p=ROOT/'artifacts'/study/'metrics'/f'{run}_predictions.csv.gz'
                row=tables[study].query('run_id == @run').iloc[0].to_dict()
                sources[(kind,fold,seed)]=(p,row);hashes[str(p)]=sha(p)
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_fixed_ensemble.py',ROOT/'artifacts/score_novelty_study/summary.csv',ROOT/'artifacts/score_roll_study/summary.csv'):
        hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    return sources,contract,hashes


def main():
    torch.set_num_threads(2);started=time.monotonic();sources,contract,hashes=prepare()
    state=dict(status='running',pid=os.getpid(),contract=contract,completed=[]);write(OUT/'STATE.json',state)
    rows=[];diagnostics=[];member_errors=[];audit=[]
    for fold in (0,1):
        data=dataset(split_ids(fold)['validation'],'B')
        for seed in (42,43):
            frames={k:pd.read_csv(sources[(k,fold,seed)][0]) for k in ('B','N','R')}
            # Recompute the member control metrics before computing any fusion.
            for k,f in frames.items():
                r=sources[(k,fold,seed)][1];_,_,score=metrics(raw_from_frame(f,data),data,r['threshold'])
                error=max(abs(score[c]-r[c]) for c in ('macro_f1_tol0','macro_f1_tol1','raw_ap'))
                assert error<1e-10;member_errors.append(dict(kind=k,fold=fold,seed=seed,error=error))
            nr=frames['N'].merge(frames['R'],on=META,suffixes=('_N','_R'),validate='one_to_one')
            assert len(nr)==len(frames['N'])==len(frames['R'])
            for (pid,perf),g in nr.groupby(['piece_id','performance_id']):
                g=g.sort_values('beat');valid=g.valid.to_numpy().astype(bool);label=g.label.to_numpy()
                n,truth=matched_truth(g.probability_N.to_numpy(),label,valid,sources[('N',fold,seed)][1]['threshold'])
                r,_=matched_truth(g.probability_R.to_numpy(),label,valid,sources[('R',fold,seed)][1]['threshold'])
                corr=np.corrcoef(g.probability_N[valid],g.probability_R[valid])[0,1]
                diagnostics.append(dict(fold=fold,seed=seed,piece_id=pid,performance_id=perf,truth=len(truth),both=len(n&r),N_only=len(n-r),R_only=len(r-n),neither=len(truth-(n|r)),correlation=float(corr)))
            for kind,members in MEMBERS.items():
                if time.monotonic()-started>600:raise TimeoutError('600 second evaluation cap')
                run=f'{kind}_seed{seed}_fold{fold}'
                f=aligned_average([frames[k] for k in members]);raw=raw_from_frame(f,data)
                threshold,_=choose_single_threshold(raw,data,GRID);perfs,pieces,score=metrics(raw,data,threshold)
                target=ART/'metrics'/f'{run}_predictions.csv.gz';f.to_csv(target,index=False)
                perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
                saved=pd.read_csv(target).sort_values(KEY).reset_index(drop=True)
                rebuilt=sum(frames[k].sort_values(KEY).probability.to_numpy() for k in members)/len(members)
                error=float(np.max(abs(saved.probability.to_numpy()-rebuilt)));assert error<1e-12
                _,_,replay=metrics(raw_from_frame(saved,data),data,threshold)
                metric_error=max(abs(score[c]-replay[c]) for c in ('macro_f1_tol0','macro_f1_tol1','raw_ap'));assert metric_error<1e-10
                result=dict(run_id=run,kind=kind,fold=fold,seed=seed,members=','.join(members),contract=contract,training_runs=0,**score)
                write(ART/'metrics'/f'{run}.json',result);rows.append(result)
                audit.append(dict(run_id=run,probability_error=error,metric_error=metric_error))
                state['completed'].append(run);write(OUT/'STATE.json',state)
                print(run,'F1',round(score['macro_f1_tol1'],6),'exact',round(score['macro_f1_tol0'],6),'AP',round(score['raw_ap'],6),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(ART/'summary.csv',index=False)
    cols=['macro_f1_tol0','macro_f1_tol1','raw_ap','macro_precision_tol1','macro_recall_tol1']
    means=frame.groupby('kind')[cols].mean();means.to_csv(OUT/'model_means.csv')
    comparisons=[]
    for kind in MEMBERS:
        a=frame[frame.kind==kind].set_index(['fold','seed'])
        for ref in ('B','N','R'):
            b=pd.DataFrame([sources[(ref,f,s)][1] for f in (0,1) for s in (42,43)]).set_index(['fold','seed'])
            d=a[cols[:3]]-b[cols[:3]]
            comparisons.append(dict(kind=kind,reference=ref,f1_delta=d.macro_f1_tol1.mean(),exact_delta=d.macro_f1_tol0.mean(),ap_delta=d.raw_ap.mean(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion_vs_B=bool(ref=='B' and d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    diag=pd.DataFrame(diagnostics);diag.to_csv(OUT/'complementarity.csv',index=False)
    diag[['truth','both','N_only','R_only','neither']].sum().to_csv(OUT/'complementarity_totals.csv')
    pd.DataFrame(audit).to_csv(OUT/'run_audit.csv',index=False);pd.DataFrame(member_errors).to_csv(OUT/'member_replay.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',evaluations=8,member_metric_replays=12,new_training_runs=0,prediction_reloads=8,source_hashes_unchanged=True,test_used=False,seconds=time.monotonic()-started))
    state.update(status='complete',pid=None,seconds=time.monotonic()-started);write(OUT/'STATE.json',state)
    print('MEANS\n'+means.to_string(),flush=True);print('COMPARISONS\n'+pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise
