"""Post-run score-context evidence audit; never trains or rewrites experiment outputs."""
from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
import torch
from .score_context_study import ROOT,OUT,ART,KINDS,read,write,sha,make_model,dataset,split_ids
from .models import Normalizer
from .local_context_study import metrics,predictions

def main():
    started=time.monotonic();torch.set_num_threads(2)
    state=read(OUT/'STATE.json');assert state['status']=='complete','Run only after all training completes'
    rows=pd.read_csv(ART/'summary.csv');assert len(rows)==16
    contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    checkpoint_before={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(checkpoint_before)==32
    checks=[];histories=[];differences=[]
    for r in rows.to_dict('records'):
        run=r['run_id'];fold=int(r['fold']);seed=int(r['seed']);kind=r['kind'];ids=split_ids(fold)
        data=dataset(ids['validation'],kind);frame=pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz');assert set(frame.piece_id)==set(ids['validation'])
        raw={}
        for (pid,perf),g in frame.groupby(['piece_id','performance_id']):
            g=g.sort_values('beat');np.testing.assert_array_equal(g.beat,np.arange(len(data[pid]['labels'])))
            np.testing.assert_array_equal(g.label,data[pid]['labels']);np.testing.assert_array_equal(g.valid,data[pid]['label_mask'])
            raw.setdefault(pid,{})[perf]=g.probability.to_numpy()
        _,_,score=metrics(raw,data,float(r['threshold']))
        metric_error=max(abs(float(score[k])-r[k]) for k in ('macro_f1_tol0','macro_f1_tol1','macro_f1_tol2','raw_ap'))
        assert metric_error<1e-10
        saved=torch.load(ART/'checkpoints'/run/'best.pt',map_location='cpu',weights_only=False)
        assert saved['contract']==contract['contract'] and saved['step']==r['best_step']
        model=make_model(kind,seed).eval();model.load_state_dict(saved['model']);norm=Normalizer(saved['mean'],saved['std'])
        pid=sorted(data)[0];perf=sorted(raw[pid])[len(raw[pid])//2];pi=list(data[pid]['performance_ids'].astype(str)).index(perf)
        x=torch.from_numpy(norm.apply(data[pid]['curves'][pi:pi+1]).astype(np.float32))
        with torch.no_grad():replayed=torch.sigmoid(model(x))[0].numpy()
        error=float(np.max(np.abs(replayed-raw[pid][perf])))
        assert error<2e-4,(run,error)
        # Original and replayed decoding, rather than requiring bitwise CPU/GPU equality.
        _,_,orig=metrics({pid:{perf:raw[pid][perf]}},{pid:data[pid]},float(r['threshold']))
        _,_,re=metrics({pid:{perf:replayed}},{pid:data[pid]},float(r['threshold']))
        assert abs(orig['macro_f1_tol1']-re['macro_f1_tol1'])<1e-12
        latest=torch.load(ART/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False)
        assert latest['step']==300
        for h in latest['history']:histories.append(dict(run_id=run,kind=kind,fold=fold,seed=seed,**h))
        checks.append(dict(run_id=run,metric_max_error=metric_error,replay_piece=pid,replay_performance=perf,replay_max_probability_error=error,replayed_f1_equal=True,params=int(r['params'])))
        print('AUDITED',run,'max_cpu_delta',error,flush=True)
    pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);pd.DataFrame(histories).to_csv(OUT/'training_history.csv',index=False)
    hist=pd.DataFrame(histories)
    diagnostics=[]
    for run,h in hist.groupby('run_id'):
        h=h.sort_values('step');best=rows[rows.run_id==run].iloc[0]
        diagnostics.append(dict(run_id=run,kind=best.kind,best_step=int(best.best_step),best_f1=float(best.macro_f1_tol1),
                                final_f1=float(h.iloc[-1].macro_f1_tol1),first_loss=float(h.iloc[0].loss),last_loss=float(h.iloc[-1].loss),
                                first_ap=float(h.iloc[0].raw_ap),last_ap=float(h.iloc[-1].raw_ap)))
    pd.DataFrame(diagnostics).to_csv(OUT/'selection_diagnostics.csv',index=False)
    assert checkpoint_before=={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')}
    assert all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'completion_audit.json',dict(status='complete',runs=16,checkpoints=32,all_metric_recomputations_pass=True,
                                        checkpoint_replays=16,replayed_decoding_equal=True,source_and_checkpoint_hashes_unchanged=True,
                                        test_predictions_accessed=False,training_runs=0,audit_seconds=time.monotonic()-started))

if __name__=='__main__':main()
