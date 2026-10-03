"""Matched continuous versus independent beat roll inputs, no old artifact edits."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import ensemble_seed_replication as engine
from .crossbeat_roll_branch import CrossbeatBoundary,predictions
from .label_repaired_rebaseline import dataset
from .phrase_end_auxiliary import ROOT,read,write,sha,normalizer,split_ids
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/crossbeat_roll_study';ART=ROOT/'artifacts/crossbeat_roll_study'
MODES=('independent','continuous');COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def prepare():
    hashes=dict(read(ROOT/'reports/label_repaired_rebaseline/contract.json')['hashes'])
    assert all(sha(p)==h for p,h in hashes.items())
    for p in (Path(__file__),ROOT/'src/crossbeat_roll_branch.py',ROOT/'tests/test_crossbeat_roll_branch.py',OUT/'PROTOCOL.md'):
        hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    for mode in MODES:
        for p in (ART/mode/'checkpoints',ART/mode/'metrics',OUT/mode):p.mkdir(parents=True,exist_ok=True)
        done=[read(p) for p in (ART/mode/'metrics').glob('*_fold*.json')];assert all(r['contract']==contract for r in done)
        write(OUT/mode/'STATE.json',dict(status='ready',contract=contract,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None))
    write(OUT/'STATE.json',dict(status='ready',contract=contract))
    return contract


def audit_report():
    began=time.monotonic();hashes=read(OUT/'contract.json')['hashes'];assert all(sha(p)==h for p,h in hashes.items())
    rows=[];audits=[];cps=list(ART.glob('*/checkpoints/*/*.pt'));assert len(cps)==16
    before={str(p):sha(p) for p in cps}
    for mode in MODES:
        results=[read(p) for p in (ART/mode/'metrics').glob('*_fold*.json')];assert len(results)==4
        assert set((r['fold'],r['seed']) for r in results)=={(f,s) for f in (0,1) for s in (42,43)}
        for r in results:
            ids=split_ids(r['fold']);assert not set(ids['train'])&set(ids['validation'])
            train=dataset(ids['train'],'R');val=dataset(ids['validation'],'R');norm=normalizer(train)
            cp=ART/mode/'checkpoints'/r['run_id'];best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
            assert last['step']==300 and best['contract']==last['contract']==read(OUT/'contract.json')['contract']
            assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
            np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
            saved=checked_raw(pd.read_csv(ART/mode/'metrics'/f"{r['run_id']}_predictions.csv.gz"),val)
            score=metrics(saved,val,r['threshold'])[2];assert max(abs(score[k]-r[k]) for k in COLS)<1e-10
            model=CrossbeatBoundary(mode=='continuous',r['seed']);initial_stem={k:v.clone() for k,v in model.stem.state_dict().items()};model.load_state_dict(best['model'])
            assert all(torch.equal(initial_stem[k],v) for k,v in model.stem.state_dict().items())
            model.to('cuda').eval();replay=predictions(model,val,norm,torch.device('cuda'))
            delta=max(float(np.max(abs(replay[p][k]-saved[p][k]))) for p in saved for k in saved[p]);assert delta<2e-4
            error=max(abs(metrics(replay,val,r['threshold'])[2][k]-score[k]) for k in COLS);assert error<1e-10
            ts=metrics(predictions(model,train,norm,torch.device('cuda')),train,r['threshold'])[2]
            audits.append(dict(mode=mode,run_id=r['run_id'],probability_replay_error=delta,metric_replay_error=error,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']))
            rows.append(dict(mode=mode,**{k:v for k,v in r.items() if k!='history'}));pd.DataFrame(audits).to_csv(OUT/'run_audit.csv',index=False)
            print('AUDITED',mode,r['run_id'],flush=True)
    df=pd.DataFrame(rows);df.to_csv(ART/'summary.csv',index=False);means=df.groupby('mode')[COLS+['macro_precision_tol1','macro_recall_tol1','seconds']].mean();means.to_csv(OUT/'means.csv')
    a=df[df['mode']=='continuous'].set_index(['fold','seed']);b=df[df['mode']=='independent'].set_index(['fold','seed']);d=a[COLS]-b[COLS];d.to_csv(OUT/'paired_deltas.csv')
    write(OUT/'comparison.json',dict(mean_delta=d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    assert all(sha(p)==h for p,h in hashes.items()) and before=={str(p):sha(p) for p in cps}
    write(OUT/'checkpoint_hashes.json',before);write(OUT/'completion_audit.json',dict(status='complete',runs=8,checkpoints=16,full_gpu_replays=8,train_norms_recomputed=True,stem_weights_unchanged=True,source_and_checkpoint_hashes_unchanged=True,test_predictions_accessed=False,audit_seconds=time.monotonic()-began,training_validation_seconds=float(df.seconds.sum())))
    write(OUT/'STATE.json',dict(status='complete',contract=read(OUT/'contract.json')['contract']));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=p.parse_args();torch.set_num_threads(2)
    try:
        if args.stage=='audit':audit_report();return
        contract=prepare()
        if args.stage=='prepare':print(contract);return
        engine.dataset=dataset;engine.predictions=predictions;engine.CAP=1200.
        for mode in MODES:
            engine.OUT=OUT/mode;engine.ART=ART/mode;engine.build_model=lambda kind,seed:CrossbeatBoundary(mode=='continuous',seed)
            write(OUT/'STATE.json',dict(status='running',mode=mode,contract=contract))
            for fold in (0,1):
                for seed in (42,43):engine.train_one('R',fold,seed,contract)
            write(OUT/mode/'STATE.json',{**read(OUT/mode/'STATE.json'),'status':'training_complete','pid':None})
        write(OUT/'STATE.json',dict(status='auditing',contract=contract));audit_report()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed'));raise


if __name__=='__main__':main()
