"""Revisit fixed shallow trees with the verified ordered-recurrence inputs."""
import hashlib,json,os,time,traceback
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from . import tabular_context_study as engine
from . import run_recurrence_depth_study as base
from .score_context_study import ROOT,read,write,sha,normalizer
from .three_round_round2 import split_ids
from .audit_external_stem_transfer import checked_raw
from .local_context_study import metrics

OUT=ROOT/'reports/recurrence_tabular_study';ART=ROOT/'artifacts/recurrence_tabular_study'
KINDS={'R0':0,'R2':2};COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids,kind):return base.dataset(ids,'C3')


def prepare():
    for p in (OUT,ART/'models',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/halo_decoder_composition/completion_audit.json')['status']=='complete'
    hashes=dict(read(base.OUT/'contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    for f in (0,1):
        ids=split_ids(f);data=dataset(ids['train'],'B');norm=normalizer(data)
        assert len(norm.mean)==58 and all(v['curves'].shape[-1]==58 for v in data.values())
        for seed in (42,43):
            run=f'C3_seed{seed}_fold{f}';path=base.ART/'checkpoints'/run/'best.pt';cp=torch.load(path,map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(norm.mean,cp['mean']);np.testing.assert_array_equal(norm.std,cp['std'])
            for p in (path,base.ART/'metrics'/f'{run}.json',base.ART/'metrics'/f'{run}_predictions.csv.gz'):hashes[str(p)]=sha(p)
    for p in (Path(__file__),ROOT/'src/tabular_context_study.py',ROOT/'tests/test_tabular_context.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes,sklearn_version=engine.sklearn.__version__))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None,contract=digest))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');checks=[];results=[]
    model_hashes={str(p):sha(p) for p in (ART/'models').glob('*/*.joblib')};assert len(model_hashes)==16
    for kind in KINDS:
        for fold in (0,1):
            ids=split_ids(fold);train=dataset(ids['train'],'B');val=dataset(ids['validation'],'B');norm=normalizer(train)
            for seed in (42,43):
                assert time.monotonic()-began<1800
                run=f'{kind}_seed{seed}_fold{fold}';r=read(ART/'metrics'/f'{run}.json');cp=ART/'models'/run
                best=joblib.load(cp/'best.joblib');last=joblib.load(cp/'latest.joblib')
                assert (best['contract'],best['kind'],best['fold'],best['seed'])==(contract['contract'],kind,fold,seed)
                assert last['contract']==contract['contract'] and last['step']==150
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                assert not best['model'].do_early_stopping_ and best['model'].n_features_in_==58*(2*KINDS[kind]+1)
                np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val)
                score=metrics(raw,val,r['threshold'])[2];assert max(abs(score[c]-r[c]) for c in COLS)<1e-10
                replay=engine.raw_predictions(best['model'],val,norm,KINDS[kind])
                err=max(float(abs(raw[p][q]-replay[p][q]).max()) for p in raw for q in raw[p]);assert err<1e-12
                tr=metrics(engine.raw_predictions(best['model'],train,norm,KINDS[kind]),train,r['threshold'])[2]
                assert abs(tr['macro_f1_tol1']-r['train_f1'])<1e-10
                results.append(r);checks.append(dict(run_id=run,replay_error=err,train_f1=tr['macro_f1_tol1'],gap=r['gap']))
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    for fold in (0,1):
        val=dataset(split_ids(fold)['validation'],'B')
        for seed in (42,43):
            run=f'C3_seed{seed}_fold{fold}';r=read(base.ART/'metrics'/f'{run}.json')
            raw=checked_raw(pd.read_csv(base.ART/'metrics'/f'{run}_predictions.csv.gz'),val)
            score=metrics(raw,val,r['threshold'])[2];assert max(abs(score[c]-r[c]) for c in COLS)<1e-10
            results.append(r)
    frame=pd.DataFrame(results);assert len(frame)==12;frame.to_csv(ART/'summary.csv',index=False)
    frame.groupby('kind')[COLS].mean().to_csv(OUT/'means.csv');comparisons=[]
    for kind in KINDS:
        a=frame[frame.kind==kind].set_index(['fold','seed']);b=frame[frame.kind=='C3'].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        d.to_csv(OUT/f'deltas_{kind}.csv')
        passed=d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3
        comparisons.append(dict(kind=kind,**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),passed=bool(passed)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in model_hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'model_hashes.json',model_hashes)
    write(OUT/'completion_audit.json',dict(status='complete',new_cpu_training_runs=8,cpu_prediction_replays=8,model_files=16,
        reused_cnn_metric_recomputations=4,source_hashes_unchanged=True,norms_rebuilt=True,test_used=False,
        training_seconds=float(frame[frame.kind.isin(KINDS)].seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']))


def main():
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1800.;engine.KINDS=KINDS;engine.dataset=dataset;engine.normalizer=normalizer
    contract=prepare()
    with threadpool_limits(limits=2):
        for fold in (0,1):
            for seed in (42,43):
                for kind in KINDS:engine.run_one(kind,fold,seed,contract)
        write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
