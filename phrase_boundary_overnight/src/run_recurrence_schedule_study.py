"""Fixed C3 CNN: constant versus predeclared cosine, exact-loop reproduction probe."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import recurrence_schedule_training as engine
from . import run_recurrence_depth_study as prior
from .recurrence_schedule_training import make_model
from .phrase_end_auxiliary import ROOT,read,write,sha,normalizer,split_ids
from .local_context_study import predictions,metrics
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/recurrence_schedule_study';ART=ROOT/'artifacts/recurrence_schedule_study';COLS=prior.COLS


def dataset(ids,kind):
    assert kind in ('C3','K3')
    return prior.dataset(ids,'C3')


def prepare():
    started=time.monotonic()
    assert read(ROOT/'reports/recurrence_lr_duration_study/completion_audit.json')['status']=='complete'
    for p in (OUT,ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(prior.OUT/'contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    checks=[]
    for f in (0,1):
        ids=split_ids(f);assert not set(ids['train'])&set(ids['validation'])
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        data=dataset(ids['train'],'K3');norm=normalizer(data)
        for s in (42,43):
            run=f'C3_seed{s}_fold{f}';cp=prior.ART/'checkpoints'/run
            best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
            for p in (cp/'best.pt',cp/'latest.pt',prior.ART/'metrics'/f'{run}.json',prior.ART/'metrics'/f'{run}_predictions.csv.gz'):
                hashes[str(p)]=sha(p)
            checks.append(dict(fold=f,seed=s,same_input_loader=True,train_norm_matches=True,opus_disjoint=True))
    for s in (42,43):
        a=make_model('C3',s).eval();b=make_model('K3',s).eval()
        assert sum(p.numel() for p in b.parameters())==5921 and len(b.blocks)==0
        x=torch.randn(2,67,58);mask=torch.zeros(2,67,dtype=torch.bool);mask[1,55:]=True
        torch.testing.assert_close(a(x,padding_mask=mask),b(x,padding_mask=mask),atol=2e-6,rtol=2e-6)
    pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    for p in (Path(__file__),ROOT/'src/recurrence_schedule_training.py',ROOT/'tests/test_recurrence_schedule_training.py',
              ROOT/'src/recurrence_depth_models.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==digest for r in done)
    assert time.monotonic()-started<300
    write(OUT/'STATE.json',dict(status='ready',contract=digest,completed=[r['run_id'] for r in done],
              seconds=sum(r['seconds'] for r in done),prepare_seconds=time.monotonic()-started,pid=None))
    return digest


def verify_reproduction():
    began=time.monotonic()
    run='C3_seed42_fold0';original=prior.ART/'checkpoints'/run;new=ART/'checkpoints'/run
    expected_contract=read(OUT/'contract.json')['contract'];hashes={}
    for name in ('best.pt','latest.pt'):
        a=torch.load(original/name,map_location='cpu',weights_only=False)
        b=torch.load(new/name,map_location='cpu',weights_only=False)
        assert b['contract']==expected_contract and a['step']==b['step'] and a['sampler']==b['sampler']
        assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
        for index,values in a['optimizer']['state'].items():
            for key,value in values.items():
                torch.testing.assert_close(value,b['optimizer']['state'][index][key],atol=0,rtol=0)
        assert torch.equal(a['rng'],b['rng'])
        assert len(a['cuda_rng'])==len(b['cuda_rng']) and all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))
        for key,value in a['model'].items():torch.testing.assert_close(value,b['model'][key],atol=0,rtol=0)
        np.testing.assert_array_equal(a['mean'],b['mean']);np.testing.assert_array_equal(a['std'],b['std'])
        hashes[str(new/name)]=sha(new/name)
    val=dataset(split_ids(0)['validation'],'C3');norm=normalizer(dataset(split_ids(0)['train'],'C3'))
    a=read(prior.ART/'metrics'/f'{run}.json');b=read(ART/'metrics'/f'{run}.json')
    assert max(abs(a[c]-b[c]) for c in COLS)<1e-10
    raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val)
    old=checked_raw(pd.read_csv(prior.ART/'metrics'/f'{run}_predictions.csv.gz'),val)
    assert max(float(abs(raw[p][q]-old[p][q]).max()) for p in raw for q in raw[p])<1e-10
    model=make_model('C3',42).cuda()
    model.load_state_dict(torch.load(new/'best.pt',map_location='cpu',weights_only=False)['model'])
    replay=predictions(model,val,norm,torch.device('cuda'))
    error=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert error<2e-4
    assert max(abs(metrics(replay,val,b['threshold'])[2][c]-b[c]) for c in COLS)<1e-10
    write(OUT/'baseline_reproduction.json',dict(status='complete',contract=expected_contract,exact_best_and_latest_weights=True,
          exact_sampler=True,exact_optimizer_and_rng=True,metrics_equal=True,replay_error=error,checkpoint_hashes=hashes,
          seconds=b['seconds'],verification_seconds=time.monotonic()-began))
    print('BASELINE REPRODUCTION PASSED',flush=True)


def audit():
    started=time.monotonic();contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    rows=[];checks=[];hashes={}
    for kind in ('C3','K3'):
        source=prior.ART if kind=='C3' else ART;expected=read(prior.OUT/'contract.json')['contract'] if kind=='C3' else contract['contract']
        for f in (0,1):
            for s in (42,43):
                assert time.monotonic()-started<900
                run=f'{kind}_seed{s}_fold{f}';r=read(source/'metrics'/f'{run}.json');ids=split_ids(f)
                train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind);norm=normalizer(train);cp=source/'checkpoints'/run
                best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
                for p in (cp/'best.pt',cp/'latest.pt'):hashes[str(p)]=sha(p)
                assert (best['contract'],best['kind'],best['fold'],best['seed'])==(expected,kind,f,s)
                assert last['contract']==r['contract']==expected and last['step']==300
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'),val);score=metrics(raw,val,r['threshold'])[2]
                assert max(abs(score[c]-r[c]) for c in COLS)<1e-10
                m=make_model(kind,s).cuda();m.load_state_dict(best['model']);assert len(m.blocks)==0
                assert r['params']==5921
                dropout=.2
                expected_lr=engine.learning_rate(kind,best['step']-1)
                assert all(g['lr']==engine.learning_rate(kind,cp_state['step']-1) and g['weight_decay']==.0001 for cp_state in (best,last) for g in cp_state['optimizer']['param_groups'])
                assert len(m.frontend.layers)==3 and m.input_projection.out_features==32
                assert [layer.dropout.p for layer in m.frontend.layers]==[dropout]*3
                replay=predictions(m,val,norm,torch.device('cuda'))
                error=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert error<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-score[c]) for c in COLS)<1e-10
                tr=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
                checks.append(dict(run_id=run,replay_error=error,learning_rate=expected_lr,
                     train_f1=tr['macro_f1_tol1'],gap=tr['macro_f1_tol1']-score['macro_f1_tol1']));rows.append(r)
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    reproduction=read(OUT/'baseline_reproduction.json');assert reproduction['status']=='complete' and reproduction['contract']==contract['contract']
    hashes.update(reproduction['checkpoint_hashes'])
    assert len(hashes)==18 and all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    df=pd.DataFrame(rows);df.to_csv(ART/'summary.csv',index=False);means=df.groupby('kind')[COLS].mean();means.to_csv(OUT/'means.csv')
    a=df[df.kind=='K3'].set_index(['fold','seed']);b=df[df.kind=='C3'].set_index(['fold','seed']);d=a[COLS]-b[COLS];d.to_csv(OUT/'deltas.csv')
    passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)
    write(OUT/'comparison.json',dict(**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),passed=passed))
    write(OUT/'checkpoint_hashes.json',hashes);write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=4,reused_runs=4,additional_baseline_reproduction_runs=1,
          full_gpu_replays=9,checkpoints=18,hashes_unchanged=True,norms_rebuilt=True,test_predictions_accessed=False,
          optimizer_parameters_verified=True,new_training_validation_seconds=float(df[df.kind=='K3'].seconds.sum()),baseline_reproduction_seconds=read(ART/'metrics'/'C3_seed42_fold0.json')['seconds'],
          baseline_verification_seconds=reproduction['verification_seconds'],audit_seconds=time.monotonic()-started))
    write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1500.;engine.make_model=make_model;engine.dataset=dataset
    try:
        if args.stage=='audit':verify_reproduction();audit();return
        digest=prepare()
        if args.stage=='prepare':print(digest,flush=True);return
        engine.train_one('C3',0,42,digest)
        verify_reproduction()
        for f in (0,1):
            for s in (42,43):engine.train_one('K3',f,s,digest)
        write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise


if __name__=='__main__':main()
