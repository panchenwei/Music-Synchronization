"""C3 with one or two bounded local-attention blocks, fixed 300 updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import score_context_study as engine
from . import run_recurrence_depth_study as prior
from .recurrence_local_attention import make_model
from .phrase_end_auxiliary import ROOT,read,write,sha,normalizer,split_ids
from .local_context_study import predictions,metrics
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/recurrence_attention_study';ART=ROOT/'artifacts/recurrence_attention_study';COLS=prior.COLS


def dataset(ids,kind):
    assert kind in ('C3','A1','A2')
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
        data=dataset(ids['train'],'A1');norm=normalizer(data)
        for s in (42,43):
            run=f'C3_seed{s}_fold{f}';cp=prior.ART/'checkpoints'/run
            best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
            for p in (cp/'best.pt',cp/'latest.pt',prior.ART/'metrics'/f'{run}.json',prior.ART/'metrics'/f'{run}_predictions.csv.gz'):
                hashes[str(p)]=sha(p)
            checks.append(dict(fold=f,seed=s,same_input_loader=True,train_norm_matches=True,opus_disjoint=True))
    for s in (42,43):
        a=make_model('C3',s).eval();b=make_model('A1',s).eval()
        assert sum(p.numel() for p in b.parameters())==14533 and len(b.attention)==1
        x=torch.randn(2,67,58);mask=torch.zeros(2,67,dtype=torch.bool);mask[1,55:]=True
        torch.testing.assert_close(a(x,padding_mask=mask),b(x,padding_mask=mask),atol=2e-6,rtol=2e-6)
    pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    for p in (Path(__file__),ROOT/'src/recurrence_local_attention.py',ROOT/'tests/test_recurrence_local_attention.py',
              ROOT/'src/recurrence_depth_models.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==digest for r in done)
    assert time.monotonic()-started<300
    write(OUT/'STATE.json',dict(status='ready',contract=digest,completed=[r['run_id'] for r in done],
              seconds=sum(r['seconds'] for r in done),prepare_seconds=time.monotonic()-started,pid=None))
    return digest


def audit():
    started=time.monotonic();contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    rows=[];checks=[];hashes={}
    for kind in ('C3','A1','A2'):
        source=prior.ART if kind=='C3' else ART;expected=read(prior.OUT/'contract.json')['contract'] if kind=='C3' else contract['contract']
        for f in (0,1):
            for s in (42,43):
                assert time.monotonic()-started<1800
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
                m=make_model(kind,s).cuda();m.load_state_dict(best['model'])
                assert r['params']=={'C3':5921,'A1':14533,'A2':23145}[kind]
                split_difference=0.
                if kind!='C3':
                    assert len(m.attention)==int(kind[-1])
                    split_difference=min(float(layer.proj.weight.detach().abs().max()) for layer in m.attention)
                    assert split_difference>1e-6
                replay=predictions(m,val,norm,torch.device('cuda'))
                error=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert error<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-score[c]) for c in COLS)<1e-10
                tr=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
                checks.append(dict(run_id=run,replay_error=error,attention_projection_nonzero=split_difference,
                     train_f1=tr['macro_f1_tol1'],gap=tr['macro_f1_tol1']-score['macro_f1_tol1']));rows.append(r)
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    assert len(hashes)==24 and all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    df=pd.DataFrame(rows);df.to_csv(ART/'summary.csv',index=False);means=df.groupby('kind')[COLS].mean();means.to_csv(OUT/'means.csv')
    comparisons=[]
    for kind in ('A1','A2'):
        a=df[df.kind==kind].set_index(['fold','seed']);b=df[df.kind=='C3'].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        d.to_csv(OUT/f'deltas_{kind}.csv')
        passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)
        comparisons.append(dict(kind=kind,**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),passed=passed))
    write(OUT/'comparisons.json',comparisons)
    write(OUT/'checkpoint_hashes.json',hashes);write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,reused_runs=4,
          full_gpu_replays=12,checkpoints=24,hashes_unchanged=True,norms_rebuilt=True,test_predictions_accessed=False,
          attention_projections_learned=True,new_training_validation_seconds=float(df[df.kind!='C3'].seconds.sum()),audit_seconds=time.monotonic()-started))
    write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=2400.;engine.make_model=make_model;engine.dataset=dataset
    try:
        if args.stage=='audit':audit();return
        digest=prepare()
        if args.stage=='prepare':print(digest,flush=True);return
        for f in (0,1):
            for s in (42,43):
                for kind in ('A1','A2'):engine.train_one(kind,f,s,digest)
        write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise


if __name__=='__main__':main()
