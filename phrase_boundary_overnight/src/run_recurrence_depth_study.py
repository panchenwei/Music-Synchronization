"""After three feature rounds: fixed-input CNN depth comparison."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import score_context_study as engine
from . import run_motif_recurrence_study as prior
from .recurrence_depth_models import make_model, DEPTHS
from .phrase_end_auxiliary import ROOT, read, write, sha, normalizer, split_ids
from .local_context_study import metrics, predictions
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/recurrence_depth_study';ART=ROOT/'artifacts/recurrence_depth_study'
OLD=prior.ART;COLS=prior.COLS;PARAMS={'O':3297,'C2':4609,'C3':5921}


def dataset(ids,kind):
    assert kind in DEPTHS
    return prior.dataset(ids,'O')


def bind():
    engine.OUT=OUT;engine.ART=ART;engine.CAP=2400.;engine.dataset=dataset;engine.make_model=make_model


def prepare():
    began=time.monotonic()
    for p in (OUT,ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    for name in ('transposed_recurrence_study','recurrence_novelty_fusion','recurrence_modality_study'):
        assert read(ROOT/'reports'/name/'completion_audit.json')['status']=='complete'
    previous=pd.read_csv(ROOT/'reports/recurrence_modality_study/comparisons.csv')
    assert not previous[previous.reference=='O'].passed.any(), 'Reconsider frozen input if modality candidate was promoted'
    hashes=dict(read(prior.OUT/'contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    checks=[]
    for f in (0,1):
        ids=split_ids(f);assert not set(ids['train'])&set(ids['validation'])
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        old=prior.dataset(ids['train'],'O');a=normalizer(old)
        for k in ('C2','C3'):
            new=dataset(ids['train'],k);b=normalizer(new)
            np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.std,b.std)
            for pid,v in new.items():
                for field in ('curves','pitch_profiles','labels','label_mask','performance_ids'):
                    np.testing.assert_array_equal(v[field],old[pid][field])
            checks.append(dict(fold=f,kind=k,all_inputs_norm_targets_equal=True,opus_disjoint=True))
        for s in (42,43):
            run=f'O_seed{s}_fold{f}'
            for p in (OLD/'metrics'/f'{run}.json',OLD/'metrics'/f'{run}_predictions.csv.gz',
                      OLD/'checkpoints'/run/'best.pt',OLD/'checkpoints'/run/'latest.pt'):hashes[str(p)]=sha(p)
    pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    for s in (42,43):
        old=make_model('O',s)
        for k in ('C2','C3'):
            m=make_model(k,s);assert len(m.frontend.layers)==DEPTHS[k] and len(m.blocks)==0
            assert sum(p.numel() for p in m.parameters())==PARAMS[k]
            for name,v in old.state_dict().items():
                key=name.replace('frontend.','frontend.layers.0.',1) if name.startswith('frontend.') else name
                torch.testing.assert_close(v,m.state_dict()[key],atol=0,rtol=0)
    for p in (Path(__file__),ROOT/'src/recurrence_depth_models.py',ROOT/'tests/test_recurrence_depth_models.py',OUT/'PROTOCOL.md'):
        hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==digest for r in done)
    assert time.monotonic()-began<300
    write(OUT/'STATE.json',dict(status='ready',contract=digest,pid=None,completed=[r['run_id'] for r in done],
          seconds=sum(r['seconds'] for r in done),prepare_seconds=time.monotonic()-began))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    rows=[];checks=[];hashes={}
    for kind in ('O','C2','C3'):
        source=OLD if kind=='O' else ART
        expected=read(prior.OUT/'contract.json')['contract'] if kind=='O' else contract['contract']
        for fold in (0,1):
            for seed in (42,43):
                assert time.monotonic()-began<900
                run=f'{kind}_seed{seed}_fold{fold}';r=read(source/'metrics'/f'{run}.json');ids=split_ids(fold)
                train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind);norm=normalizer(train)
                cp=source/'checkpoints'/run
                for p in (cp/'best.pt',cp/'latest.pt'):hashes[str(p)]=sha(p)
                best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
                last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
                assert (best['contract'],best['kind'],best['fold'],best['seed'])==(expected,kind,fold,seed)
                assert last['contract']==r['contract']==expected and last['step']==300
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'),val)
                score=metrics(raw,val,r['threshold'])[2];assert max(abs(score[c]-r[c]) for c in COLS)<1e-10
                m=make_model(kind,seed).cuda();m.load_state_dict(best['model'])
                assert r['params']==PARAMS[kind] and len(m.blocks)==0
                learned=[]
                if kind!='O':
                    assert len(m.frontend.layers)==DEPTHS[kind]
                    learned=[float(layer.pointwise.weight.detach().abs().sum()) for layer in m.frontend.layers[1:]]
                    assert all(x>1e-8 for x in learned)
                replay=predictions(m,val,norm,torch.device('cuda'))
                error=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert error<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-score[c]) for c in COLS)<1e-10
                train_score=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
                checks.append(dict(run_id=run,replay_error=error,reused=kind=='O',depth=DEPTHS[kind],
                              added_projection_l1=json.dumps(learned),train_f1=train_score['macro_f1_tol1'],
                              gap=train_score['macro_f1_tol1']-score['macro_f1_tol1']));rows.append(r)
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    assert len(hashes)==24 and all(sha(p)==h for p,h in hashes.items())
    assert all(sha(p)==h for p,h in contract['hashes'].items())
    df=pd.DataFrame(rows);df.to_csv(ART/'summary.csv',index=False)
    means=df.groupby('kind')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for kind in ('C2','C3'):
        a=df[df.kind==kind].set_index(['fold','seed']);b=df[df.kind=='O'].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        d.to_csv(OUT/f'delta_{kind}_O.csv')
        comparisons.append(dict(candidate=kind,reference='O',**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),
          ap_positive=int((d.raw_ap>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0
          and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False);write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,reused_runs=4,full_gpu_replays=12,
          checkpoints=24,hashes_unchanged=True,norms_rebuilt=True,test_predictions_accessed=False,
          deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),added_layers_learned=True,
          new_training_validation_seconds=float(df[df.kind!='O'].seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',contract=contract['contract'],pid=None));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','all','audit'),default='all');args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    try:
        bind()
        if args.stage=='audit':audit();return
        digest=prepare()
        if args.stage=='prepare':print(digest,flush=True);return
        for f in (0,1):
            for s in (42,43):
                for k in ('C2','C3'):engine.train_one(k,f,s,digest)
        write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise


if __name__=='__main__':main()
