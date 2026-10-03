"""Frozen label-free score recurrence comparison, separate artifacts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import score_context_study as engine
from .motif_recurrence_features import features
from .label_repaired_rebaseline import dataset as base_data
from .phrase_end_auxiliary import ROOT,read,write,sha,normalizer,split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/motif_recurrence_study';ART=ROOT/'artifacts/motif_recurrence_study'
KINDS=('Z','U','O');COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids,kind):
    assert kind in KINDS;data=base_data(ids,'B')
    for pid,v in data.items():
        with np.load(ART/'cache'/f'{pid}.npz',allow_pickle=False) as z:feat=z['O' if kind=='O' else 'U'].copy()
        if kind=='Z':feat[:]=0
        assert feat.shape==(len(v['labels']),24)
        v['pitch_profiles']=feat
        v['curves'][...,34:]=feat[None]
    return data


def bind_engine():
    # score_novelty_study's base dataset uses its own directory, not this engine.
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1800.;engine.dataset=dataset


def prepare():
    began=time.monotonic()
    for p in (OUT,ART/'cache',ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(ROOT/'reports/note_relation_study/contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    assert read(ROOT/'reports/preaggregate_note_study/deterministic_supplement/audit.json')['status']=='supplement_complete_with_caveat'
    rows=[]
    for source in sorted((ROOT/'artifacts/note_relation_study/cache').glob('*.npz')):
        assert time.monotonic()-began<300
        with np.load(source,allow_pickle=False) as z:events=z['note_events'].copy();n=int(z['n_beats'])
        computed={k:features(events,n,k=='O') for k in ('U','O')};dest=ART/'cache'/source.name
        assert all(x.shape==(n,24) and np.isfinite(x).all() and ((x>=0)&(x<=1)).all() for x in computed.values())
        np.testing.assert_array_equal(computed['O'][:,3::4],computed['U'][:,3::4])
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                assert set(z.files)==set(computed)
                for k,v in computed.items():np.testing.assert_array_equal(z[k],v)
        else:np.savez_compressed(dest,**computed)
        hashes[str(source)]=sha(source);hashes[str(dest)]=sha(dest)
        rows.append(dict(piece_id=source.stem,beats=n,staves=len(np.unique(events[:,3])),ordered_bag_max_difference=float(abs(computed['O']-computed['U']).max()),ordered_bag_mean_difference=float(abs(computed['O']-computed['U']).mean()),availability=float(computed['O'][:,3::4].mean())))
    assert len(rows)==43;pd.DataFrame(rows).to_csv(OUT/'feature_audit.csv',index=False)
    # Compare all model inputs to the old baseline, and verify split opus isolation.
    checks=[]
    for f in (0,1):
        ids=split_ids(f);assert not set(ids['train'])&set(ids['validation'])
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        old=base_data(ids['train'],'B');norm_old=normalizer(old)
        for kind in KINDS:
            new=dataset(ids['train'],kind);norm_new=normalizer(new)
            np.testing.assert_array_equal(norm_new.mean[:34],norm_old.mean[:34]);np.testing.assert_array_equal(norm_new.std[:34],norm_old.std[:34])
            for pid,v in new.items():
                for field in ('labels','label_mask'):np.testing.assert_array_equal(v[field],old[pid][field])
                np.testing.assert_array_equal(v['curves'][...,:34],old[pid]['curves'][...,:34])
            checks.append(dict(fold=f,kind=kind,train=len(new),base_input_labels_masks_equal=True))
    pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    for p in (Path(__file__),ROOT/'src/motif_recurrence_features.py',ROOT/'tests/test_motif_recurrence_features.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==digest for r in done)
    write(OUT/'STATE.json',dict(status='ready',contract=digest,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None,prepare_seconds=time.monotonic()-began))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    cps=list((ART/'checkpoints').glob('*/*.pt'));assert len(cps)==24;before={str(p):sha(p) for p in cps};rows=[];checks=[]
    files=sorted((ART/'metrics').glob('*_fold*.json'));assert len(files)==12
    for rp in files:
        assert time.monotonic()-began<900
        r=read(rp);ids=split_ids(r['fold']);train=dataset(ids['train'],r['kind']);val=dataset(ids['validation'],r['kind']);norm=normalizer(train)
        cp=ART/'checkpoints'/r['run_id'];best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
        assert (best['contract'],best['kind'],best['fold'],best['seed'])==(contract['contract'],r['kind'],r['fold'],r['seed'])
        assert last['contract']==r['contract']==contract['contract'] and last['step']==300 and best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
        np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
        raw=checked_raw(pd.read_csv(ART/'metrics'/f"{r['run_id']}_predictions.csv.gz"),val);score=metrics(raw,val,r['threshold'])[2]
        assert max(abs(score[k]-r[k]) for k in COLS)<1e-10
        m=engine.make_model(r['kind'],r['seed']).cuda();m.load_state_dict(best['model']);replay=predictions(m,val,norm,torch.device('cuda'))
        error=max(float(abs(replay[p][k]-raw[p][k]).max()) for p in raw for k in raw[p]);assert error<2e-4
        assert max(abs(metrics(replay,val,r['threshold'])[2][k]-score[k]) for k in COLS)<1e-10
        ts=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
        checks.append(dict(run_id=r['run_id'],replay_error=error,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-score['macro_f1_tol1']));pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
        rows.append({k:v for k,v in r.items() if k!='history'});print('AUDITED',r['run_id'],flush=True)
    df=pd.DataFrame(rows);assert set(zip(df.kind,df.fold,df.seed))=={(k,f,s) for k in KINDS for f in (0,1) for s in (42,43)}
    df.to_csv(ART/'summary.csv',index=False);means=df.groupby('kind')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for reference in ('U','Z'):
        a=df[df.kind=='O'].set_index(['fold','seed']);b=df[df.kind==reference].set_index(['fold','seed']);d=a[COLS]-b[COLS];d.to_csv(OUT/f'delta_O_{reference}.csv')
        comparisons.append(dict(reference=reference,**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    assert before=={str(p):sha(p) for p in cps} and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',before);write(OUT/'completion_audit.json',dict(status='complete',runs=12,checkpoints=24,full_gpu_replays=12,norms_rebuilt=True,hashes_unchanged=True,test_predictions_accessed=False,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),training_validation_seconds=float(df.seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',contract=contract['contract'],pid=None));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','all','audit'),default='all');a=p.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    try:
        bind_engine()
        if a.stage=='audit':audit();return
        with threadpool_limits(2):digest=prepare()
        if a.stage=='prepare':print(digest,flush=True);return
        for f in (0,1):
            for s in (42,43):
                for k in KINDS:engine.train_one(k,f,s,digest)
        write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise


if __name__=='__main__':main()
