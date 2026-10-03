"""Matched absolute/estimated-key-relative additions to frozen C3 inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch,music21
from torch import nn
from . import score_context_study as engine
from . import run_recurrence_depth_study as base
from .score_context_study import ROOT,read,write,sha,normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw
from .recurrence_tonal_features import features

OUT=ROOT/'reports/recurrence_tonal_study';ART=ROOT/'artifacts/recurrence_tonal_study'
KINDS=('TA','TR');COLS=base.COLS


def make_model(kind,seed):
    assert kind in KINDS
    m=base.make_model('C3',seed);old=m.input_projection
    with torch.random.fork_rng(devices=[]):new=nn.Linear(84,32)
    with torch.no_grad():
        new.weight.zero_();new.weight[:,:58].copy_(old.weight);new.bias.copy_(old.bias)
    m.input_projection=new
    return m


def dataset(ids,kind):
    assert kind in KINDS
    data=base.dataset(ids,'C3')
    for pid,v in data.items():
        with np.load(ART/'cache'/f'{pid}.npz') as a:extra=a[kind]
        v['pitch_profiles']=np.c_[v['pitch_profiles'],extra]
        v['curves']=np.concatenate([v['curves'],np.broadcast_to(extra,(*v['curves'].shape[:2],26))],axis=-1)
    return data


def prepare():
    for p in (OUT,ART/'cache',ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    ids=sorted(set(p for f in (0,1) for role in ('train','validation') for p in split_ids(f)[role]));keys=[]
    for pid in ids:
        source=ROOT/'artifacts/note_relation_study/cache'/f'{pid}.npz'
        with np.load(source) as a:events=a['note_events'].copy()
        with np.load(ROOT/'artifacts/slice_energy_study/cache'/f'{pid}.npz') as a:n=len(a['start_labels'])
        # Only score events and sequence length enter feature extraction.
        absolute,relative,key=features(events,n);path=ART/'cache'/f'{pid}.npz'
        if not path.exists():np.savez_compressed(path,TA=absolute,TR=relative)
        with np.load(path) as a:
            np.testing.assert_array_equal(a['TA'],absolute);np.testing.assert_array_equal(a['TR'],relative)
        assert absolute.shape==relative.shape==(n,26) and np.isfinite(relative).all()
        keys.append(dict(piece_id=pid,**key));hashes[str(source)]=sha(source);hashes[str(path)]=sha(path)
    pd.DataFrame(keys).to_csv(OUT/'estimated_keys.csv',index=False)
    for f in (0,1):
        old=base.dataset(split_ids(f)['train'],'C3');orig=normalizer(old)
        for k in KINDS:
            data=dataset(split_ids(f)['train'],k);norm=normalizer(data)
            np.testing.assert_array_equal(norm.mean[:58],orig.mean);np.testing.assert_array_equal(norm.std[:58],orig.std)
            for pid,v in data.items():
                np.testing.assert_array_equal(v['curves'][...,:58],old[pid]['curves'])
                for field in ('labels','label_mask','performance_ids'):np.testing.assert_array_equal(v[field],old[pid][field])
        for seed in (42,43):
            run=f'C3_seed{seed}_fold{f}'
            for p in (base.ART/'metrics'/f'{run}.json',base.ART/'metrics'/f'{run}_predictions.csv.gz',base.ART/'checkpoints'/run/'best.pt'):hashes[str(p)]=sha(p)
    for p in (Path(__file__),ROOT/'src/recurrence_tonal_features.py',ROOT/'tests/test_recurrence_tonal.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes,music21_version=music21.__version__))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',seconds=0.,completed=[],pid=None,contract=digest))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');rows=[];checks=[]
    cp_hash={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(cp_hash)==16
    for kind in KINDS:
        for fold in (0,1):
            ids=split_ids(fold);train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind);norm=normalizer(train)
            for seed in (42,43):
                assert time.monotonic()-began<900
                run=f'{kind}_seed{seed}_fold{fold}';r=read(ART/'metrics'/f'{run}.json');cp=ART/'checkpoints'/run
                best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
                assert (best['contract'],best['kind'],best['fold'],best['seed'])==(contract['contract'],kind,fold,seed)
                assert last['contract']==r['contract']==contract['contract'] and last['step']==300
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val);score=metrics(raw,val,r['threshold'])[2]
                assert max(abs(score[c]-r[c]) for c in COLS)<1e-10
                m=make_model(kind,seed).cuda();m.load_state_dict(best['model']);assert r['params']==6753
                learned=float(m.input_projection.weight[:,58:].detach().abs().sum());assert learned>0
                replay=predictions(m,val,norm,torch.device('cuda'));err=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert err<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                ts=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
                rows.append(r);checks.append(dict(run_id=run,replay_error=err,added_weights_l1=learned,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']))
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    for f in (0,1):
        val=base.dataset(split_ids(f)['validation'],'C3')
        for s in (42,43):
            run=f'C3_seed{s}_fold{f}';r=read(base.ART/'metrics'/f'{run}.json');raw=checked_raw(pd.read_csv(base.ART/'metrics'/f'{run}_predictions.csv.gz'),val)
            assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            rows.append(r)
    frame=pd.DataFrame(rows);assert len(frame)==12;frame.to_csv(ART/'summary.csv',index=False);frame.groupby('kind')[COLS].mean().to_csv(OUT/'means.csv');comparisons=[]
    for k,ref in (('TA','C3'),('TR','C3'),('TR','TA')):
        a=frame[frame.kind==k].set_index(['fold','seed']);b=frame[frame.kind==ref].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        d.to_csv(OUT/f'delta_{k}_{ref}.csv')
        passed=d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3
        comparisons.append(dict(kind=k,reference=ref,**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),passed=bool(passed)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in cp_hash.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',cp_hash)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,full_gpu_replays=8,checkpoints=16,reused_cnn_metric_recomputations=4,hashes_unchanged=True,norms_rebuilt=True,test_used=False,training_seconds=float(frame[frame.kind.isin(KINDS)].seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']))


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1800.;engine.dataset=dataset;engine.make_model=make_model
    contract=prepare()
    for f in (0,1):
        for s in (42,43):
            for k in KINDS:engine.train_one(k,f,s,contract)
    write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
