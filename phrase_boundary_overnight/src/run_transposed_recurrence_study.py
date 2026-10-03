"""One fixed-shift recurrence candidate, reusing the frozen ordered controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import score_context_study as engine
from . import run_motif_recurrence_study as prior
from .transposed_recurrence import features
from .phrase_end_auxiliary import ROOT, read, write, sha, normalizer, split_ids
from .local_context_study import metrics, predictions
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/transposed_recurrence_study'; ART=ROOT/'artifacts/transposed_recurrence_study'
OLD=prior.ART; COLS=prior.COLS


def dataset(ids, kind):
    assert kind in ('O','T')
    data=prior.dataset(ids,'O')
    if kind=='T':
        for pid,v in data.items():
            feat=np.load(ART/'cache'/f'{pid}.npy',allow_pickle=False)
            assert feat.shape==(len(v['labels']),24)
            v['pitch_profiles']=feat; v['curves'][...,34:]=feat[None]
    return data


def bind():
    engine.OUT=OUT; engine.ART=ART; engine.CAP=900.; engine.dataset=dataset
    # O/T are both narrow in the unchanged model factory; assert in prepare/audit.


def prepare():
    began=time.monotonic()
    for p in (OUT,ART/'cache',ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/recurrence_confound_study/completion_audit.json')['status']=='complete'
    hashes=dict(read(prior.OUT/'contract.json')['hashes'])
    assert all(sha(p)==h for p,h in hashes.items())
    for f in (0,1):
        for s in (42,43):
            run=f'O_seed{s}_fold{f}'
            for p in (OLD/'metrics'/f'{run}.json',OLD/'metrics'/f'{run}_predictions.csv.gz',
                      OLD/'checkpoints'/run/'best.pt',OLD/'checkpoints'/run/'latest.pt'):hashes[str(p)]=sha(p)
    rows=[]
    for source in sorted((ROOT/'artifacts/note_relation_study/cache').glob('*.npz')):
        assert time.monotonic()-began<600
        with np.load(source,allow_pickle=False) as z:events=z['note_events'].copy();n=int(z['n_beats'])
        with np.load(OLD/'cache'/source.name,allow_pickle=False) as z:o=z['O'].copy()
        zero=features(events,n,(0,));np.testing.assert_array_equal(zero,o)
        dest=ART/'cache'/f'{source.stem}.npy'; t=features(events,n)
        assert t.shape==(n,24) and np.isfinite(t).all() and ((t>=0)&(t<=1)).all()
        np.testing.assert_array_equal(t[:,3::4],o[:,3::4]);assert (t>=o-1e-7).all()
        if dest.exists():np.testing.assert_array_equal(np.load(dest,allow_pickle=False),t)
        else:np.save(dest,t)
        hashes[str(dest)]=sha(dest)
        rows.append(dict(piece_id=source.stem,beats=n,zero_shift_reproduces_O=True,
                         shift_gain_mean=float((t-o).mean()),shift_gain_max=float((t-o).max())))
    assert len(rows)==43;pd.DataFrame(rows).to_csv(OUT/'feature_audit.csv',index=False)
    checks=[]
    for f in (0,1):
        ids=split_ids(f);assert not set(ids['train'])&set(ids['validation'])
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        old=dataset(ids['train'],'O');new=dataset(ids['train'],'T');a=normalizer(old);b=normalizer(new)
        np.testing.assert_array_equal(a.mean[:34],b.mean[:34]);np.testing.assert_array_equal(a.std[:34],b.std[:34])
        for pid,v in new.items():
            for k in ('labels','label_mask'):np.testing.assert_array_equal(v[k],old[pid][k])
            np.testing.assert_array_equal(v['curves'][...,:34],old[pid]['curves'][...,:34])
        checks.append(dict(fold=f,base_inputs_labels_masks_equal=True,opus_disjoint=True))
    for s in (42,43):
        a=engine.make_model('O',s);b=engine.make_model('T',s)
        assert not b.frontend.wide and sum(p.numel() for p in b.parameters())==3297
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    for p in (Path(__file__),ROOT/'src/transposed_recurrence.py',ROOT/'tests/test_transposed_recurrence.py',OUT/'PROTOCOL.md'):
        hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==digest for r in done)
    write(OUT/'STATE.json',dict(status='ready',contract=digest,pid=None,completed=[r['run_id'] for r in done],
          seconds=sum(r['seconds'] for r in done),prepare_seconds=time.monotonic()-began))
    return digest


def audit():
    began=time.monotonic(); contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    rows=[];checks=[];hashes={}
    for k in ('O','T'):
        source=OLD if k=='O' else ART
        expected=read(prior.OUT/'contract.json')['contract'] if k=='O' else contract['contract']
        for f in (0,1):
            for s in (42,43):
                assert time.monotonic()-began<600
                run=f'{k}_seed{s}_fold{f}';r=read(source/'metrics'/f'{run}.json');ids=split_ids(f)
                train=dataset(ids['train'],k);val=dataset(ids['validation'],k);norm=normalizer(train)
                cp=source/'checkpoints'/run
                for p in (cp/'best.pt',cp/'latest.pt'):hashes[str(p)]=sha(p)
                best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False)
                last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
                assert (best['contract'],best['kind'],best['fold'],best['seed'])==(expected,k,f,s)
                assert last['contract']==r['contract']==expected and last['step']==300
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'),val)
                score=metrics(raw,val,r['threshold'])[2];assert max(abs(score[c]-r[c]) for c in COLS)<1e-10
                m=engine.make_model(k,s).cuda();m.load_state_dict(best['model'])
                assert not m.frontend.wide and r['params']==3297
                replay=predictions(m,val,norm,torch.device('cuda'))
                error=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert error<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-score[c]) for c in COLS)<1e-10
                checks.append(dict(run_id=run,replay_error=error,reused=k=='O'));rows.append(r)
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    assert len(hashes)==16 and all(sha(p)==h for p,h in hashes.items())
    assert all(sha(p)==h for p,h in contract['hashes'].items())
    df=pd.DataFrame(rows);df.to_csv(ART/'summary.csv',index=False)
    means=df.groupby('kind')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean();means.to_csv(OUT/'means.csv')
    a=df[df.kind=='T'].set_index(['fold','seed']);b=df[df.kind=='O'].set_index(['fold','seed']);d=a[COLS]-b[COLS]
    d.to_csv(OUT/'delta_T_O.csv');write(OUT/'comparison.json',dict(**d.mean().to_dict(),
          f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),
          promotion=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0
                         and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=4,reused_runs=4,full_gpu_replays=8,
          checkpoints=16,hashes_unchanged=True,norms_rebuilt=True,test_predictions_accessed=False,
          deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
          new_training_validation_seconds=float(df[df.kind=='T'].seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',contract=contract['contract'],pid=None));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','all','audit'),default='all');args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    try:
        bind()
        if args.stage=='audit':audit();return
        with threadpool_limits(2):digest=prepare()
        if args.stage=='prepare':print(digest,flush=True);return
        for f in (0,1):
            for s in (42,43):engine.train_one('T',f,s,digest)
        write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise


if __name__=='__main__':main()
