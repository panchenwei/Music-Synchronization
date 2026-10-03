"""Frozen axis-split versus joint-kernel music study, isolated from old runs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from . import score_roll_study as engine
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .axis_split_music import AxisBoundary,predictions
from .score_piano_roll import piano_roll
from .score_context_study import ROOT,read,write,sha,normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior

OUT=ROOT/'reports/axis_split_music_study';ART=ROOT/'artifacts/axis_split_music_study'
KINDS=('S','J');COLS=base.COLS


def dataset(ids,kind):
    assert kind in KINDS
    data=base.dataset(ids,'C3')
    for pid,item in data.items():
        item['piano_roll']=np.load(ART/'cache'/f'{pid}.npy',allow_pickle=False)
        assert len(item['piano_roll'])==len(item['labels'])
    return data


def prepare():
    for p in (OUT,ART/'cache',ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    hashes.update(read(ROOT/'reports/note_relation_study/contract.json')['hashes'])
    assert all(sha(p)==h for p,h in hashes.items())
    ids=sorted({p for f in (0,1) for split in ('train','validation') for p in split_ids(f)[split]})
    audits=[]
    for pid in ids:
        source=ROOT/'artifacts/note_relation_study/cache'/f'{pid}.npz'
        with np.load(source,allow_pickle=False) as z:
            roll=piano_roll(z['note_events'],int(z['n_beats'])); count=len(z['note_events'])
        dest=ART/'cache'/f'{pid}.npy'
        if not dest.exists():np.save(dest,roll)
        np.testing.assert_array_equal(roll,np.load(dest,allow_pickle=False))
        hashes[str(source)]=sha(source);hashes[str(dest)]=sha(dest)
        audits.append(dict(piece_id=pid,beats=len(roll),events=count,occupancy_sum=float(roll[:,0].sum()),onset_cells=float(roll[:,1].sum())))
    pd.DataFrame(audits).to_csv(OUT/'feature_audit.csv',index=False)
    for f in (0,1):
        split=split_ids(f); assert not set(split['train'])&set(split['validation'])
        frame=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=frame[frame.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        old=base.dataset(split['train'],'C3');new=dataset(split['train'],'S')
        for pid in old:
            for key in ('curves','labels','label_mask','pitch_profiles','performance_ids'):np.testing.assert_array_equal(old[pid][key],new[pid][key])
        for s in (42,43):
            run=f'C3_seed{s}_fold{f}'
            for suffix in ('.json','_predictions.csv.gz'):
                p=base.ART/'metrics'/f'{run}{suffix}';hashes[str(p)]=sha(p)
    for p in (Path(__file__),ROOT/'src/axis_split_music.py',ROOT/'tests/test_axis_split_music.py',Path(engine.__file__),Path(dec.__file__),ROOT/'src/interstart_decoder.py',ROOT/'src/score_roll_branch.py',ROOT/'src/score_piano_roll.py',OUT/'PROTOCOL.md',ROOT/'artifacts/interstart_decoder_study/summary.csv'):
        hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==digest for r in done)
    write(OUT/'STATE.json',dict(status='ready',contract=digest,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None))
    write(OUT/'architecture_audit.json',{k:dict(params=sum(p.numel() for p in AxisBoundary(k,42).parameters()),all_parameters_trainable=True) for k in KINDS})
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');rows=[];checks=[];decoded=[]
    cp_hash={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(cp_hash)==16
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'S');val=dataset(ids['validation'],'S');norm=normalizer(train);prior=fit_prior(train)
        write(OUT/f'prior_fold{f}.json',dict(prior=prior,train_ids=ids['train']))
        for s in (42,43):
            oldrun=f'C3_seed{s}_fold{f}';old=read(base.ART/'metrics'/f'{oldrun}.json')
            oldraw=checked_raw(pd.read_csv(base.ART/'metrics'/f'{oldrun}_predictions.csv.gz'),val)
            assert max(abs(metrics(oldraw,val,old['threshold'])[2][c]-old[c]) for c in COLS)<1e-10
            for k in KINDS:
                run=f'{k}_seed{s}_fold{f}';r=read(ART/'metrics'/f'{run}.json');cp=ART/'checkpoints'/run
                best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
                assert (best['contract'],best['kind'],best['fold'],best['seed'])==(contract['contract'],k,f,s)
                assert last['contract']==r['contract']==contract['contract'] and last['step']==300
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                m=AxisBoundary(k,s);initial=m.stem.input.weight.detach().clone();m.load_state_dict(best['model'])
                stem_changed=not torch.equal(initial,m.stem.input.weight);assert stem_changed
                m.cuda();replay=predictions(m,val,norm,torch.device('cuda'))
                err=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert err<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                ts=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
                checks.append(dict(run_id=run,replay_error=err,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1'],stem_changed=stem_changed))
                rows.append({key:value for key,value in r.items() if key!='history'})
                for strength in (0.,.5):
                    row=dec.evaluate(raw,val,r['threshold'],prior,strength,f'{run}_lambda{strength:g}',contract['contract'])
                    if strength==0:assert max(abs(row[c]-r[c]) for c in COLS)<1e-10
                    decoded.append(dict(kind=k,fold=f,seed=s,**row))
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    neural=pd.DataFrame(rows);neural.to_csv(ART/'summary.csv',index=False)
    d=pd.DataFrame(decoded);assert len(d)==16;d.to_csv(ART/'decoder/summary.csv',index=False)
    d.groupby(['kind','strength'])[COLS].mean().to_csv(OUT/'means.csv')
    ref=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');ref=ref[ref.strength.isin((0.,.5))].copy();ref['kind']='C3'
    allrows=pd.concat([d,ref],ignore_index=True);comparisons=[]
    for candidate,control in (('S','C3'),('J','C3'),('S','J')):
        for strength in (0.,.5):
            a=allrows[(allrows.kind==candidate)&(allrows.strength==strength)].set_index(['fold','seed'])
            b=allrows[(allrows.kind==control)&(allrows.strength==strength)].set_index(['fold','seed']);delta=a[COLS]-b[COLS]
            delta.to_csv(OUT/f'{candidate}_minus_{control}_lambda{strength:g}.csv')
            comparisons.append(dict(candidate=candidate,control=control,strength=strength,**delta.mean().to_dict(),f1_positive=int((delta.macro_f1_tol1>0).sum()),passed=bool(delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>=0 and (delta.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in cp_hash.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',cp_hash)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,full_gpu_replays=8,checkpoints=16,decoder_cells=16,old_c3_metrics_recomputed=4,hashes_unchanged=True,norms_rebuilt=True,test_used=False,training_seconds=float(neural.seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'decoder_state/STATE.json',dict(status='complete',pid=None));write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']))
    print('COMPLETE\n'+allrows.groupby(['kind','strength'])[COLS].mean().to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=5400.;engine.dataset=dataset;engine.RollBoundary=AxisBoundary;engine.predictions=predictions
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    for f in (0,1):
        for s in (42,43):
            for k in KINDS:
                guard=read(ROOT/'reports/research_resource_guard.json')
                assert guard['observed_used_percent']<guard['post_reset_stop_used_percent'],'Budget soft stop; checkpoints retained'
                engine.train_one(k,f,s,digest)
    write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
