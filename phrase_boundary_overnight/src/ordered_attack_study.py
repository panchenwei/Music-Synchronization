"""Ordered attack slots versus within-beat averaging and zero-input controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from . import score_context_study as engine
from . import run_recurrence_depth_study as base
from .recurrence_depth_models import make_model as c3_model
from torch import nn
from . import run_halo_decoder_composition as dec

from .models import Normalizer
from .local_context_study import predictions
from .score_context_study import ROOT,read,write,sha,normalizer as base_normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior

OUT=ROOT/'reports/ordered_attack_study';ART=ROOT/'artifacts/ordered_attack_study';KINDS=('R','U','Z');COLS=base.COLS;PARAMS={'R':7777,'U':7777,'Z':7777}


class ArcBoundary(nn.Module):
    def __init__(self,kind,seed):
        super().__init__();assert kind in KINDS
        self.core=c3_model('C3',seed);old=self.core.input_projection
        with torch.random.fork_rng(devices=[]):new=nn.Linear(116,32)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:58].copy_(old.weight);new.bias.copy_(old.bias)
        self.core.input_projection=new
    def forward(self,x,padding_mask=None):
        return self.core(x,padding_mask=padding_mask)

def dataset(ids,kind):
    assert kind in KINDS
    data=base.dataset(ids,'C3')
    for pid,item in data.items():
        with np.load(ART/'cache'/f'{pid}.npz',allow_pickle=False) as z:extra=z['U' if kind=='U' else 'R'].copy()
        assert extra.shape==(len(item['labels']),58)
        if kind=='Z':extra[:]=0
        item['curves']=np.concatenate([item['curves'],np.broadcast_to(extra[None],(*item['curves'].shape[:2],58))],axis=-1)
    return data

def normalizer(data):
    old=base_normalizer({p:{**v,'curves':v['curves'][...,:58]} for p,v in data.items()})
    return Normalizer(np.r_[old.mean,np.zeros(58,np.float32)],np.r_[old.std,np.ones(58,np.float32)])


def prepare():
    for p in (OUT,ART/'cache',ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    from .ordered_attack_features import features
    assert read(ROOT/'reports/current_boundary_diagnosis/completion_audit.json')['status']=='complete'
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    cache_rows=[]
    for source in sorted((ROOT/'artifacts/note_relation_study/cache').glob('*.npz')):
        with np.load(source,allow_pickle=False) as z:events=z['note_events'].copy();n=int(z['n_beats'])
        r,u=features(events,n);dest=ART/'cache'/source.name
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                np.testing.assert_array_equal(r,z['R']);np.testing.assert_array_equal(u,z['U'])
        else:np.savez_compressed(dest,R=r,U=u)
        hashes[str(source)]=sha(source);hashes[str(dest)]=sha(dest)
        cache_rows.append(dict(piece_id=source.stem,beats=n,overflow_fraction=float(r[:,-2:].mean()),ordered_bag_difference=float(abs(r-u).mean())))
    assert len(cache_rows)==43
    pd.DataFrame(cache_rows).to_csv(OUT/'feature_audit.csv',index=False)
    for p in (Path(__file__),ROOT/'tests/test_ordered_attack_features.py',OUT/'PROTOCOL.md',ROOT/'src/ordered_attack_features.py',ROOT/'src/relation_interval_features.py',ROOT/'artifacts/interstart_decoder_study/summary.csv'):hashes[str(p)]=sha(p)
    checks=[]
    for f in (0,1):
        ids=split_ids(f);old=dataset(ids['train'],'Z');a=normalizer(old)
        for kind in KINDS:
            new=dataset(ids['train'],kind);b=normalizer(new)
            np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.std,b.std)
            for pid,item in new.items():
                np.testing.assert_array_equal(item['curves'][...,:58],old[pid]['curves'][...,:58])
                for key in ('labels','label_mask','performance_ids'):np.testing.assert_array_equal(item[key],old[pid][key])
            for seed in (42,43):
                sa=engine.CurvePieceBalancedSampler(old,a,64,32,seed);sb=engine.CurvePieceBalancedSampler(new,b,64,32,seed)
                for _ in range(3):
                    ba=sa.batch();bb=sb.batch();torch.testing.assert_close(ba[0][...,:58],bb[0][...,:58],atol=0,rtol=0)
                    for x,y in zip(ba[1:],bb[1:]):torch.testing.assert_close(x,y,atol=0,rtol=0)
            checks.append(dict(fold=f,kind=kind,norm_equal=True,old58_labels_sampling_equal=True))
        for seed in (42,43):
            run=f'C3_seed{seed}_fold{f}'
            for suffix in ('.json','_predictions.csv.gz'):
                p=base.ART/'metrics'/f'{run}{suffix}';hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==digest for r in done)
    write(OUT/'STATE.json',dict(status='ready',contract=digest,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');rows=[];checks=[];decoded=[]
    cp_hash={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(cp_hash)==24
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'R');val=dataset(ids['validation'],'R');norm=normalizer(train);prior=fit_prior(train)
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        write(OUT/f'prior_fold{f}.json',dict(prior=prior,train_ids=ids['train']))
        for s in (42,43):
            for oldkind,source in (('C3',base.ART),):
                run=f'{oldkind}_seed{s}_fold{f}';r=read(source/'metrics'/f'{run}.json')
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            for k in KINDS:
                train=dataset(ids['train'],k);val=dataset(ids['validation'],k)
                run=f'{k}_seed{s}_fold{f}';r=read(ART/'metrics'/f'{run}.json');cp=ART/'checkpoints'/run
                best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
                assert (best['contract'],best['kind'],best['fold'],best['seed'])==(contract['contract'],k,f,s)
                assert last['contract']==r['contract']==contract['contract'] and last['step']==300
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                m=ArcBoundary(k,s);m.load_state_dict(best['model'])
                assert r['params']==PARAMS[k]
                m.cuda().eval()
                replay=predictions(m,val,norm,torch.device('cuda'))
                err=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert err<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                ts=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
                checks.append(dict(run_id=run,replay_error=err,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']))
                rows.append({key:value for key,value in r.items() if key!='history'})
                for strength in (0.,.5):
                    row=dec.evaluate(raw,val,r['threshold'],prior,strength,f'{run}_lambda{strength:g}',contract['contract'])
                    if strength==0:assert max(abs(row[c]-r[c]) for c in COLS)<1e-10
                    decoded.append(dict(kind=k,fold=f,seed=s,**row))
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    neural=pd.DataFrame(rows);neural.to_csv(ART/'summary.csv',index=False)
    d=pd.DataFrame(decoded);assert len(d)==24;d.to_csv(ART/'decoder/summary.csv',index=False)
    ref=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');ref=ref[ref.strength.isin((0.,.5))].copy();ref['kind']='C3'
    allrows=pd.concat([d,ref],ignore_index=True);comparisons=[]
    allrows.groupby(['kind','strength'])[COLS].mean().to_csv(OUT/'means.csv')
    for candidate,control in (('R','C3'),('R','Z'),('R','U'),('U','Z'),('Z','C3')):
        for strength in (0.,.5):
            a=allrows[(allrows.kind==candidate)&(allrows.strength==strength)].set_index(['fold','seed'])
            b=allrows[(allrows.kind==control)&(allrows.strength==strength)].set_index(['fold','seed']);delta=a[COLS]-b[COLS]
            delta.to_csv(OUT/f'{candidate}_minus_{control}_lambda{strength:g}.csv')
            comparisons.append(dict(candidate=candidate,control=control,strength=strength,**delta.mean().to_dict(),f1_positive=int((delta.macro_f1_tol1>0).sum()),passed=bool(delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>=0 and (delta.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in cp_hash.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',cp_hash)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=12,full_gpu_replays=12,checkpoints=24,decoder_cells=24,old_metrics_recomputed=4,hashes_unchanged=True,norms_rebuilt=True,test_used=False,training_seconds=float(neural.seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'decoder_state/STATE.json',dict(status='complete',pid=None));write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']))
    print('COMPLETE\n'+allrows.groupby(['kind','strength'])[COLS].mean().to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1800.;engine.dataset=dataset;engine.make_model=ArcBoundary;engine.predictions=predictions;engine.normalizer=normalizer
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    for f in (0,1):
        for s in (42,43):
            for k in KINDS:
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent'],'Budget soft stop'
                engine.train_one(k,f,s,digest)
    write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
