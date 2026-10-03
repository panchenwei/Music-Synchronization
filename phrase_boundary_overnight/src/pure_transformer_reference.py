"""Same-input pure Transformer reference; historical C3 outputs remain frozen."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from . import score_context_study as engine
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .phase7_models import Phase7BoundaryModel
from .models import seed_everything
from .score_context_study import ROOT, read, write, sha, normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics, predictions
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior

OUT=ROOT/'reports/pure_transformer_reference'
ART=ROOT/'artifacts/pure_transformer_reference'
COLS=base.COLS

def make_model(kind, seed):
    assert kind=='T'
    seed_everything(seed)
    m=Phase7BoundaryModel('A',input_dim=58,d_model=32,layers=2,heads=4,ffn_dim=64,dropout=.2)
    assert m.frontend is None and len(m.blocks)==2
    assert not any(isinstance(x,(nn.Conv1d,nn.Conv2d,nn.GRU,nn.LSTM)) for x in m.modules())
    return m

def dataset(ids,kind):
    assert kind=='T'
    return base.dataset(ids,'C3')

def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_pure_transformer_reference.py',ROOT/'artifacts/interstart_decoder_study/summary.csv'):
        hashes[str(p)]=sha(p)
    checks=[]
    for f in (0,1):
        ids=split_ids(f)
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        a=dataset(ids['train'],'T');b=base.dataset(ids['train'],'C3')
        na,nb=normalizer(a),normalizer(b)
        np.testing.assert_array_equal(na.mean,nb.mean);np.testing.assert_array_equal(na.std,nb.std)
        for pid in a:
            for key in ('curves','labels','label_mask','performance_ids'):np.testing.assert_array_equal(a[pid][key],b[pid][key])
        for s in (42,43):
            sa=engine.CurvePieceBalancedSampler(a,na,64,32,s);sb=engine.CurvePieceBalancedSampler(b,nb,64,32,s)
            for _ in range(3):
                for x,y in zip(sa.batch(),sb.batch()):torch.testing.assert_close(x,y,atol=0,rtol=0)
            run=f'C3_seed{s}_fold{f}'
            for suffix in ('.json','_predictions.csv.gz'):
                p=base.ART/'metrics'/f'{run}{suffix}';hashes[str(p)]=sha(p)
        checks.append(dict(fold=f,input_label_normalizer_sampling_equal=True,opus_disjoint=True))
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0,pid=None,contract=digest))
    write(OUT/'architecture.json',dict(kind='pure Transformer',input_dim=58,d_model=32,layers=2,heads=4,ffn_dim=64,params=sum(p.numel() for p in make_model('T',42).parameters()),conv=False,position='sinusoidal',training_steps=300,training_window=64,inference='whole work'))
    return digest

def audit():
    start=time.monotonic();contract=read(OUT/'contract.json');checks=[];decoded=[];rows=[]
    cp_hash={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(cp_hash)==8
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'T');val=dataset(ids['validation'],'T');norm=normalizer(train);prior=fit_prior(train)
        write(OUT/f'prior_fold{f}.json',dict(prior=prior,train_ids=ids['train']))
        for s in (42,43):
            old=f'C3_seed{s}_fold{f}';r0=read(base.ART/'metrics'/f'{old}.json')
            raw0=checked_raw(pd.read_csv(base.ART/'metrics'/f'{old}_predictions.csv.gz'),val)
            assert max(abs(metrics(raw0,val,r0['threshold'])[2][c]-r0[c]) for c in COLS)<1e-10
            run=f'T_seed{s}_fold{f}';r=read(ART/'metrics'/f'{run}.json');cp=ART/'checkpoints'/run
            best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
            assert (best['kind'],best['fold'],best['seed'])==('T',f,s)
            assert best['contract']==last['contract']==r['contract']==contract['contract'] and last['step']==300
            assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
            np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
            raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val)
            assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            m=make_model('T',s);m.load_state_dict(best['model']);m.cuda().eval()
            assert sum(p.numel() for p in m.parameters())==r['params']
            replay=predictions(m,val,norm,torch.device('cuda'))
            err=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert err<2e-4
            assert max(abs(metrics(replay,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            tr=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
            checks.append(dict(run_id=run,replay_error=err,train_f1=tr['macro_f1_tol1'],gap=tr['macro_f1_tol1']-r['macro_f1_tol1']))
            rows.append(r)
            for strength in (0.,.5):
                d=dec.evaluate(raw,val,r['threshold'],prior,strength,f'{run}_lambda{strength:g}',contract['contract'])
                if strength==0:assert max(abs(d[c]-r[c]) for c in COLS)<1e-10
                decoded.append(dict(kind='T',fold=f,seed=s,**d))
            pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(ART/'summary.csv',index=False)
    d=pd.DataFrame(decoded);assert len(d)==8;d.to_csv(ART/'decoder/summary.csv',index=False)
    ref=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');ref=ref[ref.strength.isin((0.,.5))].copy();ref['kind']='C3'
    allrows=pd.concat([d,ref],ignore_index=True);means=allrows.groupby(['kind','strength'])[COLS].mean();means.to_csv(OUT/'means.csv')
    comparisons=[]
    for strength in (0.,.5):
        a=allrows[(allrows.kind=='C3')&(allrows.strength==strength)].set_index(['fold','seed'])
        b=allrows[(allrows.kind=='T')&(allrows.strength==strength)].set_index(['fold','seed'])
        delta=a[COLS]-b[COLS];delta.to_csv(OUT/f'C3_minus_T_lambda{strength:g}.csv')
        comparisons.append(dict(candidate='C3',control='T',strength=strength,**delta.mean().to_dict(),f1_positive=int((delta.macro_f1_tol1>0).sum())))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in cp_hash.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',cp_hash)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=4,full_gpu_replays=4,checkpoints=8,decoder_cells=8,old_metrics_recomputed=4,hashes_unchanged=True,norms_rebuilt=True,test_used=False,training_seconds=float(frame.seconds.sum()),audit_seconds=time.monotonic()-start))
    write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']))
    write(OUT/'decoder_state/STATE.json',dict(status='complete',pid=None))
    print('COMPLETE\n'+means.to_string(),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1800.;engine.dataset=dataset;engine.make_model=make_model;engine.normalizer=normalizer;engine.predictions=predictions
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    for f in (0,1):
        for s in (42,43):
            guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
            engine.train_one('T',f,s,digest)
    audit()

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
