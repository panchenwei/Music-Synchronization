"""Retrain matched N/R position controls; do not mutate historical T/C3 files."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import score_context_study as engine
from . import pure_transformer_reference as old
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .relative_position_models import make_model
from .score_context_study import ROOT,read,write,sha,normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior

OUT=ROOT/'reports/relative_position_study';ART=ROOT/'artifacts/relative_position_study'
KINDS=('N','R');COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']

def dataset(ids,kind):
    assert kind in KINDS
    return base.dataset(ids,'C3')

def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(old.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),ROOT/'src/relative_position_models.py',ROOT/'tests/test_relative_position_models.py',OUT/'PROTOCOL.md',old.ART/'decoder/summary.csv'):
        hashes[str(p)]=sha(p)
    for f in (0,1):
        ids=split_ids(f);a=dataset(ids['train'],'N');b=old.dataset(ids['train'],'T')
        na,nb=normalizer(a),normalizer(b)
        np.testing.assert_array_equal(na.mean,nb.mean);np.testing.assert_array_equal(na.std,nb.std)
        for p in a:
            for k in ('curves','labels','label_mask','performance_ids'):np.testing.assert_array_equal(a[p][k],b[p][k])
        for s in (42,43):
            ma=make_model('N',s);mb=old.make_model('T',s);mr=make_model('R',s)
            for k,v in mb.state_dict().items():
                torch.testing.assert_close(v,ma.state_dict()[k],atol=0,rtol=0)
                torch.testing.assert_close(v,mr.state_dict()[k],atol=0,rtol=0)
            sa=engine.CurvePieceBalancedSampler(a,na,64,32,s);sb=engine.CurvePieceBalancedSampler(b,nb,64,32,s)
            for _ in range(3):
                for x,y in zip(sa.batch(),sb.batch()):torch.testing.assert_close(x,y,atol=0,rtol=0)
            for kind,source in [('T',old.ART),('C3',base.ART)]:
                run=f'{kind}_seed{s}_fold{f}'
                for p in (source/'metrics'/f'{run}.json',source/'metrics'/f'{run}_predictions.csv.gz'):
                    hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0,pid=None,contract=digest))
    write(OUT/'input_audit.json',dict(input_dimensions=58,params=19073,initial_parameters_equal=True,normalizers_equal=True,sampling_equal=True,new_features=False,test_used=False))
    return digest

def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');checks=[];rows=[];decoded=[]
    cp_hash={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(cp_hash)==16
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'N');val=dataset(ids['validation'],'N');norm=normalizer(train);prior=fit_prior(train)
        for s in (42,43):
            for kind,source in [('T',old.ART),('C3',base.ART)]:
                r=read(source/'metrics'/f'{kind}_seed{s}_fold{f}.json')
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{kind}_seed{s}_fold{f}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            for kind in KINDS:
                run=f'{kind}_seed{s}_fold{f}';r=read(ART/'metrics'/f'{run}.json');cp=ART/'checkpoints'/run
                best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
                assert (best['kind'],best['fold'],best['seed'])==(kind,f,s)
                assert best['contract']==last['contract']==r['contract']==contract['contract'] and last['step']==300
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                m=make_model(kind,s);m.load_state_dict(best['model']);m.cuda().eval()
                replay=predictions(m,val,norm,torch.device('cuda'))
                err=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert err<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                tr=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
                checks.append(dict(run=run,replay_error=err,train_f1=tr['macro_f1_tol1'],gap=tr['macro_f1_tol1']-r['macro_f1_tol1']))
                rows.append(r)
                for strength in (0.,.5):
                    v=dec.evaluate(raw,val,r['threshold'],prior,strength,f'{run}_lambda{strength:g}',contract['contract'])
                    if strength==0:assert max(abs(v[c]-r[c]) for c in COLS)<1e-10
                    decoded.append(dict(kind=kind,fold=f,seed=s,**v))
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(ART/'summary.csv',index=False)
    d=pd.DataFrame(decoded);assert len(d)==16;d.to_csv(ART/'decoder/summary.csv',index=False)
    t=pd.read_csv(old.ART/'decoder/summary.csv')
    c=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');c=c[c.strength.isin([0,.5])].copy();c['kind']='C3'
    allrows=pd.concat([d,t,c],ignore_index=True);allrows.to_csv(OUT/'all_runs.csv',index=False)
    means=allrows.groupby(['kind','strength'])[COLS].mean();means.to_csv(OUT/'means.csv')
    comp=[]
    for a,b in [('N','T'),('R','T'),('R','N'),('N','C3'),('R','C3')]:
        for strength in (0.,.5):
            left=allrows[(allrows.kind==a)&(allrows.strength==strength)].set_index(['fold','seed'])
            right=allrows[(allrows.kind==b)&(allrows.strength==strength)].set_index(['fold','seed'])
            delta=left[COLS]-right[COLS];delta.to_csv(OUT/f'{a}_minus_{b}_lambda{strength:g}.csv')
            passed=delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>=0 and delta.raw_ap.mean()>=0 and (delta.macro_f1_tol1>0).sum()>=3
            comp.append(dict(candidate=a,control=b,strength=strength,**delta.mean().to_dict(),positive=int((delta.macro_f1_tol1>0).sum()),passed=bool(passed)))
    write(OUT/'comparisons.json',comp)
    assert all(sha(p)==h for p,h in cp_hash.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',cp_hash)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,gpu_replays=8,checkpoints=16,decoder_cells=16,old_metrics_recomputed=8,hashes_unchanged=True,norms_rebuilt=True,test_used=False,training_seconds=float(frame.seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None))
    print('COMPLETE\n'+means.to_string(),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=3600.;engine.dataset=dataset;engine.make_model=make_model;engine.normalizer=normalizer;engine.predictions=predictions
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    for f in (0,1):
        for s in (42,43):
            for kind in KINDS:
                g=read(ROOT/'reports/research_resource_guard.json');assert g['observed_used_percent']<g['post_reset_stop_used_percent']
                engine.train_one(kind,f,s,digest)
    audit()

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
