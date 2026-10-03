"""Current representation: matched BiGRU and preregistered CNN/GRU fusion."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import score_context_study as engine
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .current_gru_models import make_model
from .score_context_study import ROOT, read, write, sha, normalizer, GRID
from .three_round_round2 import split_ids
from .local_context_study import metrics, predictions
from .phase2_models import choose_single_threshold
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities
from .context_inference_audit_v2 import save_raw, error

OUT=ROOT/'reports/current_gru_study'; ART=ROOT/'artifacts/current_gru_study'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids, kind):
    assert kind=='G'
    return base.dataset(ids, 'C3')


def prepare():
    for p in (OUT, ART/'metrics', ART/'checkpoints', ART/'decoder', ART/'fusion'):
        p.mkdir(parents=True, exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    files=[Path(__file__), ROOT/'src/current_gru_models.py', ROOT/'tests/test_current_gru.py', OUT/'PROTOCOL.md',
           ROOT/'src/recurrence_message_probe.py', ROOT/'src/recurrence_mean_control.py', ROOT/'src/interstart_decoder.py',
           ROOT/'src/run_halo_decoder_composition.py', ROOT/'src/context_inference_audit_v2.py']
    files+=list((ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'))
    checks=[]
    for f in (0,1):
        ids=split_ids(f);a=dataset(ids['train'],'G');b=base.dataset(ids['train'],'C3')
        na=normalizer(a);nb=normalizer(b)
        np.testing.assert_array_equal(na.mean,nb.mean);np.testing.assert_array_equal(na.std,nb.std)
        for pid in a:
            for k in ('curves','labels','label_mask','performance_ids'):
                np.testing.assert_array_equal(a[pid][k],b[pid][k])
        for s in (42,43):
            sa=engine.CurvePieceBalancedSampler(a,na,64,32,s);sb=engine.CurvePieceBalancedSampler(b,nb,64,32,s)
            for _ in range(3):
                for x,y in zip(sa.batch(),sb.batch()):torch.testing.assert_close(x,y,atol=0,rtol=0)
            for suffix in ('.json','_predictions.csv.gz'):
                files.append(base.ART/'metrics'/f'C3_seed{s}_fold{f}{suffix}')
        checks.append(dict(fold=f,inputs_labels_norm_sampler_equal=True,train_ids=ids['train'],validation_ids=ids['validation']))
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));write(OUT/'input_audit.json',checks)
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');checks=[];rows=[]
    cp_hash={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(cp_hash)==8
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'G');val=dataset(ids['validation'],'G');norm=normalizer(train);prior=fit_prior(train)
        for s in (42,43):
            assert time.monotonic()-began<1200
            name=f'G_seed{s}_fold{f}';r=read(ART/'metrics'/f'{name}.json')
            best=torch.load(ART/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
            last=torch.load(ART/'checkpoints'/name/'latest.pt',map_location='cpu',weights_only=False)
            assert best['contract']==last['contract']==r['contract']==contract['contract'] and last['step']==300
            assert (best['kind'],best['seed'],best['fold'])==('G',s,f)
            assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
            np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
            raw=checked_raw(pd.read_csv(ART/'metrics'/f'{name}_predictions.csv.gz'),val)
            assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            m=make_model('G',s).cuda();m.load_state_dict(best['model'])
            replay=predictions(m,val,norm,torch.device('cuda'));err=error(raw,replay);assert err<2e-4
            assert max(abs(metrics(replay,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            ts=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
            cmeta=read(base.ART/'metrics'/f'C3_seed{s}_fold{f}.json')
            craw=checked_raw(pd.read_csv(base.ART/'metrics'/f'C3_seed{s}_fold{f}_predictions.csv.gz'),val)
            assert max(abs(metrics(craw,val,cmeta['threshold'])[2][c]-cmeta[c]) for c in COLS)<1e-10
            eraw={p:{q:(a+craw[p][q])*.5 for q,a in pp.items()} for p,pp in raw.items()}
            ethreshold,_=choose_single_threshold(eraw,val,GRID)
            path=ART/'fusion'/f'E_seed{s}_fold{f}_predictions.csv.gz';save_raw(eraw,path,val)
            assert error(eraw,checked_raw(pd.read_csv(path),val))<1e-12
            for k,x,threshold in [('C3',craw,cmeta['threshold']),('G',raw,r['threshold']),('E',eraw,ethreshold)]:
                adjusted={}
                for pid,pp in x.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[pid]={q:mix_probabilities(a,g) for q,a in pp.items()}
                for policy,rr,strength in [('raw',x,0.),('B10',x,1.),('M10',adjusted,1.)]:
                    v=dec.evaluate(rr,val,threshold,prior,strength,f'{k}_seed{s}_fold{f}_{policy}',contract['contract'])
                    if policy=='raw':assert max(abs(v[c]-metrics(x,val,threshold)[2][c]) for c in COLS)<1e-10
                    rows.append(dict(kind=k,fold=f,seed=s,policy=policy,**v))
            checks.append(dict(run=name,replay_error=err,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']))
            pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
            pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False)
            print('AUDITED',name,flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==36
    means=frame.groupby(['kind','policy'])[COLS+['macro_precision_tol1','macro_recall_tol1']].mean();means.to_csv(OUT/'means.csv')
    comparisons=[]
    for k in ('G','E'):
        for policy in ('raw','B10','M10'):
            a=frame[(frame.kind==k)&(frame.policy==policy)].set_index(['fold','seed'])
            b=frame[(frame.kind=='C3')&(frame.policy==policy)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
            comparisons.append(dict(kind=k,policy=policy,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),
                                   passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in cp_hash.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',cp_hash)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=4,gpu_replays=4,checkpoints=8,
          decoded_cells=36,old_metrics_recomputed=4,inputs_norm_sampler_equal=True,hashes_unchanged=True,test_used=False,
          training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','all','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=3600.;engine.dataset=dataset;engine.make_model=make_model
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    for f in (0,1):
        for s in (42,43):
            g=read(ROOT/'reports/research_resource_guard.json');assert g['observed_used_percent']<g['post_reset_stop_used_percent']
            engine.train_one('G',f,s,digest)
    audit()


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
