"""Frozen C3 and soft decoder checked on two new seeds, not new test works."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from . import score_context_study as engine
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .score_context_study import ROOT,read,write,sha,normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior

OUT=ROOT/'reports/c3_seed_replication';ART=ROOT/'artifacts/c3_seed_replication';COLS=base.COLS


def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    for study in ('recurrence_tabular_study','recurrence_tonal_study'):
        assert read(ROOT/'reports'/study/'completion_audit.json')['status']=='complete'
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),Path(dec.__file__),ROOT/'src/interstart_decoder.py',OUT/'PROTOCOL.md',ROOT/'artifacts/interstart_decoder_study/summary.csv'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',seconds=0.,completed=[],pid=None,contract=digest))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');rows=[];checks=[];decoded=[]
    cp_hash={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(cp_hash)==8
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for fold in (0,1):
        ids=split_ids(fold);train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');norm=normalizer(train);prior=fit_prior(train)
        write(OUT/f'prior_fold{fold}.json',dict(prior=prior,train_ids=ids['train']))
        for seed in (44,45):
            assert time.monotonic()-began<900
            run=f'C3_seed{seed}_fold{fold}';r=read(ART/'metrics'/f'{run}.json');cp=ART/'checkpoints'/run
            best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
            assert (best['contract'],best['kind'],best['fold'],best['seed'])==(contract['contract'],'C3',fold,seed)
            assert last['contract']==r['contract']==contract['contract'] and last['step']==300
            assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
            np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
            raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val)
            assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            m=base.make_model('C3',seed).cuda();m.load_state_dict(best['model']);assert r['params']==5921
            replay=predictions(m,val,norm,torch.device('cuda'));err=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert err<2e-4
            assert max(abs(metrics(replay,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            ts=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
            checks.append(dict(run_id=run,replay_error=err,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']));rows.append(r)
            for strength in (0.,.5):
                row=dec.evaluate(raw,val,r['threshold'],prior,strength,f'{run}_lambda{strength:g}',contract['contract'])
                if strength==0:assert max(abs(row[c]-r[c]) for c in COLS)<1e-10
                decoded.append(dict(fold=fold,seed=seed,**row))
            pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    neural=pd.DataFrame(rows);neural.to_csv(ART/'summary.csv',index=False)
    d=pd.DataFrame(decoded);assert len(d)==8;d.to_csv(ART/'decoder/summary.csv',index=False)
    original=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');original=original[original.strength.isin([0.,.5])].copy()
    d['subset']='new_44_45';original['subset']='original_42_43';allrows=pd.concat([d,original],ignore_index=True)
    allrows.groupby(['subset','strength'])[COLS].mean().to_csv(OUT/'means.csv')
    allrows.groupby('strength')[COLS].mean().to_csv(OUT/'all_four_seed_means.csv')
    a=d[d.strength==.5].set_index(['fold','seed']);b=d[d.strength==0].set_index(['fold','seed']);delta=a[COLS]-b[COLS]
    delta.to_csv(OUT/'new_seed_decoder_deltas.csv')
    write(OUT/'comparison.json',dict(**delta.mean().to_dict(),f1_positive=int((delta.macro_f1_tol1>0).sum()),passed=bool(delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>=0 and (delta.macro_f1_tol1>0).sum()>=3)))
    assert all(sha(p)==h for p,h in cp_hash.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',cp_hash)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=4,full_gpu_replays=4,checkpoints=8,decoder_cells=8,hashes_unchanged=True,norms_rebuilt=True,test_used=False,training_seconds=float(neural.seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'decoder_state/STATE.json',dict(status='complete',pid=None));write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']))


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1200.;engine.dataset=base.dataset;engine.make_model=base.make_model
    contract=prepare()
    for f in (0,1):
        for s in (44,45):engine.train_one('C3',f,s,contract)
    write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
