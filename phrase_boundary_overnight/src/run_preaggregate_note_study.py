"""Isolated adapter to the frozen full-work training loop; four new N runs."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import run_note_relation_study as engine
from .preaggregate_note_graph import PreAggregateBoundary
from .note_relation_graph import NoteRelationBoundary
from .label_repaired_rebaseline import dataset as base_data
from .phrase_end_auxiliary import ROOT,read,write,sha,normalizer,split_ids
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/preaggregate_note_study';ART=ROOT/'artifacts/preaggregate_note_study'
OLD_OUT=ROOT/'reports/note_relation_study';OLD_ART=ROOT/'artifacts/note_relation_study'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids):
    data=base_data(ids,'B')
    for pid,v in data.items():
        with np.load(OLD_ART/'cache'/f'{pid}.npz',allow_pickle=False) as z:v['graph']={k:z[k].copy() for k in z.files}
        assert int(v['graph']['n_beats'])==len(v['labels'])
    return data


def factory(legacy_use_relations,seed):
    # Old loop derives S/G flag from run kind; N always uses real relations.
    return PreAggregateBoundary(seed)


def bind_engine():
    engine.OUT=OUT;engine.ART=ART;engine.dataset=dataset;engine.NoteRelationBoundary=factory


def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    assert read(OLD_OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(OLD_OUT/'contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    for p in (Path(__file__),ROOT/'src/preaggregate_note_graph.py',ROOT/'tests/test_preaggregate_note_graph.py',OUT/'PROTOCOL.md',OLD_OUT/'completion_audit.json',OLD_OUT/'inference_interventions/audit.json'):
        hashes[str(p)]=sha(p)
    for pattern in ('metrics/G_seed*_fold*','checkpoints/G_seed*_fold*/*.pt'):
        for p in OLD_ART.glob(pattern):
            if p.is_file():hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('N_seed*_fold*.json')];assert all(r['contract']==digest for r in done)
    write(OUT/'STATE.json',dict(status='ready',contract=digest,seconds=sum(r['seconds'] for r in done),completed=[r['run_id'] for r in done],pid=None))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    cps=list((ART/'checkpoints').glob('*/*.pt'))+list((OLD_ART/'checkpoints').glob('G*/*.pt'));assert len(cps)==16
    before={str(p):sha(p) for p in cps};rows=[];checks=[]
    for label,root in (('N',ART),('L',OLD_ART)):
        pattern='N_seed*_fold*.json' if label=='N' else 'G_seed*_fold*.json'
        results=sorted((root/'metrics').glob(pattern));assert len(results)==4
        for rp in results:
            assert time.monotonic()-began<600
            r=read(rp);ids=split_ids(r['fold']);train=dataset(ids['train']);val=dataset(ids['validation']);norm=normalizer(train)
            cp=root/'checkpoints'/r['run_id'];best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
            expected=contract['contract'] if label=='N' else read(OLD_OUT/'contract.json')['contract']
            assert r['contract']==best['contract']==last['contract']==expected
            assert last['step']==300 and best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
            np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
            if label=='N':
                control=torch.load(OLD_ART/'checkpoints'/f"G_seed{r['seed']}_fold{r['fold']}"/'best.pt',map_location='cpu',weights_only=False)
                np.testing.assert_array_equal(best['mean'],control['mean']);np.testing.assert_array_equal(best['std'],control['std'])
            raw=checked_raw(pd.read_csv(root/'metrics'/f"{r['run_id']}_predictions.csv.gz"),val)
            score=metrics(raw,val,r['threshold'])[2];assert max(abs(score[k]-r[k]) for k in COLS)<1e-10
            model=(PreAggregateBoundary(r['seed']) if label=='N' else NoteRelationBoundary(True,r['seed'])).cuda();model.load_state_dict(best['model'])
            replay=engine.predictions(model,val,norm);delta=max(float(np.max(abs(replay[p][k]-raw[p][k]))) for p in raw for k in raw[p]);assert delta<2e-4
            assert max(abs(metrics(replay,val,r['threshold'])[2][k]-score[k]) for k in COLS)<1e-10
            ts=metrics(engine.predictions(model,train,norm),train,r['threshold'])[2]
            rows.append({**{k:v for k,v in r.items() if k!='history'},'mode':label})
            checks.append(dict(run_id=r['run_id'],mode=label,gpu_replay_error=delta,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']))
            pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',label,r['run_id'],flush=True)
    df=pd.DataFrame(rows);assert set(zip(df['mode'],df.fold,df.seed))=={(m,f,s) for m in ('L','N') for f in (0,1) for s in (42,43)}
    df.to_csv(ART/'summary.csv',index=False);means=df.groupby('mode')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean();means.to_csv(OUT/'means.csv')
    n=df[df['mode']=='N'].set_index(['fold','seed']);l=df[df['mode']=='L'].set_index(['fold','seed']);d=n[COLS]-l[COLS];d.to_csv(OUT/'paired_deltas.csv')
    write(OUT/'comparison.json',dict(mean_delta=d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    assert before=={str(p):sha(p) for p in cps} and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',before)
    write(OUT/'completion_audit.json',dict(status='complete',new_runs=4,reused_controls=4,total_checkpoints=16,full_gpu_replays=8,norms_rebuilt=True,hashes_unchanged=True,test_predictions_accessed=False,training_validation_seconds=float(df[df['mode']=='N'].seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',contract=contract['contract'],pid=None));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','all','audit'),default='all');args=p.parse_args();torch.set_num_threads(2)
    try:
        bind_engine()
        if args.stage=='audit':audit();return
        contract=prepare()
        if not (OUT/'numerical_probe.json').exists():engine.probe()
        if args.stage=='prepare':print(contract,flush=True);return
        for f in (0,1):
            for s in (42,43):engine.train_one('N',f,s,contract)
        write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise


if __name__=='__main__':main()
