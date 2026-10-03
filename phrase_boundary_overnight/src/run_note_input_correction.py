"""Frozen note-input ablation, preserving all historical models and caches."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import ensemble_seed_replication as engine
from .label_repaired_rebaseline import dataset as base_data
from .crossbeat_roll_branch import CrossbeatBoundary,predictions
from .score_local_coordinates import local_events
from .score_piano_roll import piano_roll
from .tie_aware_roll import build
from .data import discover_dcml_pieces
from .slice_energy_study import DCML
from .phrase_end_auxiliary import ROOT,read,write,sha,normalizer,split_ids
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/note_input_correction';ART=ROOT/'artifacts/note_input_correction'
MODES=('P','T');COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids,mode):
    data=base_data(ids,'R')
    for pid,v in data.items():
        with np.load(ART/'cache'/f'{pid}.npz',allow_pickle=False) as z:v['piano_roll']=z[mode].copy()
    return data


def prepare():
    assert read(ROOT/'reports/crossbeat_roll_study/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/crossbeat_roll_study/contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    (ART/'cache').mkdir(parents=True,exist_ok=True);pieces=discover_dcml_pieces(DCML);rows=[]
    for cp in sorted((ROOT/'artifacts/score_roll_study/cache').glob('*.npy')):
        pid=cp.stem;n=len(np.load(cp,allow_pickle=False));piece=pieces[pid]
        events,ties=local_events(pd.read_csv(piece.notes_path,sep='\t'),pd.read_csv(piece.measures_path,sep='\t'),n)
        p=piano_roll(events,n);t,info=build(events,ties,n)
        np.testing.assert_array_equal(p[:,0],t[:,0]);assert np.all(t[:,1]<=p[:,1])
        dest=ART/'cache'/f'{pid}.npz'
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:np.testing.assert_array_equal(z['P'],p);np.testing.assert_array_equal(z['T'],t)
        else:np.savez_compressed(dest,P=p,T=t)
        rows.append(dict(piece_id=pid,beats=n,removed_onset_cells=int(np.count_nonzero(p[:,1]!=t[:,1])),**{k:v for k,v in info.items() if k!='unresolved_indices'}))
        for path in (cp,piece.notes_path,piece.measures_path,dest):hashes[str(path)]=sha(path)
    counts=pd.DataFrame(rows);assert len(counts)==43 and counts.verified_continuations.sum()==1404 and counts.unresolved_continuations.sum()==10
    counts.to_csv(OUT/'input_counts.csv',index=False)
    files=[Path(__file__),ROOT/'src/tie_aware_roll.py',ROOT/'src/resolve_tie_audit.py',ROOT/'tests/test_tie_aware_roll.py',ROOT/'tests/test_tie_connection_audit.py',OUT/'PROTOCOL.md',ROOT/'artifacts/crossbeat_roll_study/summary.csv']
    files.extend((ROOT/'artifacts/crossbeat_roll_study/independent/metrics').glob('*_fold*.json'))
    files.extend((ROOT/'artifacts/crossbeat_roll_study/independent/checkpoints').glob('*/*.pt'))
    for p in files:hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    for mode in MODES:
        for p in (OUT/mode,ART/mode/'metrics',ART/mode/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
        done=[read(p) for p in (ART/mode/'metrics').glob('*_fold*.json')];assert all(r['contract']==contract for r in done)
        write(OUT/mode/'STATE.json',dict(status='ready',contract=contract,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None))
    write(OUT/'STATE.json',dict(status='ready',contract=contract));return contract


def audit_report():
    began=time.monotonic();contract=read(OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    cps=list(ART.glob('*/checkpoints/*/*.pt'));assert len(cps)==16;before={str(p):sha(p) for p in cps};rows=[];audits=[]
    for mode in MODES:
        results=[read(p) for p in (ART/mode/'metrics').glob('*_fold*.json')];assert len(results)==4
        assert set((r['fold'],r['seed']) for r in results)=={(f,s) for f in (0,1) for s in (42,43)}
        for r in results:
            ids=split_ids(r['fold']);assert not set(ids['train'])&set(ids['validation'])
            train=dataset(ids['train'],mode);val=dataset(ids['validation'],mode);norm=normalizer(train)
            # Only the roll changes: labels, original input and normalizer match old R.
            old=base_data(ids['train'],'R');onorm=normalizer(old)
            np.testing.assert_array_equal(onorm.mean,norm.mean);np.testing.assert_array_equal(onorm.std,norm.std)
            for pid in train:
                for k in ('curves','labels','label_mask'):np.testing.assert_array_equal(train[pid][k],old[pid][k])
            del old
            cp=ART/mode/'checkpoints'/r['run_id'];best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
            assert last['step']==300 and best['contract']==last['contract']==contract['contract']
            assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
            np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
            raw=checked_raw(pd.read_csv(ART/mode/'metrics'/f"{r['run_id']}_predictions.csv.gz"),val);score=metrics(raw,val,r['threshold'])[2]
            assert max(abs(score[k]-r[k]) for k in COLS)<1e-10
            model=CrossbeatBoundary(False,r['seed']);stem={k:v.clone() for k,v in model.stem.state_dict().items()};model.load_state_dict(best['model']);assert all(torch.equal(stem[k],v) for k,v in model.stem.state_dict().items())
            model.to('cuda').eval();replay=predictions(model,val,norm,torch.device('cuda'))
            delta=max(float(np.max(abs(replay[p][k]-raw[p][k]))) for p in raw for k in raw[p]);assert delta<2e-4
            assert max(abs(metrics(replay,val,r['threshold'])[2][k]-score[k]) for k in COLS)<1e-10
            ts=metrics(predictions(model,train,norm,torch.device('cuda')),train,r['threshold'])[2]
            rows.append(dict(mode=mode,**{k:v for k,v in r.items() if k!='history'}));audits.append(dict(mode=mode,run_id=r['run_id'],gpu_replay_error=delta,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']))
            pd.DataFrame(audits).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',mode,r['run_id'],flush=True)
    df=pd.DataFrame(rows);old=pd.read_csv(ROOT/'artifacts/crossbeat_roll_study/summary.csv');old=old[old['mode']=='independent'].copy();old['mode']='O'
    all_results=pd.concat([old,df],ignore_index=True);all_results.to_csv(ART/'summary.csv',index=False)
    means=all_results.groupby('mode')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for a,b in (('P','O'),('T','P'),('T','O')):
        aa=all_results[all_results['mode']==a].set_index(['fold','seed']);bb=all_results[all_results['mode']==b].set_index(['fold','seed']);d=aa[COLS]-bb[COLS];d.to_csv(OUT/f'{a}_minus_{b}.csv')
        comparisons.append(dict(candidate=a,reference=b,**{'delta_'+k:float(d[k].mean()) for k in COLS},f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(b=='O' and d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    assert all(sha(p)==h for p,h in contract['hashes'].items()) and before=={str(p):sha(p) for p in cps}
    write(OUT/'checkpoint_hashes.json',before);write(OUT/'completion_audit.json',dict(status='complete',new_runs=8,new_checkpoints=16,original_controls_reused=4,gpu_replays=8,original34_labels_masks_norms_unchanged=True,stem_unchanged=True,hashes_unchanged=True,test_predictions_accessed=False,training_validation_seconds=float(df.seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',contract=contract['contract']));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=p.parse_args();torch.set_num_threads(2)
    try:
        if args.stage=='audit':audit_report();return
        contract=prepare()
        if args.stage=='prepare':print(contract);return
        engine.predictions=predictions;engine.build_model=lambda kind,seed:CrossbeatBoundary(False,seed);engine.CAP=1200.
        for mode in MODES:
            engine.OUT=OUT/mode;engine.ART=ART/mode;engine.dataset=lambda ids,kind:dataset(ids,mode)
            write(OUT/'STATE.json',dict(status='running',mode=mode,contract=contract))
            for fold in (0,1):
                for seed in (42,43):engine.train_one('R',fold,seed,contract)
            write(OUT/mode/'STATE.json',{**read(OUT/mode/'STATE.json'),'status':'training_complete','pid':None})
        write(OUT/'STATE.json',dict(status='auditing',contract=contract));audit_report()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed'));raise


if __name__=='__main__':main()
