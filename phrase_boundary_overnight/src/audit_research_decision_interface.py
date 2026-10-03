"""Replay label-free interface against every saved candidate prediction/event."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .research_decision_postprocess import ScoreCoverageDecision
from . import coverage_decoder_composition as study
from . import four_seed_decision_audit as parent
from . import run_recurrence_depth_study as base
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .interstart_decoder import fit_prior
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/decision_interface_audit'

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(study.OUT/'contract.json')['hashes'])
    for item in read(parent.OUT/'probability_hashes.json'):hashes[item['path']]=item['sha256']
    for p in [Path(__file__),ROOT/'src/research_decision_postprocess.py',ROOT/'tests/test_research_decision_postprocess.py']+list((study.ART/'decoder').glob('*lambda1_positions.csv.gz'))+list((study.ART/'decoder').glob('*lambda1_pieces.csv')):
        hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    checks=[];piece_frames=[]
    for f in (0,1):
        ids=split_ids(f);train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');prior=fit_prior(train)
        processors={}
        for pid,v in val.items():
            with np.load(ROOT/'artifacts/note_relation_study/cache'/f'{pid}.npz',allow_pickle=False) as z:
                assert int(z['n_beats'])==len(v['labels'])
                processors[pid]=ScoreCoverageDecision(z['note_events'],int(z['n_beats']))
        for s in (42,43,44,45):
            original=checked_raw(pd.read_csv(parent.ART/f'B_seed{s}_fold{f}_predictions.csv.gz'),val)
            expected=checked_raw(pd.read_csv(parent.ART/f'M_seed{s}_fold{f}_predictions.csv.gz'),val)
            name=f'M_seed{s}_fold{f}_lambda1';threshold=read(study.ART/'decoder'/f'{name}.json')['threshold']
            events=pd.read_csv(study.ART/'decoder'/f'{name}_positions.csv.gz')
            groups={(p,q):v.beat.to_numpy(int) for (p,q),v in events.groupby(['piece_id','performance_id'])}
            maximum=0.;performances=0
            for pid,pp in original.items():
                for perf,prob in pp.items():
                    result=processors[pid].predict(prob,threshold,prior)
                    maximum=max(maximum,float(abs(result['adjusted_probabilities']-expected[pid][perf]).max()))
                    np.testing.assert_array_equal(result['boundaries'],groups.get((pid,perf),np.array([],int)))
                    performances+=1
            assert maximum<1e-12
            checks.append(dict(fold=f,seed=s,performances=performances,max_probability_error=maximum,saved_events_identical=True))
            df=pd.read_csv(study.ART/'decoder'/f'{name}_pieces.csv');df['fold']=f;df['seed']=s;piece_frames.append(df)
    pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
    new=pd.concat(piece_frames).groupby(['fold','piece_id']).f1_tol1.mean()
    old=pd.read_csv(parent.OUT/'all_piece_metrics.csv')
    labels=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv',dtype={'opus':str})
    labels=labels[labels.split=='validation'][['fold','piece_id','opus']].drop_duplicates()
    results=[]
    for condition in ('B05','B10','M'):
        control=old[old.condition==condition].groupby(['fold','piece_id']).f1_tol1.mean()
        delta=(new-control).reset_index(name='delta').merge(labels,on=['fold','piece_id'],validate='one_to_one')
        rng=np.random.default_rng(20260913);samples=[]
        for f in (0,1):
            groups=[g.delta.to_numpy() for _,g in delta[delta.fold==f].groupby('opus')]
            sums=np.array([g.sum() for g in groups]);counts=np.array([len(g) for g in groups]);draw=rng.integers(0,len(groups),(5000,len(groups)))
            samples.append(sums[draw].sum(1)/counts[draw].sum(1))
        lower,upper=np.quantile(np.mean(samples,axis=0),[.025,.975])
        results.append(dict(candidate='M10',control='M05' if condition=='M' else condition,delta=float(delta.groupby('fold').delta.mean().mean()),ci95_low=float(lower),ci95_high=float(upper),works=19,opus_groups=6,repeats=5000,scope='conditional development comparison; does not remove model-selection bias'))
    pd.DataFrame(results).to_csv(OUT/'cluster_bootstrap.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',model_cells=8,source_hashes_unchanged=True,interface_probability_replay=True,interface_event_replay=True,test_used=False,seconds=time.monotonic()-start))
    print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
