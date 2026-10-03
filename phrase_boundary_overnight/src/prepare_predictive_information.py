"""LOPO training feature generation: no phrase labels used by predictors."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .score_context_study import ROOT,read,write,sha
from .data import discover_dcml_pieces
from .slice_energy_study import DCML
from .three_round_round2 import split_ids
from .score_local_coordinates import local_events
from .note_relation_graph import merged_events
from .predictive_information_features import streams,fit_predictors,features

OUT=ROOT/'reports/predictive_information';ART=ROOT/'artifacts/predictive_information'


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True);sources=discover_dcml_pieces(DCML);voices={};lengths={};hashes={};rows=[]
    ids=sorted({p for f in (0,1) for s in ('train','validation') for p in split_ids(f)[s]})
    for pid in ids:
        source=sources[pid];cache=ROOT/'artifacts/coordinate_repair_preview'/f'{pid}.npz'
        with np.load(cache,allow_pickle=False) as z:n=len(z['labels'])
        events,ties=local_events(pd.read_csv(source.notes_path,sep='\t'),pd.read_csv(source.measures_path,sep='\t'),n);events,_=merged_events(events,ties)
        voices[pid]=streams(events);lengths[pid]=n
        for p in (source.notes_path,source.measures_path,cache):hashes[str(p)]=sha(p)
    for fold in (0,1):
        folder=ART/f'fold{fold}';folder.mkdir(exist_ok=True);parts=split_ids(fold);train=parts['train'];full=fit_predictors([voices[p] for p in train])
        for split in ('train','validation'):
            for pid in parts[split]:
                fit_ids=[p for p in train if p!=pid] if split=='train' else train
                assert pid not in fit_ids
                predictors=fit_predictors([voices[p] for p in fit_ids]) if split=='train' else full
                payload={}
                for kind,unigram in [('G',False),('D',True)]:
                    x,stats=features(voices[pid],predictors,lengths[pid],unigram);payload[kind]=x
                    for r in stats:rows.append(dict(fold=fold,split=split,piece_id=pid,kind=kind,**r))
                payload['Z']=np.zeros_like(payload['G']);payload['Z'][:,[6,13]]=payload['G'][:,[6,13]]
                np.testing.assert_array_equal(payload['G'][:,[6,13]],payload['D'][:,[6,13]])
                path=folder/f'{pid}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        for k,v in payload.items():np.testing.assert_array_equal(z[k],v)
                else:np.savez_compressed(path,**payload)
                hashes[str(path)]=sha(path);assert time.monotonic()-began<300
    df=pd.DataFrame(rows);df.to_csv(OUT/'event_prediction_losses.csv.gz',index=False)
    means=df.groupby(['kind','split','voice','field']).nll.mean();means.to_csv(OUT/'prediction_means.csv')
    # Primary sanity gate on per-work prediction loss, not boundary performance.
    work=df.groupby(['fold','split','piece_id','kind']).nll.mean().unstack('kind');delta=work.G-work.D
    val=delta.xs('validation',level='split');baseline=work.D.xs('validation',level='split').mean()
    passed=bool(val.mean()<-0.01*baseline and (val<0).mean()>.5)
    for p in (Path(__file__),ROOT/'src/predictive_information_features.py',ROOT/'tests/test_predictive_information.py',OUT/'PROTOCOL.md',ROOT/'src/score_local_coordinates.py',ROOT/'src/note_relation_graph.py'):
        hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    result=dict(status='features_prepared',works=len(ids),folds=2,features=14,training_features_leave_one_piece_out=True,
        validation_g_minus_unigram_nll=float(val.mean()),validation_unigram_nll=float(baseline),positive_piece_fraction=float((val<0).mean()),predictor_gate_passed=passed,
        label_supervision=False,boundary_training_runs=0,hashes_unchanged=True,seconds=time.monotonic()-began)
    write(OUT/'completion_audit.json',result);print(result,flush=True);print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
