"""Read-only diagnosis of C3 start-to-start spacing; no decoder fitting or test access."""
import numpy as np
import pandas as pd
from . import run_recurrence_depth_study as base
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .phase2_models import nms_probabilities
from .audit_external_stem_transfer import checked_raw
from .evaluation import one_to_one_counts


def main():
    out=ROOT/'reports/boundary_spacing_diagnosis';out.mkdir(exist_ok=True)
    rows=[];train_rows=[];hashes={}
    for fold in (0,1):
        ids=split_ids(fold)
        train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3')
        for pid,v in train.items():
            true=np.flatnonzero((v['labels']>.5)&v['label_mask'].astype(bool))
            gaps=[int(b-a) for a,b in zip(true[:-1],true[1:]) if v['label_mask'][a:b+1].all()]
            train_rows.extend(dict(fold=fold,piece_id=pid,gap=d) for d in gaps)
        for seed in (42,43):
            stem=f'C3_seed{seed}_fold{fold}';meta=base.ART/'metrics'/f'{stem}.json'
            path=base.ART/'metrics'/f'{stem}_predictions.csv.gz';hashes[str(path)]=sha(path);hashes[str(meta)]=sha(meta)
            threshold=read(meta)['threshold'];raw=checked_raw(pd.read_csv(path),val)
            for pid,perfs in raw.items():
                v=val[pid];mask=v['label_mask'].astype(bool);truth=np.flatnonzero((v['labels']>.5)&mask)
                for perf,p in perfs.items():
                    pred=np.flatnonzero((nms_probabilities(p)>=threshold)&mask)
                    used_p=set();used_t=set()
                    for _,a,b in sorted((abs(int(a)-int(b)),int(a),int(b)) for a in pred for b in truth if abs(a-b)<=1):
                        if a not in used_p and b not in used_t:used_p.add(a);used_t.add(b)
                    counts=one_to_one_counts(pred,truth,1);assert counts.tp==len(used_p)
                    pair=[(int(a),int(b)) for a,b in zip(pred[:-1],pred[1:]) if mask[a:b+1].all()]
                    tg=[b-a for a,b in zip(truth[:-1],truth[1:]) if mask[a:b+1].all()]
                    short=[(a,b) for a,b in pair if b-a<6]
                    rows.append(dict(fold=fold,seed=seed,piece_id=pid,performance_id=perf,
                        pred_count=len(pred),true_count=len(truth),fp=counts.fp,fn=counts.fn,
                        pred_gap_under6_fraction=np.mean([b-a<6 for a,b in pair]) if pair else np.nan,
                        true_gap_under6_fraction=np.mean(np.asarray(tg)<6) if tg else np.nan,
                        short_pairs=len(short),short_pairs_with_fp=sum(a not in used_p or b not in used_p for a,b in short)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'performances.csv',index=False)
    pieces=frame.groupby(['fold','seed','piece_id']).mean(numeric_only=True);pieces.to_csv(out/'pieces.csv')
    pd.DataFrame(train_rows).to_csv(out/'train_interstart_gaps.csv',index=False)
    means=pieces.groupby(['fold','seed']).mean();means.to_csv(out/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items())
    write(out/'audit.json',dict(source_hashes=hashes,source_unchanged=True,test_used=False,
        evaluation_only_masks=True,decoder_trained=False,limitation='Start-to-start gaps, not start-to-end phrase durations; 19 reused development works.'))
    print(means.to_string())


if __name__=='__main__':main()
