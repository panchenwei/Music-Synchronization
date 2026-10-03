"""Trace existing false negatives; no alternate decoding or changed metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
from . import external_audio_trial as base
from .external_case_audit import pairing
from .phase2_models import nms_probabilities
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_case_audit'


def main():
    data=base.load_data();splits=read(base.OUT/'splits.json');choice=pd.read_csv(ROOT/'reports/external_linear_novelty/run_audit.csv');rows=[];totals=[];files=[Path(__file__),ROOT/'src/external_case_audit.py',ROOT/'src/phase2_models.py',OUT/'MISS_TRACE_PROTOCOL.md']
    for fold in range(3):
        held=base.subset(data,splits[str(fold)]['test']);threshold=float(choice[(choice.kind=='E')&(choice.fold==fold)].iloc[0].threshold)
        path=ROOT/f'artifacts/external_linear_novelty/E_fold{fold}_test.npz';files.append(path)
        with np.load(path,allow_pickle=False) as z:raw={k:z[k].copy() for k in held}
        for key,item in held.items():
            prob=raw[key];mask=item['label_mask']>.5;nms=nms_probabilities(prob);truth=np.flatnonzero((item['labels']>.5)&mask);pred=np.flatnonzero((nms>=threshold)&mask);pairs,fp,miss=pairing(pred,truth)
            totals.append(dict(fold=fold,record=key,piece_id=item['piece_id'],positive_appearances=len(truth),tp=len(pairs),fn=len(miss),truth_near_unknown=sum(not mask[max(0,t-2):min(len(mask),t+3)].all() for t in truth)))
            for t in miss:
                near=np.arange(max(0,t-1),min(len(mask),t+2));known=near[mask[near]];unknown=near[~mask[near]]
                sufficient=bool((prob[known]>=threshold).any());survived=bool((nms[known]>=threshold).any())
                reason='low_local_score' if not sufficient else ('NMS_suppressed' if not survived else 'one_to_one_competition')
                rows.append(dict(fold=fold,record=key,piece_id=item['piece_id'],quarterbeat=int(t),reason=reason,
                    score_at_truth=float(prob[t]),local_known_max=float(prob[known].max()),threshold=threshold,
                    within2_of_unknown=bool(not mask[max(0,t-2):min(len(mask),t+3)].all()),
                    surviving_peak_in_unknown_within1=bool((nms[unknown]>=threshold).any()),known_candidate_before_NMS=sufficient))
    frame=pd.DataFrame(rows);total=pd.DataFrame(totals);assert len(frame)==int(total.fn.sum())
    frame.to_csv(OUT/'miss_trace.csv',index=False);total.to_csv(OUT/'miss_trace_totals.csv',index=False)
    result=dict(status='complete',records=len(total),works=int(total.piece_id.nunique()),positive_appearances=int(total.positive_appearances.sum()),
        false_negative_appearances=len(frame),counts_by_reason={str(k):int(v) for k,v in frame.reason.value_counts().items()},
        truth_within2_of_unknown=int(total.truth_near_unknown.sum()),miss_within2_of_unknown=int(frame.within2_of_unknown.sum()),
        miss_with_unknown_peak_within1=int(frame.surviving_peak_in_unknown_within1.sum()),training_runs=0,alternative_metrics_computed=False,
        caution='Counts are performance appearances, NOT independent works; categories describe this frozen linear model, not all project models.')
    write(OUT/'miss_trace_completion.json',result);write(OUT/'miss_trace_source_hashes.json',{str(p):sha(p) for p in files});print(result)


if __name__=='__main__':main()
