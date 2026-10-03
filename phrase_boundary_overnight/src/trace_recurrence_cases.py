"""Trace new recurrence cues at the thirty previously frozen case locations."""
import time
import numpy as np
import pandas as pd
from .motif_recurrence_features import staff_sequences,window_similarity,SCALES
from .phrase_end_auxiliary import ROOT,read,write,sha
from .label_repaired_rebaseline import dataset as base_data


def main():
    began=time.monotonic();out=ROOT/'reports/motif_recurrence_study/case_traces';out.mkdir(exist_ok=True)
    cases=pd.read_csv(ROOT/'reports/error_review_30/cases.csv');assert len(cases)==30
    hashes=read(ROOT/'reports/motif_recurrence_study/contract.json')['hashes'];assert all(sha(p)==h for p,h in hashes.items());rows=[]
    for pid,group in cases.groupby('piece_id'):
        with np.load(ROOT/'artifacts/note_relation_study/cache'/f'{pid}.npz',allow_pickle=False) as z:events=z['note_events'].copy();n=int(z['n_beats'])
        with np.load(ROOT/'artifacts/motif_recurrence_study/cache'/f'{pid}.npz',allow_pickle=False) as z:cached=z['O'].copy()
        data=base_data([pid],'B')[pid];sequences=staff_sequences(events,n)
        for stream,x in enumerate(sequences):
            for ix,w in enumerate(SCALES):
                sim=window_similarity(x,w,True);positions=np.arange(w,n-w+1)
                for case in group.to_dict('records'):
                    b=int(case['beat']);common=dict(case_id=case['case_id'],piece_id=pid,beat=b,old_category=case['category'],current_start_label=int(data['labels'][b]),valid=int(data['label_mask'][b]),notated_staff='upper' if stream==0 else 'lower',window_quarterbeats=w)
                    js=positions[abs(positions-b)>=2*w]
                    if not (w<=b<=n-w) or not len(js):rows.append({**common,'available':False});continue
                    f=sim[b,js];back=sim[b-w,js-w];cue=f*(1-back);j=int(np.argmax(cue));col=stream*12+ix*4
                    expected=[f.max(),back.max(),cue[j],1.];np.testing.assert_allclose(cached[b,col:col+4],expected,rtol=0,atol=1e-6)
                    rows.append({**common,'available':True,'restart_match_beat':int(js[j]),'forward_similarity':float(f[j]),'backward_similarity':float(back[j]),'restart_cue':float(cue[j]),'max_forward_similarity':float(f.max()),'max_backward_similarity':float(back.max())})
    assert len(rows)==180 and all(sha(p)==h for p,h in hashes.items());pd.DataFrame(rows).to_csv(out/'case_matches.csv',index=False)
    write(out/'audit.json',dict(status='complete',cases=30,source_selection='Previously frozen error_review_30/cases.csv, not selected using new model performance',rows=180,cache_cues_rebuilt=True,input_hashes_unchanged=True,seconds=time.monotonic()-began,labels_only_used_for_report_not_matching=True,scope='Cue trace, not music-expert adjudication or new phrase annotation.'))


if __name__=='__main__':main()
