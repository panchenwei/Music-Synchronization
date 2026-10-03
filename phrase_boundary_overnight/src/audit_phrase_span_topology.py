"""Read-only pairing feasibility from source topology; never invent span gold."""
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids


def source_pairs(events):
    pending=None;spans=[];issues=[]
    for q,g in events.groupby('quarterbeat',sort=True):
        markers=g.marker.fillna('').astype(str)
        ending=any('}' in m for m in markers);starting=any('{' in m for m in markers)
        # A }{ closes the preceding group before opening the following one.
        if ending:
            if pending is not None:
                if q>pending:spans.append(dict(start_qb=pending,end_qb=float(q),length_qb=float(q)-pending))
                else:issues.append(dict(quarterbeat=float(q),issue='nonpositive_span'))
                pending=None
            else:issues.append(dict(quarterbeat=float(q),issue='end_without_open_start'))
        if starting:
            if pending is not None:issues.append(dict(quarterbeat=float(q),issue='new_start_before_end',previous_start=pending))
            pending=float(q)
    if pending is not None:issues.append(dict(quarterbeat=pending,issue='open_start_at_observation_end'))
    return spans,issues


def main():
    source=ROOT/'reports/phrase_semantics_review/source_annotation_events.csv';h=sha(source);df=pd.read_csv(source)
    out=ROOT/'reports/phrase_semantics_review/span_topology';out.mkdir(parents=True,exist_ok=True);spans=[];issues=[];counts=[]
    for pid,g in df.groupby('piece_id',sort=True):
        a,b=source_pairs(g);spans.extend(dict(piece_id=pid,**r) for r in a);issues.extend(dict(piece_id=pid,**r) for r in b)
        counts.append(dict(piece_id=pid,topologically_paired_spans=len(a),issues=len(b)))
    pairs=pd.DataFrame(spans);pairs.to_csv(out/'source_pairs.csv',index=False);pd.DataFrame(issues).to_csv(out/'issues.csv',index=False);pd.DataFrame(counts).to_csv(out/'counts.csv',index=False)
    # A potential prior is descriptive only and uses each fold's training pieces.
    priors=[]
    for fold in (0,1):
        v=pairs[pairs.piece_id.isin(split_ids(fold)['train'])].length_qb
        priors.append(dict(fold=fold,training_pairs=len(v),q05=float(v.quantile(.05)),median=float(v.median()),q95=float(v.quantile(.95)),minimum=float(v.min()),maximum=float(v.max())))
    pd.DataFrame(priors).to_csv(out/'train_only_length_descriptions.csv',index=False)
    assert sha(source)==h
    write(out/'manifest.json',dict(status='topology_audit_complete',source_sha256=h,script_sha256=sha(__file__),paired_spans=len(spans),issues=len(issues),source_labels_changed=False,training_runs=0,scope='Deterministic annotation pairing feasibility, not expert adjudication or musical span gold. All works/start/end labels retained. No inference prior applied.',next_requirement='Check issues and source phrase grammar before any span-gold or duration-prior claim.'))
    print(pd.DataFrame(counts).sum(numeric_only=True).to_dict());print(pd.DataFrame(issues).issue.value_counts().to_dict() if issues else {})
    print(pd.DataFrame(priors).to_string(index=False))


if __name__=='__main__':main()
