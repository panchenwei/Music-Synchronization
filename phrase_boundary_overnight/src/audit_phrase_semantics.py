"""Read-only source annotation topology; no relabelling or model evaluation."""
import numpy as np
import pandas as pd
from .score_context_study import ROOT,sha,write
from .slice_energy_study import DCML
from .data import discover_dcml_pieces
from .score_local_coordinates import positioned_rows,local_labels


def main():
    out=ROOT/'reports/phrase_semantics_review';out.mkdir(parents=True,exist_ok=True);pieces=discover_dcml_pieces(DCML);details=[];summary=[];hashes={}
    for cache in sorted((ROOT/'artifacts/coordinate_repair_preview').glob('*.npz')):
        pid=cache.stem;piece=pieces[pid]
        with np.load(cache,allow_pickle=False) as z:n=len(z['labels']);saved=z['labels'].copy();savedmask=z['label_mask'].copy()
        h=pd.read_csv(piece.harmony_path,sep='\t');m=pd.read_csv(piece.measures_path,sep='\t')
        rows,_,_=positioned_rows(h,m,n);starts=set();ends=set();cadences=set();both=set()
        for q,i,occ,r in rows:
            if not 0<=q<n:continue
            marker='' if pd.isna(r.get('phraseend')) else str(r.phraseend)
            cadence='' if pd.isna(r.get('cadence')) else str(r.cadence)
            if '{' in marker:starts.add(q)
            if '}' in marker:ends.add(q)
            if '{' in marker and '}' in marker:both.add(q)
            if cadence:cadences.add(q)
            if marker or cadence:details.append(dict(piece_id=pid,quarterbeat=q,mc=occ['mc'],visit=occ['visit'],marker=marker,cadence=cadence,source_row=i))
        labels,mask,mapping,_,_,_=local_labels(h,m,n);np.testing.assert_array_equal(saved,labels);np.testing.assert_array_equal(savedmask,mask)
        gaps=[q-max(e for e in ends if e<=q) for q in sorted(starts)[1:] if any(e<=q for e in ends)]
        summary.append(dict(piece_id=pid,source_start_positions=len(starts),source_end_positions=len(ends),same_time_start_end=len(starts&ends),same_row_interlocking=len(both),cadence_positions=len(cadences),end_without_same_time_cadence=len(ends-cadences),end_with_cadence_within_one_qb=sum(any(abs(e-c)<=1 for c in cadences) for e in ends),off_integer_start_positions=sum(abs(q-round(q))>1e-8 for q in starts),halfbeat_ambiguous_starts=sum(r['status']=='ambiguous_tie' for r in mapping),valid_starts=int((labels*mask).sum()),start_with_preceding_end=len(gaps),start_strictly_after_nearest_end=sum(g>1e-8 for g in gaps),median_nearest_end_gap=float(np.median(gaps)) if gaps else None))
        for p in (cache,piece.harmony_path,piece.measures_path):hashes[str(p)]=sha(p)
    df=pd.DataFrame(summary);assert len(df)==43;df.to_csv(out/'source_semantics_by_piece.csv',index=False);pd.DataFrame(details).to_csv(out/'source_annotation_events.csv',index=False)
    counts={c:int(df[c].sum()) for c in df.columns if c not in ('piece_id','median_nearest_end_gap')}
    assert counts['valid_starts']==495 and all(sha(p)==v for p,v in hashes.items())
    write(out/'source_hashes.json',hashes);write(out/'source_semantics_summary.json',dict(works=43,counts=counts,training_runs=0,model_predictions=0,labels_modified=False,note='Source ontology inventory, not label correctness adjudication. End/cadence co-location is not equivalence; nearest previous end is not a validated phrase pairing.'))
    print(counts)


if __name__=='__main__':main()
