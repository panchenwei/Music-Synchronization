"""Read-only note timing information audit; labels never used to count collisions."""
from collections import defaultdict
import numpy as np
import pandas as pd
from .score_roll_study import ROOT,ART,OUT,DCML,split_ids,discover_dcml_pieces,voiced_events,write


def main():
    pieces=discover_dcml_pieces(DCML)
    ids=sorted({p for f in (0,1) for k in ('train','validation') for p in split_ids(f)[k]})
    rows=[]
    for pid in ids:
        roll=np.load(ART/'cache'/f'{pid}.npy',allow_pickle=False)
        n=len(roll);events=voiced_events(pieces[pid],n)
        unique={(round(o,9),int(p)) for o,d,p,*_ in events if d>0 and 0<=o<n}
        bins=defaultdict(set)
        for o,p in unique:
            bins[(min(int(np.floor(o*4+1e-9)),4*n-1),p)].add(o)
        lost=sum(len(v)-1 for v in bins.values())
        off=sum(abs(o*4-round(o*4))>1e-6 for o,p in unique)
        assert int(roll[:,1].sum()) == len(bins)
        rows.append(dict(piece_id=pid,beats=n,all_note_rows=len(events),zero_duration_rows=sum(d<=0 for o,d,*_ in events),
            unique_pitch_onsets=len(unique),occupied_pitch_onset_ticks=len(bins),
            distinct_onsets_merged=lost,off_quarter_beat_grid=off,
            outside_onset_rows=sum(not 0<=o<n for o,d,*_ in events)))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'quantization_audit.csv',index=False)
    totals=frame.drop(columns='piece_id').sum().to_dict()
    totals={k:int(v) for k,v in totals.items()}
    totals['fraction_distinct_onsets_merged']=totals['distinct_onsets_merged']/max(1,totals['unique_pitch_onsets'])
    totals['fraction_off_quarter_beat_grid']=totals['off_quarter_beat_grid']/max(1,totals['unique_pitch_onsets'])
    write(OUT/'quantization_audit.json',dict(status='complete',pieces=len(ids),labels_used=False,totals=totals,
        note='Off-grid is not itself a lost onset: the occupancy channel retains fractional duration. Only multiple distinct onsets of the same pitch in one tick merge. Duplicated simultaneous voices are not counted as distinct timing loss.'))
    print(totals)


if __name__=='__main__':main()
