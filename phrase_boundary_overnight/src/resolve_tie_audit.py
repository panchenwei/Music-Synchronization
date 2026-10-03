"""Diagnose notation voice changes across ties; no musical relabelling."""
import numpy as np
import pandas as pd
from .score_context_study import ROOT,write,sha
from .data import discover_dcml_pieces
from .slice_energy_study import DCML
from .score_local_coordinates import local_events


def predecessors(events,ties,index):
    event=events[index];q,d,p,s,v=event
    pool=[j for j,(e,t) in enumerate(zip(events,ties)) if t in (0,1) and e[1]>0 and e[2]==p and abs(e[0]+e[1]-q)<1e-7 and e[0]<q]
    for name, candidates in (
        ('same_staff_voice',[j for j in pool if events[j][3:]==event[3:]]),
        ('same_staff_cross_voice',[j for j in pool if events[j][3]==s]),
        ('cross_staff',pool)):
        if candidates:return name,candidates
    return 'unresolved',[]


def main():
    out=ROOT/'reports/note_input_semantics';pieces=discover_dcml_pieces(DCML);rows=[];hashes={}
    for cp in sorted((ROOT/'artifacts/score_roll_study/cache').glob('*.npy')):
        n=len(np.load(cp,allow_pickle=False));piece=pieces[cp.stem]
        events,ties=local_events(pd.read_csv(piece.notes_path,sep='\t'),pd.read_csv(piece.measures_path,sep='\t'),n)
        for i,tie in enumerate(ties):
            if tie not in (0,-1):continue
            tier,ids=predecessors(events,ties,i)
            rows.append(dict(piece_id=cp.stem,index=i,quarterbeat=events[i][0],pitch=events[i][2],staff=events[i][3],voice=events[i][4],tier=tier,candidates=len(ids),predecessor=ids[0] if len(ids)==1 else None))
        for p in (cp,piece.notes_path,piece.measures_path):hashes[str(p)]=sha(p)
    df=pd.DataFrame(rows);df.to_csv(out/'tie_connection_audit.csv',index=False)
    counts=df.groupby(['tier','candidates']).size().reset_index(name='count');counts.to_csv(out/'tie_connection_counts.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items());write(out/'tie_connection_audit.json',dict(status='complete',source_only=True,training_updates=0,counts=counts.to_dict('records')))
    print(counts.to_string(index=False),flush=True)


if __name__=='__main__':main()
