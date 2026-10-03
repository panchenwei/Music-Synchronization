"""Source-only note timing/tie audit. Does not replace any training cache."""
import time
import numpy as np
import pandas as pd
from .score_context_study import ROOT,write,sha
from .slice_energy_study import DCML
from .slice_energy_features import voiced_events
from .score_local_coordinates import local_events
from .score_piano_roll import piano_roll
from .data import discover_dcml_pieces


def main():
    began=time.monotonic();out=ROOT/'reports/note_input_semantics';out.mkdir(parents=True,exist_ok=True)
    pieces=discover_dcml_pieces(DCML);rows=[];details=[];hashes={}
    for cp in sorted((ROOT/'artifacts/score_roll_study/cache').glob('*.npy')):
        pid=cp.stem;piece=pieces[pid];old=np.load(cp,allow_pickle=False);n=len(old)
        notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t')
        old_events=voiced_events(piece,n);rebuilt=piano_roll(old_events,n)
        np.testing.assert_array_equal(old,rebuilt)
        events,ties=local_events(notes,measures,n);local=piano_roll(events,n)
        # Hypothetical tie-aware onset channel for DIAGNOSTIC counts only.
        proposed=local.copy();proposed[:,1]=0
        for (q,d,p,*_),tie in zip(events,ties):
            if d>0 and 0<=q<n and tie in (None,1):
                tick=min(int(np.floor(q*4+1e-9)),n*4-1);proposed[tick//4,1,p,tick%4]=1
        orphan=0;ambiguous=0;continuations=0
        for i,(event,tie) in enumerate(zip(events,ties)):
            if tie not in (-1,0):continue
            continuations+=1;q,d,p,s,v=event
            previous=[j for j,(e,t) in enumerate(zip(events,ties)) if t in (0,1) and e[1]>0 and e[2:]==event[2:] and abs(e[0]+e[1]-q)<1e-7 and e[0]<q]
            issue='matched' if len(previous)==1 else ('no_predecessor' if not previous else 'ambiguous_predecessors')
            orphan+=not previous;ambiguous+=len(previous)>1
            if issue!='matched':details.append(dict(piece_id=pid,event_index=i,quarterbeat=q,pitch=p,staff=s,voice=v,tied=tie,issue=issue,predecessors=previous))
        coord_beats=np.any(abs(old-local)>1e-6,axis=(1,2,3))
        tie_beats=np.any(abs(local-proposed)>1e-6,axis=(1,2,3))
        rows.append(dict(piece_id=pid,beats=n,old_noteheads=len(old_events),local_noteheads=len(events),restored_noteheads=len(events)-len(old_events),coordinate_changed_beats=int(coord_beats.sum()),coordinate_changed_onset_cells=int(np.count_nonzero(old[:,1]!=local[:,1])),coordinate_occupancy_absolute_change=float(np.abs(old[:,0]-local[:,0]).sum()),tied_continuation_noteheads=continuations,missing_predecessor=orphan,ambiguous_predecessor=ambiguous,potential_spurious_onset_cells=int(np.count_nonzero(local[:,1]!=proposed[:,1])),tie_changed_beats=int(tie_beats.sum())))
        for p in (cp,piece.notes_path,piece.measures_path):hashes[str(p)]=sha(p)
    df=pd.DataFrame(rows);assert len(df)==43
    df.to_csv(out/'by_piece.csv',index=False);pd.DataFrame(details).to_csv(out/'tie_exceptions.csv',index=False)
    summary={c:int(df[c].sum()) for c in df if c not in ('piece_id','coordinate_occupancy_absolute_change')}
    summary.update(works=43,coordinate_affected_works=int((df.coordinate_changed_beats>0).sum()),tie_affected_works=int((df.tie_changed_beats>0).sum()),seconds=time.monotonic()-began,training_cache_modified=False,model_predictions_accessed=False,labels_accessed=False,status='complete')
    assert all(sha(p)==h for p,h in hashes.items());write(out/'source_hashes.json',hashes);write(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
