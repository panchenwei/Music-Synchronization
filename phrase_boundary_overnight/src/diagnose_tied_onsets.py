"""Read-only semantic audit of tied note heads versus acoustic attack proxies."""
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .data import discover_dcml_pieces,choose_measure_path,to_float
from .slice_energy_study import DCML
from .slice_energy_features import voiced_events
from .score_piano_roll import piano_roll


def unfold_with_ties(piece,n):
    notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t')
    assert 'tied' in notes
    path,_,_=choose_measure_path(measures,n);assert path
    folded={int(r.mc):to_float(r.quarterbeats) for r in measures.itertuples()};events=[];flags=[]
    for occurrence in path:
        mc=int(occurrence['mc'])
        for r in notes[notes.mc==mc].itertuples():
            o=occurrence['start_qb']+to_float(r.quarterbeats)-folded[mc];d=max(to_float(r.duration_qb),0)
            if np.isfinite(o) and np.isfinite(d):
                events.append((o,d,int(r.midi),int(r.staff),int(r.voice)))
                assert pd.isna(r.tied) or r.tied in (-1,0,1)
                flags.append(1 if pd.isna(r.tied) or r.tied==1 else 0)
    assert events==voiced_events(piece,n)
    return events,np.array(flags,dtype=bool)


def main():
    out=ROOT/'reports/tied_onset_diagnostic';out.mkdir(parents=True,exist_ok=True)
    pieces=discover_dcml_pieces(DCML);rows=[];hashes={}
    for cache in sorted((ROOT/'artifacts/score_roll_study/cache').glob('*.npy')):
        pid=cache.stem;old=np.load(cache,allow_pickle=False);n=len(old);piece=pieces[pid]
        for p in (cache,piece.notes_path,piece.measures_path):hashes[str(p)]=sha(p)
        events,attack=unfold_with_ties(piece,n)
        corrected=piano_roll([e for e,yes in zip(events,attack) if yes],n)[:,1]
        # Occupancy deliberately not reconstructed from attack-only short durations.
        # Full old occupancy retains all tied segments; only compare onset channel.
        assert ((old[:,1]-corrected)>=0).all()
        removed=old[:,1]>corrected
        continued=sum(not yes and e[1]>0 and 0<=e[0]<n for e,yes in zip(events,attack))
        rows.append(dict(piece_id=pid,beats=n,note_rows=len(events),continued_positive_duration_rows=continued,old_onset_cells=int(old[:,1].sum()),corrected_onset_cells=int(corrected.sum()),removed_onset_cells=int(removed.sum()),changed_beats=int(removed.any(axis=(1,2)).sum())))
    df=pd.DataFrame(rows);assert len(df)==43
    df.to_csv(out/'piece_audit.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    write(out/'summary.json',dict(status='complete',works=len(df),totals=df.select_dtypes(include='number').sum().to_dict(),source_and_cache_unchanged=True,training_runs=0,labels_accessed=False,warning='Removing tied continuations cannot prove F1 gain. Proposed correction retains all sounding occupancy; start-of-traversal orphan ties need explicit policy.'))
    print(read(out/'summary.json'))


if __name__=='__main__':main()
