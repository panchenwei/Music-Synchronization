"""Pure-score extraction with explicit score origin; no harmony/label reads."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
from src.data import choose_measure_path,to_float,discover_dcml_pieces
from src.slice_energy_study import DCML
from src.score_local_coordinates import local_events
from src.note_relation_graph import merged_events
from src.motif_recurrence_features import features
from src.score_context_study import read,write,sha
from score_semantics_audit import cues_from_events

def extract(piece,n,origin=0.):
    notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t');path,mode,error=choose_measure_path(measures,n);assert path
    lookup=measures.set_index('mc');bn=np.full(n,np.nan);mn=np.full(n,np.nan);total=sum(o['duration_qb'] for o in path)
    for i,q in enumerate(np.arange(n)+origin):
        hits=[o for o in path if o['start_qb']<=q<o['start_qb']+o['duration_qb']]
        if hits:
            assert len(hits)==1;o=hits[0];m=lookup.loc[o['mc']];phase=4*to_float(m.mc_offset);phase=phase if np.isfinite(phase) else 0.;bn[i]=phase+q-o['start_qb'];mn[i]=int(m.mn)
        elif abs(q-total)<1e-8 and i==n-1:
            o=path[-1];m=lookup.loc[o['mc']];phase=4*to_float(m.mc_offset);phase=phase if np.isfinite(phase) else 0.;bn[i]=(phase+o['duration_qb'])%(4*to_float(m.timesig));mn[i]=int(m.mn)+1
        else:raise ValueError('Uncovered score grid')
    events,ties=local_events(notes,measures,n);merged,_=merged_events(events,ties);shifted=[(e[0]-origin,*e[1:]) for e in merged if e[0]-origin<n and e[0]+e[1]-origin>0]
    assert shifted and np.isfinite(bn).all();score=cues_from_events(piece,bn,mn,shifted);motif=features(shifted,n,True);assert score.shape==(n,16) and motif.shape==(n,24) and np.isfinite(score).all() and np.isfinite(motif).all()
    return dict(score16=score,motif24=motif,beat_number=bn,measure_number=mn,note_events=np.asarray(shifted)),dict(mode=mode,length_error=error,origin_qb=origin,score_length=total)

def gate():
    # Train data are permitted here for parity only, not feature derivation.
    from score_only_export_training import data_u
    _,data=data_u();pieces=discover_dcml_pieces(DCML);rows=[];hashes={str(Path(__file__)):sha(Path(__file__)),str(BASE/'score_semantics_audit.py'):sha(BASE/'score_semantics_audit.py')};out=BASE/'score_only_export'
    for pid,v in data.items():
        piece=pieces[pid]
        for p in (piece.notes_path,piece.measures_path):hashes[str(p)]=sha(p)
        origin={'chopin_op17_no4':1.,'chopin_op68_no2':.25}.get(pid,0.);x,meta=extract(piece,len(v['labels']),origin)
        a=float(abs(x['score16']-v['selected_score']).max());b=float(abs(x['motif24']-v['pitch_profiles']).max());np.testing.assert_array_equal(x['score16'],v['selected_score']);np.testing.assert_array_equal(x['motif24'],v['pitch_profiles'])
        rows.append(dict(piece_id=pid,score16_max_error=a,motif24_max_error=b,**meta))
    pd.DataFrame(rows).to_csv(out/'feature_parity.csv',index=False);assert all(sha(p)==h for p,h in hashes.items());write(out/'feature_export_hashes.json',hashes);write(out/'feature_export_gate.json',dict(status='passed',works=24,all40_features_exact=True,harmony_read_by_extractor=False,remaining12_targets_read=False));print('SCORE_EXPORT_PARITY_PASS_24',flush=True)

if __name__=='__main__':gate()
