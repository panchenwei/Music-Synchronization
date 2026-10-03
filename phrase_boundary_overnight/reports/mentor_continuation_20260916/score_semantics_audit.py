"""Rebuild legacy score cues exactly, then inspect local/tie-aware alternatives."""
import os,sys,time,json,hashlib,traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src import phase3_features as core
from src.score_context_study import read,write,sha
from src.mentor_sequence_study import splits
from src.slice_energy_study import DCML
from src.data import discover_dcml_pieces
from src.slice_energy_features import corrected_score,voiced_events
from src.score_local_coordinates import local_events
from src.note_relation_graph import merged_events
BASE=Path(__file__).parent;OUT=BASE/'score_semantics_audit';ART=ROOT/'artifacts/mentor_score_semantics_20260916/cache'

def cues_from_events(piece,bn,mn,events):
    original=core._events_from_dcml
    try:
        core._events_from_dcml=lambda p,n: ([(float(e[0]),float(e[1]),int(e[2])) for e in events],'audited_event_source',0.)
        old=core.build_compact_score_cues(piece,mn,bn).cues
    finally:core._events_from_dcml=original
    events=[(float(e[0]),float(e[1]),int(e[2]),int(e[3]),int(e[4])) for e in events]
    return corrected_score(old,events)[1]

def main():
    start=time.monotonic();b=read(BASE/'BUDGET.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(b['observed_at_utc'].replace('Z','+00:00'))).total_seconds();assert age<1800 and b['observed_used_percent']<b['stop_new_runs_used_percent']
    OUT.mkdir(exist_ok=True);ART.mkdir(parents=True,exist_ok=True);frame,_=splits();ids=sorted(frame.piece_id.unique());assert len(ids)==24
    pieces=discover_dcml_pieces(DCML);parent=read(BASE/'sequence_v2/sequence_contract.json');sources={**parent['sources'],str(Path(__file__)):sha(Path(__file__)),str(OUT/'PROTOCOL.md'):sha(OUT/'PROTOCOL.md')}
    for pid in ids:
        for p in [pieces[pid].notes_path,pieces[pid].measures_path,ROOT/'artifacts/slice_energy_study/cache'/f'{pid}.npz',ROOT/'artifacts/note_relation_study/cache'/f'{pid}.npz']:sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items());contract=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest();write(OUT/'contract.json',dict(contract=contract,sources=sources))
    rows=[];columns=[];desthash={}
    for pid in ids:
        assert time.monotonic()-start<1200;piece=pieces[pid]
        with np.load(ROOT/'artifacts/slice_energy_study/cache'/f'{pid}.npz',allow_pickle=False) as z:bn=z['beat_number'].copy();mn=z['measure_number'].copy();old=z['old_score'].copy();fixed=z['fixed_score'].copy()
        n=len(bn);original=core.build_compact_score_cues(piece,mn,bn).cues;np.testing.assert_array_equal(original,old)
        legacy=corrected_score(original,voiced_events(piece,n))[1];np.testing.assert_array_equal(legacy,fixed)
        notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t');local,ties=local_events(notes,measures,n);merged,count=merged_events(local,ties);merged=[e for e in merged if e[0]<n and e[0]+e[1]>0]
        with np.load(ROOT/'artifacts/note_relation_study/cache'/f'{pid}.npz',allow_pickle=False) as z:np.testing.assert_array_equal(np.asarray(merged),z['note_events'])
        a=cues_from_events(piece,bn,mn,local);t=cues_from_events(piece,bn,mn,merged)
        for value in (a,t):assert value.shape==(n,16) and np.isfinite(value).all();np.testing.assert_array_equal(value[:,:3],fixed[:,:3])
        dest=ART/f'{pid}.npz'
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:np.testing.assert_array_equal(z['L'],a);np.testing.assert_array_equal(z['T'],t)
        else:np.savez_compressed(dest,L=a,T=t)
        desthash[str(dest)]=sha(dest)
        for name,x,y in [('local_minus_old',a,fixed),('tied_minus_local',t,a),('tied_minus_old',t,fixed)]:
            delta=np.abs(x-y);rows.append(dict(piece_id=pid,comparison=name,beats=n,changed_beats=int(np.any(delta>1e-6,axis=1).sum()),changed_cells=int((delta>1e-6).sum()),max_difference=float(delta.max()),merged_ties=count))
            columns.extend(dict(piece_id=pid,comparison=name,column=i,name=label,changed_beats=int((delta[:,i]>1e-6).sum()),max_difference=float(delta[:,i].max())) for i,label in enumerate(core.SCORE_CUE_NAMES))
        print('SCORE_INPUT_AUDITED',pid,'local',int(np.any(abs(a-fixed)>1e-6,axis=1).sum()),'tied',int(np.any(abs(t-a)>1e-6,axis=1).sum()),flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'piece_changes.csv',index=False);pd.DataFrame(columns).to_csv(OUT/'column_changes.csv',index=False)
    summaries={}
    for name,g in f.groupby('comparison'):summaries[name]=dict(affected_works=int((g.changed_beats>0).sum()),changed_beats=int(g.changed_beats.sum()),changed_cells=int(g.changed_cells.sum()),max_difference=float(g.max_difference.max()))
    assert all(sha(p)==h for p,h in sources.items());write(OUT/'cache_hashes.json',desthash);write(OUT/'summary.json',dict(status='complete',works=24,total_beats=int(f[f.comparison=='local_minus_old'].beats.sum()),seconds=time.monotonic()-start,comparisons=summaries))
    write(OUT/'completion_audit.json',dict(status='complete',legacy_old16_rebuilt_exact=True,legacy_fixed16_rebuilt_exact=True,merged_events_equal_existing_motif_source=True,metric_columns_unchanged=True,historical_source_cache_hashes_unchanged=True,labels_read_for_derivation=False,predictions_accessed=False,training_runs=0))

if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid()));raise
