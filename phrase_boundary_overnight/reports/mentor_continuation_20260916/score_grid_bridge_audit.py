"""Label-blind bar/beat coordinate bridge; audit only, never alter caches."""
import sys
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data import discover_dcml_pieces,choose_measure_path
from src.slice_energy_study import DCML
from src.score_context_study import write,sha
BASE=Path(__file__).parent;OUT=BASE/'score_only_compatibility/grid_bridge'

def frac(x):return Fraction(str(x))

def bridge(piece,beats):
    measures=pd.read_csv(piece.measures_path,sep='\t');path,mode,error=choose_measure_path(measures,len(beats));assert path
    lookup=measures.set_index('mc');occ=[];start=Fraction(0)
    for i,o in enumerate(path):
        r=lookup.loc[o['mc']];duration=4*frac(r.act_dur);phase=4*frac(r.mc_offset) if pd.notna(r.mc_offset) else Fraction(0)
        occ.append(dict(index=i,mc=o['mc'],mn=int(r.mn),start=start,end=start+duration,phase=phase))
        start+=duration
    # Earliest nonnegative coordinate in the first score measure with CSV's
    # source-labelled beat-in-bar. No label, length-derived or F1-derived offset.
    q0=frac(beats.beat_number.iloc[0])-occ[0]['phase']
    assert 0<=q0<occ[0]['end']
    rows=[];cursor=0
    for i,r in enumerate(beats.itertuples()):
        q=q0+i
        while cursor<len(occ) and q>=occ[cursor]['end']:cursor+=1
        if cursor==len(occ):
            rows.append(dict(grid_index=i,score_q=str(q),covered=False));continue
        o=occ[cursor];phase=o['phase']+q-o['start']
        rows.append(dict(grid_index=i,score_q=str(q),covered=True,mc=o['mc'],score_mn=o['mn'],score_occurrence=o['index'],score_beat=str(phase),curve_measure=float(r.measure_number),curve_beat=float(r.beat_number),phase_equal=phase==frac(r.beat_number)))
    f=pd.DataFrame(rows);covered=f[f.covered].copy()
    phase_ok=bool(covered.phase_equal.all());coverage_ok=bool(f.covered.all())
    transitions=np.flatnonzero(np.diff(covered.score_occurrence.to_numpy())!=0)+1
    csv_transitions=np.flatnonzero(np.diff(covered.curve_measure.to_numpy())!=0)+1
    transition_ok=bool(np.array_equal(transitions,csv_transitions))
    delta=covered.curve_measure.to_numpy()-covered.score_occurrence.to_numpy()
    ordinal_ok=bool(len(set(delta))==1)
    return f,dict(origin_qb=str(q0),mode=mode,length_error=error,rows=len(f),covered_rows=len(covered),phase_equal=phase_ok,bar_transitions_equal=transition_ok,bar_ordinal_constant=ordinal_ok,bar_ordinal_offsets=sorted(set(delta)),all_rows_covered=coverage_ok,passed=coverage_ok and phase_ok and transition_ok and ordinal_ok)

def main():
    OUT.mkdir(parents=True,exist_ok=True);pieces=discover_dcml_pieces(DCML);manifest=pd.read_csv(ROOT/'manifests/piece_manifest.csv');rows=[];hashes={str(Path(__file__)):sha(Path(__file__)),str(ROOT/'manifests/piece_manifest.csv'):sha(ROOT/'manifests/piece_manifest.csv')}
    allowed={p.stem for p in (ROOT/'artifacts/slice_energy_study/cache').glob('*.npz')};assert len(allowed)==43
    for e in manifest[manifest.piece_id.isin(allowed)].itertuples():
        piece=pieces[e.piece_id];beatpath=Path(e.beat_time_path)
        for p in (beatpath,piece.measures_path):hashes[str(p)]=sha(p)
        beats=pd.read_csv(beatpath);f,meta=bridge(piece,beats);f.to_csv(OUT/f'{e.piece_id}.csv',index=False);rows.append(dict(piece_id=e.piece_id,**meta))
    result=pd.DataFrame(rows);result.to_csv(OUT/'all43.csv',index=False);assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes);write(OUT/'summary.json',dict(status='complete',works=len(result),passed=int(result.passed.sum()),failures=result[~result.passed].to_dict('records'),nonzero_origins=result[result.origin_qb!='0'].to_dict('records'),labels_accessed=False,predictions_accessed=False,cache_modified=False,audio_timing_verified=False))
    print(result[['piece_id','origin_qb','passed']].to_string(index=False),flush=True)

if __name__=='__main__':main()
