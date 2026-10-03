"""Source-only Op50No2 repeat itinerary and raw-curve export parity."""
import sys
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
from src.data import read_csv_clean,align_dynamics_to_beats,build_curve_features,discover_dcml_pieces,to_float
from src.slice_energy_study import DCML
from src.slice_energy_features import tempo_hierarchy
from src.three_round_round2 import assemble_extra
from src.note_relation_graph import merged_events
from src.motif_recurrence_features import features
from src.score_context_study import read,write,sha
from score_semantics_audit import cues_from_events
from score_only_export_training import data_u,guard
from score_version_overlap_audit import xml_bars
OUT=BASE/'score_repeat_bridge';ART=ROOT/'artifacts/mentor_score_repeat_bridge_20260916';PID='chopin_op50_no2'

def curve_export(row):
    beat=read_csv_clean(Path(row.beat_time_path));dyn=read_csv_clean(Path(row.beat_dyn_path));aligned=align_dynamics_to_beats(beat,dyn);ids=[];curves=[]
    for pid in beat.columns:
        if pid in ('measure_number','beat_number') or pid.lower().startswith('index'):continue
        times=pd.to_numeric(beat[pid],errors='coerce').to_numpy(float)
        if np.isfinite(times).mean()<.8:continue
        dynamics=pd.to_numeric(aligned[pid],errors='coerce').to_numpy(float) if pid in aligned else np.full(len(beat),np.nan)
        curves.append(build_curve_features(times,dynamics));ids.append(pid)
    curves=np.stack(curves);energy=np.stack([tempo_hierarchy(np.exp(c[:,0].astype(float)),c[:,7])[0] for c in curves]);extra=assemble_extra(curves,energy,'B')
    return beat,curves,extra,np.asarray(ids)

def main():
    guard(True);OUT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True);manifest=pd.read_csv(ROOT/'manifests/piece_manifest.csv').set_index('piece_id');_,known=data_u();sources={str(Path(__file__)):sha(Path(__file__)),str(BASE/'score_version_monotonic/all_optimal_matches.csv'):sha(BASE/'score_version_monotonic/all_optimal_matches.csv')};parity=[]
    for pid,v in known.items():
        row=manifest.loc[pid]
        for p in (row.beat_time_path,row.beat_dyn_path):sources[str(p)]=sha(p)
        beat,c,e,ids=curve_export(row);np.testing.assert_array_equal(ids,v['performance_ids']);np.testing.assert_array_equal(c,v['curves'][...,:9]);np.testing.assert_array_equal(e,v['curves'][...,25:34]);parity.append(dict(piece_id=pid,performances=len(ids),base9_error=0,extra9_error=0))
    pd.DataFrame(parity).to_csv(OUT/'curve_export_parity.csv',index=False)
    row=manifest.loc[PID];piece=discover_dcml_pieces(DCML)[PID]
    for p in (row.xml_score_path,row.beat_time_path,row.beat_dyn_path,piece.notes_path,piece.measures_path):sources[str(p)]=sha(p)
    bridge=pd.read_csv(BASE/'score_version_monotonic/all_optimal_matches.csv');bridge=bridge[bridge.piece_id==PID].sort_values('xml_index');assert len(bridge)==127 and bridge.unique_mc_duration_verified.all() and bridge.full_xml_subsequence_exists.all()
    xb,xml_events=xml_bars(Path(row.xml_score_path));beat,c,e,ids=curve_export(row);assert len(xb)==127 and len(beat)==381
    np.testing.assert_array_equal(beat.measure_number,np.repeat(np.arange(1,128),3));np.testing.assert_array_equal(beat.beat_number,np.tile(np.arange(3),127));assert all(Fraction(b['duration'])==3 and int(b['number'])==j+1 for j,b in enumerate(xb))
    notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t').set_index('mc');events=[];ties=[];rows=[];xml_all=[]
    for r in bridge.itertuples():
        j=int(r.xml_index);mc=int(r.unique_mc);source=notes[notes.mc==mc];mr=measures.loc[mc];assert 4*to_float(mr.mc_offset)==0 and 4*to_float(mr.act_dur)==3
        actual={(round(4*to_float(n.mc_onset),6),round(float(n.duration_qb),6),int(n.midi)) for n in source.itertuples() if n.duration_qb>0};reference={(round(float(Fraction(x['onset'])),6),round(float(Fraction(x['duration'])),6),int(x['midi'])) for x in xml_events if x['measure_index']==j};assert actual==reference
        for n in source.itertuples():
            d=to_float(n.duration_qb);q=j*3+4*to_float(n.mc_onset);assert np.isfinite(d) and j*3<=q<(j+1)*3
            events.append((q,max(0,d),int(n.midi),int(n.staff),int(n.voice)));tie=n.tied;ties.append(None if pd.isna(tie) else int(tie))
        rows.append(dict(xml_index=j,xml_measure=j+1,dcml_mc=mc,start_qb=j*3,duration_qb=3,note_fingerprint_equal=True))
    merged,count=merged_events(events,ties);merged=[v for v in merged if v[0]<381 and v[0]+v[1]>0];score=cues_from_events(piece,beat.beat_number.to_numpy(),beat.measure_number.to_numpy(),merged);motif=features(merged,381,True);x=np.concatenate([c,np.broadcast_to(score,(*c.shape[:2],16)),e,np.broadcast_to(motif,(*c.shape[:2],24))],axis=-1);assert x.shape==(len(ids),381,58) and np.isfinite(x).all()
    dest=ART/'op50_no2_features.npz';values=dict(curves=x,performance_ids=ids,beat_number=beat.beat_number.to_numpy(),measure_number=beat.measure_number.to_numpy(),note_events=np.asarray(merged),score16=score,motif24=motif)
    if dest.exists():
        with np.load(dest,allow_pickle=False) as z:
            for k,v in values.items():np.testing.assert_array_equal(v,z[k])
    else:np.savez_compressed(dest,**values)
    sources[str(dest)]=sha(dest);pd.DataFrame(rows).to_csv(OUT/'bridge.csv',index=False);sources[str(OUT/'bridge.csv')]=sha(OUT/'bridge.csv');assert all(sha(p)==h for p,h in sources.items());write(OUT/'source_hashes.json',sources);write(OUT/'completion_audit.json',dict(status='complete',curve18_exact_parity_works=24,piece_id=PID,beats=381,performances=len(ids),bars=127,all_unique_monotonic_mc=True,all_onset_duration_pitch_equal=True,merged_ties=count,labels_read=False,predictions_read=False,waveform_alignment_validated=False,features_frozen=True));print(read(OUT/'completion_audit.json'))

if __name__=='__main__':main()
