"""Conservative positive-reference projection through provided ASAP beat anchors.

Not current-model predictions; not a complete training set or complete label mask.
"""
from collections import Counter
from pathlib import Path
import time
import mido
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .external_audio_candidates import ASAP
from .xml_dcml_identity_audit import xml_measures

OUT=ROOT/'reports/external_audio_projection'


def midi_timing(path):
    midi=mido.MidiFile(path);tpb=midi.ticks_per_beat;tick=0;seconds=0.;tempo=500000
    tempos=[(0.,0.,tempo)];notes=[]
    for msg in mido.merge_tracks(midi.tracks):
        tick+=msg.time;seconds+=mido.tick2second(msg.time,tpb,tempo)
        if msg.type=='set_tempo':
            tempo=msg.tempo
            if abs(tempos[-1][0]-seconds)<1e-12:tempos[-1]=(seconds,tick/tpb,tempo)
            else:tempos.append((seconds,tick/tpb,tempo))
        if msg.type=='note_on' and msg.velocity>0:notes.append((tick/tpb,msg.note))
    return np.asarray(tempos,float),np.asarray(notes,float)


def seconds_to_quarters(seconds,tempos):
    s=np.asarray(seconds,float);i=np.searchsorted(tempos[:,0],s,side='right')-1
    assert (i>=0).all()
    return tempos[i,1]+(s-tempos[i,0])*1e6/tempos[i,2]


def attacks_match(signature,notes,q0,q1):
    use=notes[(notes[:,0]>=q0-1e-4)&(notes[:,0]<q1-1e-4)]
    left=Counter((round(o*96),p) for o,p,d in signature)
    right=Counter((round((o-q0)*96),int(p)) for o,p in use)
    # Quantization is only for matching export rounding, not for time projection.
    residual=max((abs((o-q0)-round((o-q0)*96)/96) for o,p in use),default=0.)
    return bool(left and left==right and residual<1e-3)


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    cp=ROOT/'reports/external_audio_candidates/audio_candidates.csv';mp=ROOT/'reports/xml_dcml_identity/measure_identity.csv';lp=ROOT/'reports/xml_dcml_identity/start_identity.csv'
    candidates=pd.read_csv(cp);identity=pd.read_csv(mp);labels=pd.read_csv(lp);annpath=ASAP/'asap_annotations.json';ann=read(annpath)
    hashes={str(p):sha(p) for p in (Path(__file__),cp,mp,lp,annpath,OUT/'PROTOCOL.md',ROOT/'src/xml_dcml_identity_audit.py')}
    cache={};summary=[];regions=[];points=[];reject=[]
    for _,row in candidates.iterrows():
        assert time.monotonic()-began<1200
        folder=row.asap_folder;pid=row.dcml_piece_id;key=row.asap_performance;a=ann[key]
        if folder not in cache:
            xmlp=ASAP/folder/'xml_score.musicxml';midp=ASAP/folder/'midi_score.mid'
            for p in (xmlp,midp):hashes[str(p)]=sha(p)
            cache[folder]=(xml_measures(xmlp),*midi_timing(midp))
        xml,tempo,notes=cache[folder]
        sb=np.asarray(a['midi_score_beats'],float);pb=np.asarray(a['performance_beats'],float);db=np.asarray(a['midi_score_downbeats'],float)
        mapping=a['downbeats_score_map'];assert len(db)==len(mapping)
        if len(sb)!=len(pb) or not ((np.diff(sb)>0).all() and (np.diff(pb)>0).all()):
            reject.append(dict(performance=key,reason='nonmonotonic_or_unaligned_anchors'));continue
        sq=seconds_to_quarters(sb,tempo);dq=seconds_to_quarters(db,tempo)
        if not (np.diff(sq)>0).all():
            reject.append(dict(performance=key,reason='nonmonotonic_score_quarters'));continue
        known=identity[(identity.folder==folder)&identity.exact_events].set_index('xml_index')
        starts=labels[(labels.folder==folder)&labels.exact_measure_identity]
        nregions=0;npoints=0;reasons=Counter()
        for j in range(len(db)-1):
            raw=mapping[j]
            if not isinstance(raw,(int,float)) or not np.isfinite(raw) or int(raw)!=raw:
                reasons['split_or_unknown_bar_map']+=1;continue
            index=int(raw)
            if index not in known.index or not 0<=index<len(xml):
                reasons['no_exact_source_bar_identity']+=1;continue
            q0,q1=dq[j:j+2];measure=xml[index]
            if abs(q1-q0-measure['extent'])>1e-3:
                reasons['bar_duration_or_repeat_mismatch']+=1;continue
            if not attacks_match(measure['signature'],notes,q0,q1):
                reasons['xml_midi_note_attack_mismatch']+=1;continue
            if q0<sq[0]-1e-5 or q1>sq[-1]+1e-5:
                reasons['outside_beat_anchor_range']+=1;continue
            offset=float(row.audio_offset_seconds);t0=float(np.interp(q0,sq,pb))+offset;t1=float(np.interp(q1,sq,pb))+offset
            if not 0<=t0<t1<=float(row.audio_seconds):
                reasons['outside_audio_file']+=1;continue
            common=dict(piece_id=pid,folder=folder,performance=key,audio_path=row.audio_path,xml_index=index,
                source_mc=int(known.loc[index,'mc']),occurrence=j,score_quarter_start=q0,score_quarter_end=q1,
                audio_start=t0,audio_end=t1,anchor_source='ASAP reference annotations, not our estimated alignment')
            regions.append(common);nregions+=1
            for _,lab in starts[starts.xml_index==index].iterrows():
                onset=float(lab.onset_quarters);target=q0+onset
                if not 0<=onset<q1-q0-1e-5:continue
                seconds=float(np.interp(target,sq,pb))+offset
                k=int(np.searchsorted(sq,target));exact=bool(np.min(abs(sq-target))<1e-4)
                lo=max(0,k-1);hi=min(len(sq)-1,k)
                points.append(dict(**common,label=lab.original_label,source_onset_quarters=onset,score_quarter=target,
                    audio_seconds=seconds,on_provided_beat_anchor=exact,bracket_score_quarters=float(sq[hi]-sq[lo]),
                    reference_method='provided_beat_anchor' if exact else 'linear_between_provided_beats',
                    complete_phrase_training_mask=False));npoints+=1
        summary.append(dict(piece_id=pid,performance=key,verified_bar_intervals=nregions,projected_start_references=npoints,**dict(reasons)))
    pd.DataFrame(summary).to_csv(OUT/'performance_audit.csv',index=False)
    pd.DataFrame(regions).to_csv(OUT/'verified_bar_intervals.csv',index=False)
    pd.DataFrame(points).to_csv(OUT/'projected_start_references.csv',index=False)
    pd.DataFrame(reject,columns=['performance','reason']).to_csv(OUT/'rejected_performances.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    p=pd.DataFrame(points)
    result=dict(status='reference_projection_audited_not_training_ready',performances_examined=len(candidates),
        performances_with_projected_starts=p.performance.nunique() if len(p) else 0,
        movements_with_projected_starts=p.piece_id.nunique() if len(p) else 0,projected_start_occurrences=len(points),
        unique_source_starts=len(p[['piece_id','source_mc','source_onset_quarters']].drop_duplicates()) if len(p) else 0,
        exact_beat_anchor_occurrences=int(p.on_provided_beat_anchor.sum()) if len(p) else 0,
        verified_bar_intervals=len(regions),training_runs=0,complete_training_masks=False,
        wall_seconds=time.monotonic()-began,
        caution='Repeated occurrences and performances are not independent phrases. Initial/final label semantics and unknown masks are not yet adapted. Within-beat audio times are interpolated, not annotated exact onsets. Not current-model accuracy.')
    write(OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
