"""Versioned projection with sound-tie semantics and one MIDI-tick release."""
from pathlib import Path
import time
import mido
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .external_audio_candidates import ASAP
from .external_audio_projection import midi_timing,seconds_to_quarters,attacks_match
from .external_audio_tie_audit import xml_sound_events,midi_intervals

OUT=ROOT/'reports/external_audio_projection_v2'


def sustain_with_one_tick(events,intervals,q0,ticks_per_beat):
    for onset,pitch,duration in events:
        t=q0+onset
        okay=(intervals[:,2]==pitch)&(intervals[:,0]<t-1e-4)&(intervals[:,1]>t+1e-4)&(intervals[:,1]>=t+duration-1/ticks_per_beat-1e-5)
        if not okay.any():return False
    return True


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    root=ROOT/'reports';diagnostic=root/'external_audio_tie_audit/interval_diagnosis.csv'
    table=pd.read_csv(diagnostic);candidates=pd.read_csv(root/'external_audio_candidates/audio_candidates.csv').set_index('asap_performance')
    identity=pd.read_csv(root/'xml_dcml_identity/measure_identity.csv');starts=pd.read_csv(root/'xml_dcml_identity/start_identity.csv');ann=read(ASAP/'asap_annotations.json')
    hashes=read(root/'external_audio_tie_audit/source_hashes.json');hashes.update(read(root/'external_audio_projection/source_hashes.json'))
    for p in (diagnostic,Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_external_audio_projection_v2.py'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());cache={};regions=[];points=[];checks=[]
    for performance,group in table.groupby('performance'):
        r=candidates.loc[performance];folder=r.asap_folder;pid=r.dcml_piece_id
        if folder not in cache:
            mp=ASAP/folder/'midi_score.mid';xp=ASAP/folder/'xml_score.musicxml'
            cache[folder]=(xml_sound_events(xp),*midi_timing(mp),midi_intervals(mp),mido.MidiFile(mp).ticks_per_beat)
        views,tempo,notes,intervals,tpb=cache[folder];a=ann[performance]
        sq=seconds_to_quarters(a['midi_score_beats'],tempo);pb=np.asarray(a['performance_beats']);offset=float(r.audio_offset_seconds)
        known=identity[(identity.folder==folder)&identity.exact_events].set_index('xml_index');labels=starts[(starts.folder==folder)&starts.exact_measure_identity]
        for row in group.itertuples():
            assert time.monotonic()-started<120
            q0=row.score_quarter_start;q1=row.score_quarter_end;event=views[row.xml_index]
            new=(not event['conditional'] and attacks_match(event['attacks'],notes,q0,q1) and sustain_with_one_tick(event['continuations'],intervals,q0,tpb))
            if row.old_pass:assert new
            checks.append(dict(performance=performance,occurrence=row.occurrence,old_pass=row.old_pass,new_pass=new,ticks_per_beat=tpb))
            if not new:continue
            t0,t1=np.interp([q0,q1],sq,pb)+offset;assert 0<=t0<t1<=r.audio_seconds
            common=dict(piece_id=pid,folder=folder,performance=performance,audio_path=r.audio_path,xml_index=row.xml_index,
                source_mc=int(known.loc[row.xml_index,'mc']),occurrence=row.occurrence,score_quarter_start=q0,score_quarter_end=q1,
                audio_start=float(t0),audio_end=float(t1),anchor_source='ASAP reference annotations, not our estimated alignment',
                attack_identity_method='original_full_attack_match' if row.old_pass else 'sound_tie_one_midi_tick')
            regions.append(common)
            for lab in labels[labels.xml_index==row.xml_index].itertuples():
                onset=float(lab.onset_quarters);q=q0+onset
                if not 0<=onset<q1-q0-1e-5:continue
                exact=bool(np.min(abs(sq-q))<1e-4);i=int(np.searchsorted(sq,q));lo=max(0,i-1);hi=min(len(sq)-1,i)
                points.append(dict(**common,label=lab.original_label,source_onset_quarters=onset,score_quarter=q,
                    audio_seconds=float(np.interp(q,sq,pb))+offset,on_provided_beat_anchor=exact,
                    bracket_score_quarters=float(sq[hi]-sq[lo]),reference_method='provided_beat_anchor' if exact else 'linear_between_provided_beats',complete_phrase_training_mask=False))
    regions=pd.DataFrame(regions);points=pd.DataFrame(points);checks=pd.DataFrame(checks)
    regions.to_csv(OUT/'verified_bar_intervals.csv',index=False);points.to_csv(OUT/'projected_start_references.csv',index=False);checks.to_csv(OUT/'acceptance_audit.csv',index=False)
    old=pd.read_csv(root/'external_audio_projection/verified_bar_intervals.csv')
    assert set(zip(old.performance,old.occurrence))<=set(zip(regions.performance,regions.occurrence))
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    result=dict(status='reference_projection_audited_not_training_ready',version=2,performances_with_projected_starts=int(points.performance.nunique()),
        movements_with_projected_starts=int(points.piece_id.nunique()),verified_bar_intervals=len(regions),
        added_bar_intervals=len(regions)-len(old),projected_start_occurrences=len(points),
        unique_source_starts=len(points[['piece_id','source_mc','source_onset_quarters']].drop_duplicates()),
        exact_beat_anchor_occurrences=int(points.on_provided_beat_anchor.sum()),old_verified_intervals_preserved=True,
        complete_training_masks=False,training_runs=0,source_hashes_unchanged=True,wall_seconds=time.monotonic()-started)
    write(OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
