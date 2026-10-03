"""Explain conservative XML/MIDI attack rejection without changing old labels."""
from collections import defaultdict,deque
from pathlib import Path
import time,xml.etree.ElementTree as ET
import mido
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .external_audio_candidates import ASAP
from .xml_dcml_identity_audit import xml_measures
from .external_audio_projection import midi_timing,seconds_to_quarters,attacks_match

OUT=ROOT/'reports/external_audio_tie_audit'


def xml_sound_events(path):
    full=defaultdict(list);attacks=defaultdict(list);continuations=defaultdict(list);conditional=set()
    for part in ET.parse(path).getroot().findall('part'):
        divisions=1
        for index,measure in enumerate(part.findall('measure')):
            cursor=0.;last=0.
            for node in measure:
                if node.tag=='attributes':divisions=int(node.findtext('divisions',str(divisions)))
                elif node.tag in ('backup','forward'):cursor+=(1 if node.tag=='forward' else -1)*float(node.findtext('duration','0'))/divisions
                elif node.tag=='note':
                    if node.find('grace') is not None:continue
                    duration=float(node.findtext('duration','0'))/divisions;chord=node.find('chord') is not None
                    onset=last if chord else cursor;pitch=node.find('pitch');ties=node.findall('tie')
                    if any(t.get('time-only') is not None for t in ties):conditional.add(index)
                    if pitch is not None and duration>0:
                        semitone={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[pitch.findtext('step')]
                        value=12*(int(pitch.findtext('octave'))+1)+semitone+int(float(pitch.findtext('alter','0')))
                        event=(round(onset,6),value,round(duration,6));full[index].append(event)
                        (continuations if any(t.get('type')=='stop' for t in ties) else attacks)[index].append(event)
                    if not chord:cursor+=duration;last=onset
    return {i:dict(full=tuple(sorted(full[i])),attacks=tuple(sorted(attacks[i])),continuations=tuple(continuations[i]),conditional=i in conditional) for i in full}


def midi_intervals(path):
    midi=mido.MidiFile(path);active=defaultdict(deque);tick=0;intervals=[]
    for msg in mido.merge_tracks(midi.tracks):
        tick+=msg.time;q=tick/midi.ticks_per_beat
        if msg.type=='note_on' and msg.velocity>0:active[(msg.channel,msg.note)].append(q)
        elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
            key=(msg.channel,msg.note)
            if active[key]:intervals.append((active[key].popleft(),q,msg.note))
    if any(active.values()):raise ValueError('unclosed MIDI note intervals')
    return np.asarray(intervals,float).reshape(-1,3)


def continuations_sustain(events,intervals,q0):
    for onset,pitch,duration in events:
        t=q0+onset
        present=(intervals[:,2]==pitch)&(intervals[:,0]<t-1e-4)&(intervals[:,1]>=t+duration-1e-3)
        if not present.any():return False
    return True


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    cp=ROOT/'reports/external_audio_candidates/audio_candidates.csv';ip=ROOT/'reports/xml_dcml_identity/measure_identity.csv';lp=ROOT/'reports/xml_dcml_identity/start_identity.csv'
    candidates=pd.read_csv(cp);identity=pd.read_csv(ip);labels=pd.read_csv(lp);ann=read(ASAP/'asap_annotations.json')
    hashes=read(ROOT/'reports/external_audio_projection/source_hashes.json');assert all(sha(p)==h for p,h in hashes.items())
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_external_audio_tie_audit.py'):hashes[str(p)]=sha(p)
    cache={};rows=[]
    for _,r in candidates.iterrows():
        assert time.monotonic()-began<120
        folder=r.asap_folder
        if folder not in cache:
            xp=ASAP/folder/'xml_score.musicxml';mp=ASAP/folder/'midi_score.mid'
            cache[folder]=(xml_measures(xp),xml_sound_events(xp),*midi_timing(mp),midi_intervals(mp))
            hashes[str(xp)]=sha(xp);hashes[str(mp)]=sha(mp)
            for measure in cache[folder][0]:assert measure['signature']==cache[folder][1].get(measure['index'],{'full':()})['full']
        xml,events,tempo,notes,intervals=cache[folder];a=ann[r.asap_performance]
        sq=seconds_to_quarters(a['midi_score_beats'],tempo);pb=np.asarray(a['performance_beats']);dq=seconds_to_quarters(a['midi_score_downbeats'],tempo)
        known=set(identity[(identity.folder==folder)&identity.exact_events].xml_index)
        starts=labels[(labels.folder==folder)&labels.exact_measure_identity]
        for j in range(len(dq)-1):
            index=a['downbeats_score_map'][j]
            if not isinstance(index,(int,float)) or not np.isfinite(index) or int(index)!=index:continue
            index=int(index)
            if index not in known or index not in events:continue
            q0,q1=dq[j:j+2];measure=xml[index];event=events[index]
            if abs(q1-q0-measure['extent'])>1e-3 or q0<sq[0]-1e-5 or q1>sq[-1]+1e-5:continue
            t0,t1=np.interp([q0,q1],sq,pb)+float(r.audio_offset_seconds)
            if not 0<=t0<t1<=float(r.audio_seconds):continue
            old=attacks_match(measure['signature'],notes,q0,q1)
            corrected=(not event['conditional'] and attacks_match(event['attacks'],notes,q0,q1) and continuations_sustain(event['continuations'],intervals,q0))
            rows.append(dict(piece_id=r.dcml_piece_id,performance=r.asap_performance,folder=folder,xml_index=index,occurrence=j,
                score_quarter_start=q0,score_quarter_end=q1,old_pass=old,tie_aware_pass=corrected,
                recovered=bool(corrected and not old),continuation_notes=len(event['continuations']),conditional_tie=event['conditional'],
                source_start_rows=int((starts.xml_index==index).sum())))
    table=pd.DataFrame(rows);table.to_csv(OUT/'interval_diagnosis.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    recovered=table[table.recovered]
    result=dict(status='diagnostic_complete_no_projection_changed',intervals_examined=len(table),original_attack_pass=int(table.old_pass.sum()),
        recovered_intervals=len(recovered),recovered_start_occurrences=int(recovered.source_start_rows.sum()),
        recovered_score_versions=int(recovered.piece_id.nunique()),old_pass_new_fail=int((table.old_pass&~table.tie_aware_pass).sum()),
        remaining_failed=int((~table.tie_aware_pass).sum()),training_runs=0,source_hashes_unchanged=True,seconds=time.monotonic()-began)
    write(OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
