"""Conservative measure identity audit: exact events AND unique printed bar number."""
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import re,time,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .external_audio_candidates import ASAP

OUT=ROOT/'reports/xml_dcml_identity'


def rational(v):return float(Fraction(str(v)))
def number(v):
    s=str(v)
    return int(float(s)) if re.fullmatch(r'-?\d+(?:\.0+)?',s) else None


def xml_measures(path):
    # ElementTree does not retrieve the external MusicXML DTD.
    root=ET.parse(path).getroot();parts=root.findall('part');assert parts
    collected=defaultdict(list);nums={};lengths=defaultdict(float)
    for part in parts:
        divisions=1
        for index,measure in enumerate(part.findall('measure')):
            label=measure.get('number','');assert index not in nums or nums[index]==label
            nums[index]=label;cursor=0.;last_onset=0.;extent=0.
            for node in measure:
                if node.tag=='attributes':
                    d=node.findtext('divisions')
                    if d:divisions=int(d)
                elif node.tag in ('backup','forward'):
                    cursor+=(1 if node.tag=='forward' else -1)*float(node.findtext('duration','0'))/divisions
                elif node.tag=='note':
                    if node.find('grace') is not None:continue
                    duration=float(node.findtext('duration','0'))/divisions
                    chord=node.find('chord') is not None
                    onset=last_onset if chord else cursor
                    pitch=node.find('pitch')
                    if pitch is not None and duration>0:
                        semitone={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[pitch.findtext('step')]
                        value=12*(int(pitch.findtext('octave'))+1)+semitone+int(float(pitch.findtext('alter','0')))
                        collected[index].append((round(onset,6),value,round(duration,6)))
                    if not chord:cursor+=duration;last_onset=onset
                    extent=max(extent,onset+duration)
            lengths[index]=max(lengths[index],extent)
    return [dict(index=i,printed_number=nums[i],number=number(nums[i]),signature=tuple(sorted(collected[i])),extent=lengths[i]) for i in sorted(nums)]


def dcml_measures(notes,measures):
    groups=defaultdict(list)
    for _,r in notes.iterrows():
        if pd.notna(r.get('gracenote')) or not np.isfinite(float(r.midi)):continue
        duration=float(r.duration_qb)
        if duration>0:groups[int(r.mc)].append((round(4*rational(r.mc_onset),6),int(r.midi),round(duration,6)))
    return [dict(mc=int(r.mc),number=number(r.mn),signature=tuple(sorted(groups[int(r.mc)])),extent=float(r.duration_qb)) for _,r in measures.iterrows()]


def compare(xml,dcml):
    xc=Counter(x['number'] for x in xml);dc=Counter(x['number'] for x in dcml)
    lookup={x['number']:x for x in xml if x['number'] is not None and xc[x['number']]==1}
    rows=[]
    for d in dcml:
        x=lookup.get(d['number']);unique=d['number'] is not None and dc[d['number']]==1
        exact=bool(unique and x and d['signature'] and d['signature']==x['signature'])
        a=Counter(d['signature']);b=Counter(x['signature']) if x else Counter()
        shared=sum((a&b).values());den=sum(a.values())+sum(b.values())
        rows.append(dict(mc=d['mc'],printed_number=d['number'],xml_index=x['index'] if x else -1,
            unique_number=bool(unique and x),exact_events=exact,event_dice=2*shared/den if den else 0,
            dcml_notes=len(d['signature']),xml_notes=len(x['signature']) if x else 0,
            dcml_duration=d['extent'],xml_duration=x['extent'] if x else None))
    return rows


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    candidates=ROOT/'reports/external_audio_candidates/audio_candidates.csv'
    df=pd.read_csv(candidates);hashes={str(candidates):sha(candidates),str(Path(__file__)):sha(Path(__file__))};summary=[];allrows=[];labels=[]
    for (pid,folder),group in df.groupby(['dcml_piece_id','asap_folder']):
        assert time.monotonic()-began<900
        corpus,piece=pid.rsplit('_',1);source=ROOT/'external_data/dcml_expansion_20260913'/corpus
        npth=source/'notes'/f'{piece}.notes.tsv';mpth=source/'measures'/f'{piece}.measures.tsv';hpth=source/'harmonies'/f'{piece}.harmonies.tsv';xpath=ASAP/folder/'xml_score.musicxml'
        for p in (npth,mpth,hpth,xpath):hashes[str(p)]=sha(p)
        xml=xml_measures(xpath);dcml=dcml_measures(pd.read_csv(npth,sep='\t'),pd.read_csv(mpth,sep='\t'))
        rows=compare(xml,dcml);mapping={r['mc']:r for r in rows};allrows.extend([dict(piece_id=pid,folder=folder,**r) for r in rows])
        h=pd.read_csv(hpth,sep='\t');starts=h[h.phraseend.fillna('').str.contains('{',regex=False)]
        matched=0
        for _,r in starts.iterrows():
            match=mapping.get(int(r.mc),{});ok=bool(match.get('exact_events',False));matched+=ok
            labels.append(dict(piece_id=pid,folder=folder,mc=int(r.mc),xml_index=match.get('xml_index',-1),
                onset_quarters=4*rational(r.mc_onset),exact_measure_identity=ok,original_label=r.label,
                phrase_audio_coordinates_verified=False))
        summary.append(dict(piece_id=pid,folder=folder,dcml_measures=len(dcml),xml_measures=len(xml),
            exact_identity_measures=sum(r['exact_events'] for r in rows),source_start_rows=len(starts),
            starts_in_exact_identity_measures=matched,audio_candidate_performances=len(group)))
        print('IDENTITY',pid,folder,matched,'/',len(starts),flush=True)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(allrows).to_csv(OUT/'measure_identity.csv',index=False);pd.DataFrame(labels).to_csv(OUT/'start_identity.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',score_versions=len(summary),exact_comparison_only=True,
        verified_phrase_audio_pairs=0,training_runs=0,source_hashes_unchanged=True,seconds=time.monotonic()-began,
        caution='Matching bar content does not yet verify repeats, MIDI-to-audio time coordinates or the unlabelled-region mask. Do not train by treating unmatched bars as negatives.'))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__':main()
