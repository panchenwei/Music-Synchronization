"""Read-only metadata/audio-header inventory, NOT a verified phrase-to-audio map."""
import re,time
from pathlib import Path
import pandas as pd
import soundfile as sf
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_audio_candidates'
ASAP=ROOT.parent/'dataset/asap-dataset';MAESTRO=ROOT.parent/'dataset/maestro-v2.0.0'


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    metadata=ASAP/'metadata.csv';annotations=ASAP/'asap_annotations.json'
    table=pd.read_csv(metadata).fillna('');ann=read(annotations)
    audit=pd.read_csv(ROOT/'reports/external_recurrence_data/piece_audit.csv')
    available=set(audit[audit.status=='usable'].piece_id)
    lookup={};sources={str(metadata):sha(metadata),str(annotations):sha(annotations)}
    for corpus,composer in [('beethoven_piano_sonatas','Beethoven'),('mozart_piano_sonatas','Mozart')]:
        path=ROOT/'external_data/dcml_expansion_20260913'/corpus/'metadata.tsv';sources[str(path)]=sha(path)
        for _,row in pd.read_csv(path,sep='\t').fillna('').iterrows():
            match=re.search(r'(?:Sonata no\.|Sonata No\.)\s*(\d+)',str(row.workTitle),re.I)
            if match:
                key=(composer,int(match[1]),int(float(row.movementNumber)))
                assert key not in lookup
                lookup[key]=(corpus,str(row.piece),str(row.workNumber))
    rows=[];excluded=[]
    for _,r in table[table.composer.isin(['Beethoven','Mozart'])].iterrows():
        m=re.fullmatch(r'Piano_Sonatas_(\d+)-(\d+)',r.title)
        if not m:
            excluded.append(dict(folder=r.folder,title=r.title,reason='combined_movement_or_not_sonata'));continue
        key=(r.composer,int(m[1]),int(m[2]))
        if key not in lookup:
            excluded.append(dict(folder=r.folder,title=r.title,reason='no_exact_sonata_movement_metadata_match'));continue
        corpus,piece,opus=lookup[key];pid=corpus+'_'+piece
        if pid not in available:
            excluded.append(dict(folder=r.folder,title=r.title,reason='dcml_not_usable_phrase_start_labels'));continue
        audio=ASAP/r.audio_performance if r.audio_performance else None
        audio_kind='asap_cropped';offset=0.
        if audio is None or not audio.is_file():
            value=r.maestro_audio_performance
            audio=MAESTRO/value.replace('{maestro}/','') if value else None
            audio_kind='maestro_original';offset=float(r.start) if r.start!='' else 0.
        info=None;header_error=''
        if audio is not None and audio.is_file():
            try:info=sf.info(str(audio))
            except Exception as e:header_error=type(e).__name__+': '+str(e)
        a=ann.get(r.midi_performance,{})
        perf_beats=a.get('performance_beats',[]);score_beats=a.get('midi_score_beats',[])
        perf_beats=perf_beats if isinstance(perf_beats,list) else []
        score_beats=score_beats if isinstance(score_beats,list) else []
        downbeat_map=a.get('downbeats_score_map',[])
        downbeat_map=downbeat_map if isinstance(downbeat_map,list) else []
        rows.append(dict(composer=r.composer,sonata=int(m[1]),movement=int(m[2]),opus=opus,dcml_piece_id=pid,
            asap_folder=r.folder,asap_performance=r.midi_performance,audio_path=str(audio) if audio else '',audio_kind=audio_kind,
            source_audio_exists=bool(info),audio_seconds=info.duration if info else None,audio_samplerate=info.samplerate if info else None,
            audio_offset_seconds=offset,annotation_aligned_flag=a.get('score_and_performance_aligned',False) is True,
            performance_beats=len(perf_beats),score_beats=len(score_beats),equal_nonzero_beats=bool(perf_beats and len(perf_beats)==len(score_beats)),
            xml_exists=(ASAP/r.xml_score).is_file(),midi_exists=(ASAP/r.midi_score).is_file(),
            downbeat_score_map_count=len(downbeat_map),header_error=header_error,
            phrase_audio_coordinates_verified=False,training_authorized_by_this_audit=False))
    df=pd.DataFrame(rows);df.to_csv(OUT/'candidates.csv',index=False);pd.DataFrame(excluded).to_csv(OUT/'exclusions.csv',index=False)
    ready=df[df.source_audio_exists&df.annotation_aligned_flag&df.equal_nonzero_beats&df.xml_exists&df.midi_exists&(df.downbeat_score_map_count>0)]
    summary=dict(status='candidate_inventory_only',metadata_matched_performances=len(df),metadata_matched_movements=int(df.dcml_piece_id.nunique()),
        audio_header_and_asap_alignment_candidates=len(ready),audio_candidate_movements=int(ready.dcml_piece_id.nunique()),
        audio_candidate_sonatas=int(ready[['composer','sonata']].drop_duplicates().shape[0]),
        verified_phrase_audio_pairs=0,training_runs=0,seconds=time.monotonic()-began,
        next_gate='Verify DCML-to-ASAP note/measure identity, repeats/voltas/pickups and per-quarterbeat timestamps; metadata titles alone are not sufficient.')
    ready.to_csv(OUT/'audio_candidates.csv',index=False);write(OUT/'summary.json',summary);write(OUT/'source_hashes.json',sources)
    assert all(sha(p)==h for p,h in sources.items());print(summary,flush=True)


if __name__=='__main__':main()
