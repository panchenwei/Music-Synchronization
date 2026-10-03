"""Prepare independent-source score-only inputs; no model fitting or prediction."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .score_local_coordinates import local_events,local_labels
from .score_tied_events import tied_piano_roll
from .slice_energy_study import DCML
from .data import discover_dcml_pieces,to_float

OUT=ROOT/'reports/external_score_probe';ART=ROOT/'artifacts/external_score_probe'


def note_fingerprint(notes):
    # Folded notation, no harmony columns; invariant to global transposition and voice ordering.
    events=sorted(set((int(r.mc),round(4*to_float(r.mc_onset),6),round(to_float(r.duration_qb),6),int(r.midi)) for r in notes.itertuples() if to_float(r.duration_qb)>0))
    if not events:raise ValueError('empty score')
    minimum=min(e[3] for e in events);first=events[0][0]
    canon=[(mc-first,o,d,p-minimum) for mc,o,d,p in events]
    return hashlib.sha256(json.dumps(canon,separators=(',',':')).encode()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);(ART/'cache').mkdir(parents=True,exist_ok=True)
    sources=[('schumann_kinderszenen',ROOT/'external_data/dcml_schumann_probe',ROOT/'reports/external_schumann_probe'),('grieg_lyric_pieces',ROOT/'external_data/dcml_grieg_probe',ROOT/'reports/external_grieg_probe')]
    hashes={};targetprints={}
    # Read only symbolic fingerprints, not held-out model outputs.
    for pid,p in discover_dcml_pieces(DCML).items():
        targetprints.setdefault(note_fingerprint(pd.read_csv(p.notes_path,sep='\t')),[]).append(pid)
        hashes[str(p.notes_path)]=sha(p.notes_path)
    rows=[];failures=[];seen={};overlap=[]
    for corpus,source,report in sources:
        manifest=read(report/'download_manifest.json')
        for f in manifest['files']:
            p=source/f['path'];assert sha(p)==f['sha256'];hashes[str(p)]=f['sha256']
        metadata=pd.read_csv(source/'metadata.tsv',sep='\t').set_index('piece')
        for pid in sorted(metadata.index):
            uid=f'{corpus}_{pid}'
            try:
                m=metadata.loc[pid];length=float(m.length_qb_unfolded);n=int(np.ceil(length-1e-9));assert 1<=n<=10000
                notes=pd.read_csv(source/'notes'/f'{pid}.notes.tsv',sep='\t');measures=pd.read_csv(source/'measures'/f'{pid}.measures.tsv',sep='\t');harmonies=pd.read_csv(source/'harmonies'/f'{pid}.harmonies.tsv',sep='\t')
                fp=note_fingerprint(notes)
                if fp in targetprints:overlap.append(dict(piece_id=uid,matches=','.join(targetprints[fp])));raise ValueError('target normalized note fingerprint overlap')
                if fp in seen:raise ValueError('duplicate external fingerprint: '+seen[fp])
                seen[fp]=uid
                events,ties=local_events(notes,measures,n);roll,tie_counts=tied_piano_roll(events,ties,n)
                labels,mask,mapping,provenance,mode,error=local_labels(harmonies,measures,n)
                positives=int((labels*mask).sum());valid=int(mask.sum())
                # Keep pieces with valid negative supervision even if all positive
                # positions were masked by the fixed half-beat ambiguity rule.
                status='usable' if valid>=2 else 'no_valid_supervision'
                target=ART/'cache'/f'{uid}.npz'
                payload=dict(piano_roll=roll,labels=labels,label_mask=mask)
                if target.exists():
                    with np.load(target,allow_pickle=False) as z:
                        for key,value in payload.items():np.testing.assert_array_equal(z[key],value)
                else:np.savez_compressed(target,**payload)
                hashes[str(target)]=sha(target)
                rows.append(dict(piece_id=uid,corpus=corpus,opus=str(m.get('workNumber','unknown')),beats=n,metadata_length_qb=length,notes=len(events),valid_beats=valid,valid_positives=positives,zero_retained_positives=positives==0,ambiguous_starts=sum(r['status']=='ambiguous_tie' for r in mapping),mode=mode,length_error=error,**tie_counts,note_fingerprint=fp,status=status))
                write(OUT/'provenance'/f'{uid}.json',dict(mapping=mapping,source_start_provenance=provenance))
            except Exception as e:failures.append(dict(piece_id=uid,error=repr(e)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'piece_audit.csv',index=False);pd.DataFrame(failures).to_csv(OUT/'failures.csv',index=False);write(OUT/'overlap.json',dict(exact_normalized_note_overlaps=overlap,method='Folded MC/onset/duration/pitch fingerprint, invariant to global pitch transposition; no fuzzy arrangement guarantee.'))
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'source_hashes.json',hashes)
    result=dict(status='prepared' if not failures else 'partial_requires_review',source_works=79,prepared_works=len(df),usable_works=int((df.status=='usable').sum()),failures=len(failures),usable_valid_positives=int(df[df.status=='usable'].valid_positives.sum()),beats=int(df.beats.sum()),exact_target_overlaps=len(overlap),model_training_runs=0,predictions=0,split_assignment='not assigned; freeze by opus before training',note='Score-only prepared inputs; not additional Mazurka tests, acoustic performances, or human perception labels.')
    write(OUT/'STATE.json',result);print(result);print(pd.DataFrame(failures).to_string(index=False))


if __name__=='__main__':main()
