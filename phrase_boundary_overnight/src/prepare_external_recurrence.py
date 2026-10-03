"""Same ordered recurrence representation on independent DCML composers.

No audio is invented; phrase labels are never passed to feature extraction.
"""
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .score_context_study import ROOT, read, write, sha
from .data import discover_dcml_pieces
from .slice_energy_study import DCML
from .score_local_coordinates import local_events, local_labels
from .note_relation_graph import merged_events
from .motif_recurrence_features import features
from .prepare_external_score_probe import note_fingerprint

OUT=ROOT/'reports/external_recurrence_data'; ART=ROOT/'artifacts/external_recurrence_data/cache'
SOURCES=(
    ('grieg_lyric_pieces',ROOT/'external_data/dcml_grieg_probe',ROOT/'reports/external_grieg_probe','train'),
    ('beethoven_piano_sonatas',ROOT/'external_data/dcml_expansion_20260913/beethoven_piano_sonatas',ROOT/'reports/dataset_expansion_20260913/beethoven_piano_sonatas','train'),
    ('mozart_piano_sonatas',ROOT/'external_data/dcml_expansion_20260913/mozart_piano_sonatas',ROOT/'reports/dataset_expansion_20260913/mozart_piano_sonatas','validation'),
)


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    hashes={};target={};seen={};rows=[];failures=[]
    for pid,p in discover_dcml_pieces(DCML).items():
        target[note_fingerprint(pd.read_csv(p.notes_path,sep='\t'))]=pid
        hashes[str(p.notes_path)]=sha(p.notes_path)
    for corpus,source,report,split in SOURCES:
        manifest=read(report/'download_manifest.json')
        for entry in manifest['files']:
            p=source/entry['path'];assert sha(p)==entry['sha256'];hashes[str(p)]=entry['sha256']
        metadata=pd.read_csv(source/'metadata.tsv',sep='\t').set_index('piece')
        for pid in sorted(metadata.index):
            assert time.monotonic()-start<1800
            uid=f'{corpus}_{pid}';m=metadata.loc[pid]
            hpath=source/'harmonies'/f'{pid}.harmonies.tsv'
            row=dict(piece_id=uid,corpus=corpus,split=split,source_work=str(m.get('workNumber','unknown')))
            if not hpath.exists():
                rows.append(dict(**row,status='excluded_missing_harmonies'));continue
            h=pd.read_csv(hpath,sep='\t')
            phrase=h.phraseend.fillna('').astype(str) if 'phraseend' in h else pd.Series([],dtype=str)
            if not phrase.str.contains('{',regex=False).any():
                rows.append(dict(**row,status='excluded_no_start_annotations'));continue
            try:
                n=int(np.ceil(float(m.length_qb_unfolded)-1e-9));assert 1<=n<=6000
                notes=pd.read_csv(source/'notes'/f'{pid}.notes.tsv',sep='\t')
                measures=pd.read_csv(source/'measures'/f'{pid}.measures.tsv',sep='\t')
                fp=note_fingerprint(notes)
                assert fp not in target, 'Exact target score overlap'
                assert fp not in seen, 'Exact external duplicate: '+str(seen.get(fp))
                labels,mask,mapping,prov,mode,err=local_labels(h,measures,n)
                if mask.sum()<2 or (labels*mask).sum()<1:
                    rows.append(dict(**row,status='excluded_no_retained_starts',beats=n));continue
                events,ties=local_events(notes,measures,n);merged,count=merged_events(events,ties)
                merged=np.asarray([e for e in merged if e[0]<n and e[0]+e[1]>0],float)
                path=ART/f'{uid}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        for k,v in [('labels',labels),('label_mask',mask),('note_events',merged)]:np.testing.assert_array_equal(z[k],v)
                        feat=z['recurrence'].copy()
                else:
                    feat=features(merged,n,True)
                    np.savez_compressed(path,recurrence=feat,labels=labels,label_mask=mask,note_events=merged)
                assert feat.shape==(n,24) and np.isfinite(feat).all() and feat.min()>=0 and feat.max()<=1
                seen[fp]=uid;hashes[str(path)]=sha(path)
                rows.append(dict(**row,status='usable',beats=n,valid_positions=int(mask.sum()),valid_starts=int((labels*mask).sum()),
                                 source_start_rows=int(phrase.str.contains('{',regex=False).sum()),mapped_starts=len(mapping),
                                 ambiguous_starts=sum(r['status']=='ambiguous_tie' for r in mapping),
                                 note_events=len(merged),merged_ties=count,note_fingerprint=fp,mode=mode,length_error=err))
                write(OUT/'provenance'/f'{uid}.json',dict(mapping=mapping,source_start_provenance=prov))
            except Exception as e:
                failures.append(dict(piece_id=uid,error=repr(e)));rows.append(dict(**row,status='failed_mapping'))
            pd.DataFrame(rows).to_csv(OUT/'piece_audit.csv',index=False)
            if len(rows)%10==0:print('PREPARED',len(rows),uid,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'piece_audit.csv',index=False)
    pd.DataFrame(failures,columns=['piece_id','error']).to_csv(OUT/'failures.csv',index=False)
    for p in (Path(__file__),ROOT/'src/score_local_coordinates.py',ROOT/'src/note_relation_graph.py',
              ROOT/'src/resolve_tie_audit.py',ROOT/'src/motif_recurrence_features.py',ROOT/'src/prepare_external_score_probe.py',
              ROOT/'src/data.py',ROOT/'src/slice_energy_features.py'):
        hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    good=frame[frame.status=='usable']
    summary=good.groupby(['corpus','split'])[['beats','valid_positions','valid_starts']].sum()
    summary['score_files']=good.groupby(['corpus','split']).size();summary.to_csv(OUT/'summary.csv')
    write(OUT/'STATE.json',dict(status='prepared' if not failures else 'partial_requires_review',failures=failures,
        usable_score_files=len(good),valid_starts=int(good.valid_starts.sum()),target_exact_score_overlaps=0,
        external_exact_duplicates=0,near_duplicate_limit='Exact normalized notation fingerprint only; not an arrangement plagiarism detector.',
        split='Whole composers: Grieg and Beethoven train; Mozart validation. No Chopin in external preparation.',
        feature_dim=24,features='Identical label-free ordered recurrence of tie-merged note events; no audio or tempo imputation.',
        seconds=time.monotonic()-start,new_training_runs=0))
    print(summary.to_string(),flush=True);print(read(OUT/'STATE.json'),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
