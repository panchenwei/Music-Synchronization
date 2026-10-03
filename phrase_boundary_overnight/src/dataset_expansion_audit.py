"""Pinned public DCML downloads and read-only audit of local Islamey slices.

Downloads data only, never remote code. Originals and older experiments are untouched.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from .acquire_schumann_probe import fetch
from .score_context_study import ROOT, read, write, sha

OUT = ROOT / 'reports/dataset_expansion_20260913'


def local_islamey():
    base = Path('C:/Users/pa1018/Desktop/learn/柴柴/VQVAE/model_data/model_data')
    rows, frames, hashes = [], [], {}
    for split in ('train', 'val', 'test'):
        path = base / f'pre_data_{split}_split.csv'
        df = pd.read_csv(path)
        df = df[df.Song.astype(str).eq('Islamey')].copy()
        df['source_split'] = split
        frames.append(df)
        hashes[str(path)] = sha(path)
        rows.append(dict(split=split, rows=len(df), performers=df.Performer.nunique(),
                         phrase_positions=df.Phrase_Index.nunique()))
    allrows = pd.concat(frames, ignore_index=True)
    lengths = [int(sum(float(t) > 0 for t in str(x).split(';'))) for x in allrows.padding_mask]
    groups = allrows.groupby(['Song', 'Performer']).source_split.nunique()
    result = dict(rows=len(allrows), works=allrows.Song.nunique(),
                  performers=allrows.Performer.nunique(),
                  phrase_positions=allrows.Phrase_Index.nunique(),
                  phrase_index_min=int(allrows.Phrase_Index.min()),
                  phrase_index_max=int(allrows.Phrase_Index.max()),
                  repeated_performer_groups_across_splits=int((groups > 1).sum()),
                  repeated_song_across_splits=True,
                  duplicate_performer_phrase_rows=int(allrows.duplicated(['Song', 'Performer', 'Phrase_Index']).sum()),
                  valid_sequence_lengths=sorted(set(lengths)), source_hashes=hashes,
                  interpretation='Already segmented normalized/padded tempo slices, not raw continuous recordings or new DCML start annotations. Same work crosses old split; these splits are not an unseen-work boundary test.',
                  historical_thesis='25 performances x 40 phrases; local counts are measured independently and may differ.',
                  source_coordinates_verified=False, approved_for_target_training=False)
    pd.DataFrame(rows).to_csv(OUT / 'islamey_splits.csv', index=False)
    write(OUT / 'islamey_audit.json', result)
    print('ISLAMEY', json.dumps(result, ensure_ascii=False), flush=True)


def acquire(corpus):
    repo = f'DCMLab/{corpus}'
    dest = ROOT / 'external_data/dcml_expansion_20260913' / corpus
    report = OUT / corpus
    dest.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    pin = report / 'acquisition_pin.json'
    if pin.exists():
        commit = read(pin)['commit']
    else:
        commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
        write(pin, dict(repository=repo, commit=commit,
                        selection='Every notes/measures/harmonies TSV and license/metadata. No model outputs used.'))
    tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))
    assert not tree.get('truncated')
    chosen = [e for e in tree['tree'] if e['type'] == 'blob' and
              (e['path'] in ('README.md', 'LICENSE', 'CITATION.cff', '.zenodo.json', 'metadata.tsv') or
               (e['path'].split('/')[0] in ('notes', 'measures', 'harmonies') and e['path'].endswith('.tsv')))]
    assert 0 < sum(e.get('size', 0) for e in chosen) < 150_000_000
    def one(e):
        target = (dest / e['path']).resolve()
        assert target.is_relative_to(dest.resolve())
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/{quote(e["path"])}'
        content = target.read_bytes() if target.exists() else fetch(url)
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        assert blob == e['sha'], e['path']
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return dict(path=e['path'], url=url, bytes=len(content), git_blob_sha1=blob,
                    sha256=hashlib.sha256(content).hexdigest())
    manifest = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(one, chosen):
            manifest.append(r)
            if len(manifest) % 40 == 0:
                print(corpus, 'ACQUIRED', len(manifest), '/', len(chosen), flush=True)
    write(report / 'download_manifest.json', dict(repository=repo, commit=commit, files=manifest))
    zmeta = read(dest / '.zenodo.json')
    assert zmeta['license'].upper() == 'CC-BY-NC-SA-4.0', zmeta.get('license')
    rows = []
    meta = pd.read_csv(dest / 'metadata.tsv', sep='\t').set_index('piece')
    for p in sorted((dest / 'notes').glob('*.notes.tsv')):
        pid = p.name.split('.')[0]
        harm_path = dest / 'harmonies' / f'{pid}.harmonies.tsv'
        h = pd.read_csv(harm_path, sep='\t') if harm_path.exists() else pd.DataFrame()
        phrase = h.phraseend.fillna('').astype(str) if 'phraseend' in h else pd.Series([], dtype=str)
        rows.append(dict(piece=pid, source_start_rows=int(phrase.str.contains('{', regex=False).sum()),
                         source_end_rows=int(phrase.str.contains('}', regex=False).sum()),
                         phrase_field='phraseend' in h, length_qb_unfolded=float(meta.loc[pid, 'length_qb_unfolded']),
                         workNumber=str(meta.loc[pid].get('workNumber', 'unknown'))))
    frame = pd.DataFrame(rows)
    frame.to_csv(report / 'schema_audit.csv', index=False)
    state = dict(status='acquired_schema_checked_not_trained', score_files=len(rows),
                 score_files_with_starts=int((frame.source_start_rows > 0).sum()),
                 source_start_rows=int(frame.source_start_rows.sum()),
                 source_end_rows=int(frame.source_end_rows.sum()), files=len(manifest),
                 bytes=sum(r['bytes'] for r in manifest), license=zmeta['license'], commit=commit,
                 target_score_or_audio_overlap='not yet audited; never use these as additional independent target tests',
                 new_training_runs=0)
    write(report / 'STATE.json', state)
    print(corpus, state, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    local_islamey()
    for corpus in ('mozart_piano_sonatas', 'beethoven_piano_sonatas'):
        acquire(corpus)


if __name__ == '__main__':
    main()
