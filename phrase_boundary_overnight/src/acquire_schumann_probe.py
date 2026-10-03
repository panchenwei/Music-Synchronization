"""Acquire a small official non-Chopin score corpus, without running remote code."""
import hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import quote
import pandas as pd
from .score_context_study import read,write,sha

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'external_data/dcml_schumann_probe'
OUT=ROOT/'reports/external_schumann_probe'
REPO='DCMLab/schumann_kinderszenen'


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'MusicResearchDataAudit/1.0'}),timeout=25) as r:return r.read()
        except Exception:
            if attempt==2:raise
            time.sleep(1)


def main():
    DEST.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    pin=OUT/'acquisition_pin.json'
    if pin.exists():commit=read(pin)['commit']
    else:
        commit=json.loads(fetch(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
        write(pin,dict(repository=REPO,commit=commit,purpose='schema and source-label audit only; no model training or inference',selection='all 13 Kinderszenen works, all notes/measures/harmonies TSVs; no selection using model scores'))
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1'))
    assert not tree.get('truncated')
    chosen=[e for e in tree['tree'] if e['type']=='blob' and (e['path'] in ('README.md','LICENSE','CITATION.cff','.zenodo.json','metadata.tsv') or (e['path'].split('/')[0] in ('notes','measures','harmonies') and e['path'].endswith('.tsv')))]
    assert sum(e.get('size',0) for e in chosen)<3_000_000
    def one(e):
        relative=Path(e['path']);target=(DEST/relative).resolve();assert target.is_relative_to(DEST.resolve())
        url=f'https://raw.githubusercontent.com/{REPO}/{commit}/{quote(e["path"])}'
        content=target.read_bytes() if target.exists() else fetch(url)
        gitsha=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest();assert gitsha==e['sha'],e['path']
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        return dict(path=e['path'],url=url,bytes=len(content),git_blob_sha1=gitsha,sha256=hashlib.sha256(content).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(one,chosen))
    write(OUT/'download_manifest.json',dict(repository=REPO,commit=commit,files=manifest))
    rows=[]
    for path in sorted((DEST/'notes').glob('*.notes.tsv')):
        pid=path.name.split('.')[0];notes=pd.read_csv(path,sep='\t');measures=pd.read_csv(DEST/'measures'/f'{pid}.measures.tsv',sep='\t');harm=pd.read_csv(DEST/'harmonies'/f'{pid}.harmonies.tsv',sep='\t')
        assert {'midi','staff','voice','duration_qb','quarterbeats','mc'}<=set(notes)
        assert {'mc','quarterbeats'}<=set(measures)
        phrase=harm.phraseend.fillna('').astype(str) if 'phraseend' in harm else pd.Series(['']*len(harm))
        rows.append(dict(piece_id=f'schumann_kinderszenen_{pid}',notes=len(notes),measures=len(measures),harmony_rows=len(harm),source_start_rows=int(phrase.str.contains('{',regex=False).sum()),source_end_rows=int(phrase.str.contains('}',regex=False).sum()),source_overlap_chopin=False))
    assert len(rows)==13
    pd.DataFrame(rows).to_csv(OUT/'schema_audit.csv',index=False)
    write(OUT/'STATE.json',dict(status='acquired_and_schema_checked',works=len(rows),source_start_rows=sum(r['source_start_rows'] for r in rows),files=len(manifest),bytes=sum(r['bytes'] for r in manifest),license='CC-BY-NC-SA-4.0 per official README/LICENSE',target_training_runs=0,model_predictions=0,labels_not_yet_mapped_or_used_for_training=True,commit=commit))
    print(pd.DataFrame(rows).to_string(index=False));print(read(OUT/'STATE.json'))


if __name__=='__main__':main()
