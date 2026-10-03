"""Official 66-work Grieg corpus acquisition; no remote code execution."""
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import pandas as pd
from .acquire_schumann_probe import fetch
from .score_context_study import ROOT,read,write

REPO='DCMLab/grieg_lyric_pieces';DEST=ROOT/'external_data/dcml_grieg_probe';OUT=ROOT/'reports/external_grieg_probe'


def main():
    DEST.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);pin=OUT/'acquisition_pin.json'
    if pin.exists():commit=read(pin)['commit']
    else:
        commit=json.loads(fetch(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
        write(pin,dict(repository=REPO,commit=commit,purpose='external score transfer feasibility; no training or inference yet',selection='all 66 Lyric Pieces, all notes/measures/harmonies, no score-dependent selection'))
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1'));assert not tree.get('truncated')
    chosen=[e for e in tree['tree'] if e['type']=='blob' and (e['path'] in ('README.md','LICENSE','CITATION.cff','.zenodo.json','metadata.tsv') or (e['path'].split('/')[0] in ('notes','measures','harmonies') and e['path'].endswith('.tsv')))]
    assert sum(e.get('size',0) for e in chosen)<10_000_000
    def one(e):
        target=(DEST/e['path']).resolve();assert target.is_relative_to(DEST.resolve())
        url=f'https://raw.githubusercontent.com/{REPO}/{commit}/{quote(e["path"])}'
        content=target.read_bytes() if target.exists() else fetch(url)
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest();assert blob==e['sha']
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        return dict(path=e['path'],url=url,bytes=len(content),git_blob_sha1=blob,sha256=hashlib.sha256(content).hexdigest())
    manifest=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(one,chosen):
            manifest.append(row)
            if len(manifest)%25==0:print('ACQUIRED',len(manifest),'/',len(chosen),flush=True)
    write(OUT/'download_manifest.json',dict(repository=REPO,commit=commit,files=manifest));rows=[]
    for p in sorted((DEST/'notes').glob('*.notes.tsv')):
        pid=p.name.split('.')[0];notes=pd.read_csv(p,sep='\t');measures=pd.read_csv(DEST/'measures'/f'{pid}.measures.tsv',sep='\t');harm=pd.read_csv(DEST/'harmonies'/f'{pid}.harmonies.tsv',sep='\t')
        assert {'midi','staff','voice','duration_qb','quarterbeats','mc','tied'}<=set(notes)
        assert {'mc','quarterbeats'}<=set(measures)
        phrase=harm.phraseend.fillna('').astype(str) if 'phraseend' in harm else pd.Series(['']*len(harm))
        rows.append(dict(piece_id='grieg_lyric_pieces_'+pid,notes=len(notes),measures=len(measures),source_start_rows=int(phrase.str.contains('{',regex=False).sum()),source_end_rows=int(phrase.str.contains('}',regex=False).sum())))
    assert len(rows)==66;pd.DataFrame(rows).to_csv(OUT/'schema_audit.csv',index=False)
    state=dict(status='acquired_and_schema_checked',works=66,source_start_rows=sum(r['source_start_rows'] for r in rows),files=len(manifest),bytes=sum(r['bytes'] for r in manifest),license='CC-BY-NC-SA-4.0 per official README/LICENSE',target_training_runs=0,model_predictions=0,labels_not_yet_mapped_or_used_for_training=True,commit=commit)
    write(OUT/'STATE.json',state);print(state)


if __name__=='__main__':main()
