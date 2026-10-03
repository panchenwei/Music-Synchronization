"""Download pinned public artifacts; never execute remote model code here."""
import argparse,json,time,urllib.request,hashlib
from pathlib import Path
from .score_context_study import ROOT,write,read,sha

REV='12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
REPO='m-a-p/MERT-v1-95M'
DEST=ROOT/'external_runtime/mert_reviewed';OUT=ROOT/'reports/mert_vendor'
SMALL=('README.md','config.json','preprocessor_config.json','configuration_MERT.py','modeling_MERT.py')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',action='store_true');args=ap.parse_args();began=time.monotonic()
    DEST.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/tree/{REV}',timeout=30) as r:listing=json.load(r)
    info={r['path']:r for r in listing};rows=[];total=0
    for name in (*SMALL,*(('pytorch_model.bin',) if args.weights else ())):
        entry=info[name];size=entry['size'];assert 0<size<420_000_000;path=DEST/name;expected=entry.get('lfs',{}).get('oid')
        url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.part');count=0
            with urllib.request.urlopen(url,timeout=60) as response,partial.open('wb') as target:
                while chunk:=response.read(1<<20):
                    count+=len(chunk);assert count<=size and time.monotonic()-began<900
                    target.write(chunk)
            assert count==size
            if expected:assert sha(partial)==expected
            partial.replace(path)
        assert path.stat().st_size==size
        if expected:assert sha(path)==expected
        total+=size;rows.append(dict(file=name,url=url,size=size,sha256=sha(path),upstream_lfs_sha256=expected))
        print('DOWNLOADED',name,size,flush=True)
    result=dict(status='downloaded_not_executed',repo=REPO,revision=REV,files=rows,bytes=total,license='CC-BY-NC-4.0 per pinned README',
        code_review_required=True,weights_downloaded=args.weights,seconds=time.monotonic()-began)
    write(OUT/('weights_manifest.json' if args.weights else 'source_manifest.json'),result);print(result,flush=True)


if __name__=='__main__':main()
