"""No splits selected: prospective independence groups for the paired corpus."""
from pathlib import Path
import hashlib,time
import pandas as pd
from .score_context_study import ROOT,read,write,sha


def main():
    started=time.monotonic();out=ROOT/'reports/external_audio_group_audit';out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'reports/external_audio_masks_v2/performance_audit.csv';manifest=ROOT/'reports/external_audio_masks_v2/grid_manifest.csv'
    audit=pd.read_csv(path);audit=audit[(audit.positive_grid_positions>0)&(audit.negative_grid_positions>0)].copy()
    names=pd.read_csv(manifest).set_index('performance');audit['audio_path']=[names.loc[p,'audio_path'] for p in audit.performance]
    audit['sonata_key']=[f'{c}:{int(s)}' for c,s in zip(audit.composer,audit.sonata)]
    parent={k:k for k in audit.sonata_key.unique()}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    # Same source recording must not cross future folds even when cropped into movements.
    audio_hash={};file_rows=[]
    for value in sorted(audit.audio_path.unique()):
        assert time.monotonic()-started<300
        p=Path(value);h=sha(p);audio_hash[value]=h
        file_rows.append(dict(audio_path=value,bytes=p.stat().st_size,sha256=h))
    audit['audio_sha256']=[audio_hash[p] for p in audit.audio_path]
    for _,g in audit.groupby('audio_sha256'):
        keys=list(g.sonata_key.unique())
        for key in keys[1:]:union(keys[0],key)
    audit['independence_group']=[find(k) for k in audit.sonata_key]
    audit.to_csv(out/'prospective_group_manifest.csv',index=False);pd.DataFrame(file_rows).to_csv(out/'audio_source_hashes.csv',index=False)
    summary=audit.groupby('independence_group').agg(performances=('performance','size'),movements=('piece_id','nunique'),sonatas=('sonata_key','nunique'),
        positive_grid_positions=('positive_grid_positions','sum'),known_grid_positions=('known_grid_positions','sum'))
    summary.to_csv(out/'group_summary.csv')
    hashes=read(ROOT/'reports/external_audio_masks_v2/source_hashes.json')
    for p in (Path(__file__),path,manifest):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());write(out/'source_hashes.json',hashes)
    result=dict(status='prospective_groups_audited_no_split_chosen',performances=len(audit),movements=int(audit.piece_id.nunique()),
        sonatas=int(audit.sonata_key.nunique()),independence_groups=int(audit.independence_group.nunique()),
        distinct_audio_paths=len(audio_hash),distinct_audio_hashes=int(audit.audio_sha256.nunique()),
        groups_merging_multiple_sonatas=int((summary.sonatas>1).sum()),training_runs=0,test_set_chosen=False,
        source_hashes_unchanged=True,seconds=time.monotonic()-started,
        caution='These are potential grouped evaluation units, not new independent labels. No train/test random assignment or score observed.')
    write(out/'completion_audit.json',result);print(result,flush=True);print(summary.to_string(),flush=True)


if __name__=='__main__':main()
