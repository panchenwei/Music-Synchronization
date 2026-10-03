"""Stage source-grounded label repair separately; never overwrite active labels."""
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .slice_energy_study import DCML
from .data import discover_dcml_pieces
from .score_local_coordinates import local_labels


def main():
    out=ROOT/'reports/score_coordinate_audit';dest=ROOT/'artifacts/coordinate_repair_preview';dest.mkdir(parents=True,exist_ok=True)
    pieces=discover_dcml_pieces(DCML);rows=[];changes=[];hashes={}
    for p in sorted((ROOT/'artifacts/score_roll_study/cache').glob('*.npy')):
        pid=p.stem;piece=pieces[pid];old=ROOT/'artifacts/slice_energy_study/cache'/f'{pid}.npz';hashes[str(old)]=sha(old)
        with np.load(old,allow_pickle=False) as z:y0=z['start_labels'].copy();m0=z['start_mask'].copy()
        harmonies=pd.read_csv(piece.harmony_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t')
        y,m,mapping,provenance,mode,error=local_labels(harmonies,measures,len(y0))
        np.savez_compressed(dest/f'{pid}.npz',labels=y,label_mask=m)
        delta=np.flatnonzero((y!=y0)|(m!=m0))
        for b in delta:changes.append(dict(piece_id=pid,beat=int(b),old_label=float(y0[b]),new_label=float(y[b]),old_mask=float(m0[b]),new_mask=float(m[b])))
        rows.append(dict(piece_id=pid,old_valid_positives=int((y0*m0).sum()),new_valid_positives=int((y*m).sum()),changed_labels=int((y!=y0).sum()),changed_masks=int((m!=m0).sum()),mode=mode,length_error=error))
    assert all(sha(p)==h for p,h in hashes.items());df=pd.DataFrame(rows);df.to_csv(out/'label_repair_preview.csv',index=False);pd.DataFrame(changes).to_csv(out/'label_repair_changes.csv',index=False)
    summary=dict(status='preview_only_not_active',works=len(df),original_labels_unchanged=True,training_runs=0,predictions_accessed=False,totals=df[['old_valid_positives','new_valid_positives','changed_labels','changed_masks']].sum().to_dict(),warning='Need a separately frozen re-baseline before using repaired targets. Never attribute a change in evaluation labels to model improvement.')
    write(out/'label_repair_preview.json',summary);print(summary);print(pd.DataFrame(changes).to_string(index=False))


if __name__=='__main__':main()
