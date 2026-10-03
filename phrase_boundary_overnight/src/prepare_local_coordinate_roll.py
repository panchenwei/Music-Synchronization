"""Prepare fully local-coordinate, tie-aware score roll without model evaluation."""
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .slice_energy_study import DCML
from .data import discover_dcml_pieces
from .score_local_coordinates import local_events
from .score_tied_events import tied_piano_roll


def main():
    out=ROOT/'reports/local_coordinate_roll';art=ROOT/'artifacts/local_coordinate_roll/cache';art.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    pieces=discover_dcml_pieces(DCML);rows=[];hashes={}
    for path in sorted((ROOT/'artifacts/score_roll_study/cache').glob('*.npy')):
        pid=path.stem;piece=pieces[pid];old=np.load(path,allow_pickle=False);n=len(old)
        notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t');events,ties=local_events(notes,measures,n)
        roll,counts=tied_piano_roll(events,ties,n)
        assert roll.shape==old.shape and np.isfinite(roll).all() and roll.min()>=0 and roll.max()<=1
        assert (roll[:,0]>=old[:,0]-1e-5).all()
        dest=art/f'{pid}.npy'
        if dest.exists():np.testing.assert_array_equal(np.load(dest,allow_pickle=False),roll)
        else:np.save(dest,roll)
        for p in (path,piece.notes_path,piece.measures_path,dest):hashes[str(p)]=sha(p)
        changed=np.max(abs(roll-old),axis=(1,2,3))>1e-5
        rows.append(dict(piece_id=pid,beats=n,events=len(events),**counts,old_occupancy_sum=float(old[:,0].sum()),new_occupancy_sum=float(roll[:,0].sum()),old_onset_cells=int(old[:,1].sum()),new_onset_cells=int(roll[:,1].sum()),changed_beats=int(changed.sum())))
    for p in (ROOT/'src/score_local_coordinates.py',ROOT/'src/score_tied_events.py',ROOT/'src/score_piano_roll.py',ROOT/'src/prepare_local_coordinate_roll.py'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());df=pd.DataFrame(rows);assert len(df)==43;df.to_csv(out/'piece_audit.csv',index=False);write(out/'source_hashes.json',hashes)
    result=dict(status='prepared_no_training',works=43,events=int(df.events.sum()),connected=int(df.connected.sum()),orphan=int(df.orphan.sum()),ambiguous=int(df.ambiguous.sum()),changed_beats=int(df.changed_beats.sum()),labels_modified=False,model_predictions=0,source_and_old_cache_unchanged=True)
    write(out/'STATE.json',result);print(result)


if __name__=='__main__':main()
