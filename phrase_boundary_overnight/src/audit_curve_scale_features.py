"""Input-only evidence for a non-acoustic time/scale representation."""
import json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from .curve_scale_features import transform,SCALES
from .phase6_multiscale import segment_novelty
from .score_context_study import ROOT,sha,write

OUT=ROOT/'reports/curve_scale_feature_audit';ART=ROOT/'artifacts/curve_scale_feature_audit'


def main():
    began=time.monotonic();OUT.mkdir(exist_ok=True);(ART/'cache').mkdir(parents=True,exist_ok=True)
    paths=sorted((ROOT/'artifacts/slice_energy_study/cache').glob('*.npz'));assert len(paths)==43
    hashes={str(p):sha(p) for p in paths}
    for p in (Path(__file__),ROOT/'src/curve_scale_features.py',ROOT/'src/phase6_multiscale.py',ROOT/'tests/test_curve_scale_features.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    write(OUT/'source_hashes.json',hashes);write(OUT/'STATE.json',dict(status='running'));rows=[]
    for source in paths:
        # NPZ is lazy: no labels, target masks or test predictions are accessed.
        with np.load(source,allow_pickle=False) as z:curves=z['curves'].copy()
        assert curves.shape[-1]==9 and np.isfinite(curves).all()
        feat=transform(curves);dest=ART/'cache'/f'{source.stem}.npz'
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:np.testing.assert_array_equal(z['time_scale'],feat)
        else:np.savez_compressed(dest,time_scale=feat)
        hashes[str(dest)]=sha(dest)
        n=curves.shape[1];maxerr=0.
        for row,new in zip(curves,feat):
            valid_t=np.zeros(n,bool);valid_t[:-1]=(row[:-1,7]>.5)&(row[1:,7]>.5)
            for channel,index,valid in ((0,0,valid_t),(1,3,row[:,8]>.5)):
                for si,s in enumerate(SCALES):
                    old,_=segment_novelty(row[:,index],valid,s)
                    if n>=2*s:
                        error=float(abs(old[s:n-s+1]-abs(new[s:n-s+1,channel,si])).max())
                        maxerr=max(maxerr,error);assert error<1e-5
        values=feat[:,:,:2];quality=feat[:,:,2:]
        rows.append(dict(piece_id=source.stem,performances=len(curves),quarterbeats=n,available_fraction=float((quality>0).mean()),full_observation_fraction=float((quality==1).mean()),positive_fraction=float((values>1e-7).mean()),negative_fraction=float((values< -1e-7).mean()),old_absolute_max_error=maxerr,bytes=feat.nbytes))
        print('INPUT AUDIT',source.stem,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'coverage.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',works=len(frame),performances=int(frame.performances.sum()),work_quarterbeats=int(frame.quarterbeats.sum()),old_absolute_max_error=float(frame.old_absolute_max_error.max()),source_and_output_hashes_unchanged=True,labels_accessed=False,new_training_runs=0,new_f1=None,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete'));print(frame.describe().to_string(),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'STATE.json',dict(status='failed',traceback=traceback.format_exc()));raise
