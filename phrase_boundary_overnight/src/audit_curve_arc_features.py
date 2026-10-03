"""Label-free, resumable input audit for two-sided curve shape."""
import time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from .curve_arc_features import transform
from .score_context_study import ROOT,sha,write

OUT=ROOT/'reports/curve_arc_feature_audit';ART=ROOT/'artifacts/curve_arc_feature_audit'

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True);(ART/'cache').mkdir(parents=True,exist_ok=True)
    sources=sorted((ROOT/'artifacts/slice_energy_study/cache').glob('*.npz'));assert len(sources)==43
    paths=sources+[Path(__file__),ROOT/'src/curve_arc_features.py',ROOT/'tests/test_curve_arc_features.py',OUT/'PROTOCOL.md']
    hashes={str(p):sha(p) for p in paths};rows=[]
    write(OUT/'STATE.json',dict(status='running'))
    for source in sources:
        assert time.monotonic()-start<300
        with np.load(source,allow_pickle=False) as z:x=z['curves'].copy()
        feat=transform(x);dest=ART/'cache'/source.name
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:np.testing.assert_array_equal(feat,z['arc'])
        else:np.savez_compressed(dest,arc=feat)
        hashes[str(dest)]=sha(dest);valid=feat[...,4]>0
        turning=(feat[...,0]*feat[...,1]<0)&valid
        rows.append(dict(piece_id=source.stem,performances=len(x),quarterbeats=x.shape[1],coverage=float(valid.mean()),turn_fraction_observed=float(turning.sum()/max(valid.sum(),1))))
        print('ARC INPUT',source.stem,flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    df=pd.DataFrame(rows);df.to_csv(OUT/'coverage.csv',index=False)
    write(OUT/'source_hashes.json',hashes)
    report=dict(status='complete',works=len(df),performances=int(df.performances.sum()),work_quarterbeats=int(df.quarterbeats.sum()),work_mean_coverage=float(df.coverage.mean()),labels_accessed=False,new_training_runs=0,new_f1=None,hashes_unchanged=True,seconds=time.monotonic()-start)
    write(OUT/'completion_audit.json',report);write(OUT/'STATE.json',dict(status='complete'));print(report,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'STATE.json',dict(status='failed',traceback=traceback.format_exc()));raise
