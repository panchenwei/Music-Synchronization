"""Decompose a frozen decoder's rejections, without evaluating alternatives."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from . import external_adaptive_peak_probe as trial
from .score_context_study import ROOT,read,write,sha


def main():
    hashes=dict(read(trial.OUT/'contract.json')['hashes']);hashes.update(read(trial.OUT/'artifact_hashes.json'));hashes[str(Path(__file__))]=sha(Path(__file__))
    assert read(trial.OUT/'completion_audit.json')['status']=='complete';rows=[]
    for p in sorted(trial.ART.glob('*.npz')):
        kind,fold,record=p.stem.split('_',2)
        if kind=='S':continue
        source={'A':'M','D':'MD','C':'C'}[kind]
        with np.load(trial.SOURCE/f'{source}_{fold}_predictions.npz',allow_pickle=False) as z:x=z[record].copy()
        with np.load(p,allow_pickle=False) as z:cuts=z['cuts'];smooth=z['smooth'];threshold=z['local_threshold']
        raw_peaks=(x[1:-1]>x[:-2])&(x[1:-1]>x[2:]);peaks=np.flatnonzero((smooth[1:-1]>smooth[:-2])&(smooth[1:-1]>smooth[2:]))+1
        localmedian=median_filter(smooth,size=16,mode='reflect');prominence=(smooth[peaks]-localmedian[peaks])/max(float(x.mean()),1e-12)
        passed=peaks[prominence>.05];np.testing.assert_array_equal(passed,cuts)
        rows.append(dict(kind=kind,fold=fold,record=record,raw_local_peaks=int(raw_peaks.sum()),smooth_local_peaks=len(peaks),
            above_local_median=int((prominence>0).sum()),above_final_threshold=len(cuts),offset=float(.05*x.mean()),
            median_relative_prominence=float(np.median(prominence)) if len(peaks) else 0.,p95_relative_prominence=float(np.quantile(prominence,.95)) if len(peaks) else 0.))
    df=pd.DataFrame(rows);df.to_csv(trial.OUT/'mechanism_audit.csv',index=False)
    counts=df.groupby('kind')[['raw_local_peaks','smooth_local_peaks','above_local_median','above_final_threshold']].sum();counts.to_csv(trial.OUT/'mechanism_counts.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(trial.OUT/'mechanism_completion.json',dict(status='complete',records=len(df),hashes_unchanged=True,labels_read=False,training_runs=0,alternate_metrics_computed=False))
    print(counts.to_string())


if __name__=='__main__':main()
