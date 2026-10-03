"""Read-only comparison of folded qstamp subtraction with measure-local offsets."""
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .slice_energy_study import DCML
from .data import discover_dcml_pieces,choose_measure_path,to_float


def main():
    out=ROOT/'reports/score_coordinate_audit';out.mkdir(parents=True,exist_ok=True)
    pieces=discover_dcml_pieces(DCML);rows=[];examples=[];hashes={}
    for cache in sorted((ROOT/'artifacts/score_roll_study/cache').glob('*.npy')):
        pid=cache.stem;piece=pieces[pid];n=len(np.load(cache,allow_pickle=False));measures=pd.read_csv(piece.measures_path,sep='\t');path,mode,error=choose_measure_path(measures,n);assert path
        folded={int(r.mc):to_float(r.quarterbeats) for r in measures.itertuples()}
        for kind,p in (('notes',piece.notes_path),('harmonies',piece.harmony_path)):
            hashes[str(p)]=sha(p);frame=pd.read_csv(p,sep='\t');assert 'mc_onset' in frame
            missing=0;recovered=0;shifted=0;total=0;starts_missing=0;maxdiff=0.
            for occurrence in path:
                mc=int(occurrence['mc'])
                for idx,r in frame[frame.mc==mc].iterrows():
                    total+=1
                    legacy=occurrence['start_qb']+to_float(r.quarterbeats)-folded[mc]
                    local=occurrence['start_qb']+4*to_float(r.mc_onset)
                    is_start=kind=='harmonies' and '{' in str(r.get('phraseend',''))
                    if not np.isfinite(legacy):
                        missing+=1
                        if np.isfinite(local) and 0<=local<n:
                            recovered+=1;starts_missing+=int(is_start)
                            examples.append(dict(piece_id=pid,kind=kind,mc=mc,visit=occurrence['visit'],source_row=idx,measure_local_qb=local,legacy_qb=None,start_marker=is_start,status='recoverable_missing_coordinate'))
                    elif np.isfinite(local):
                        diff=abs(local-legacy);maxdiff=max(maxdiff,diff)
                        if diff>1e-7:
                            shifted+=1;examples.append(dict(piece_id=pid,kind=kind,mc=mc,visit=occurrence['visit'],source_row=idx,measure_local_qb=local,legacy_qb=legacy,start_marker=is_start,status='finite_coordinate_disagreement'))
            rows.append(dict(piece_id=pid,kind=kind,mode=mode,length_error=error,unfolded_rows=total,legacy_nonfinite=missing,local_recoverable_rows=recovered,finite_disagreements=shifted,max_finite_difference=maxdiff,recoverable_start_marker_rows=starts_missing))
        hashes[str(piece.measures_path)]=sha(piece.measures_path)
    df=pd.DataFrame(rows);assert len(df)==86;df.to_csv(out/'piece_audit.csv',index=False);pd.DataFrame(examples).to_csv(out/'coordinate_examples.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    write(out/'summary.json',dict(status='complete',source_unchanged=True,training_runs=0,labels_modified=False,predictions_accessed=False,by_source=df.groupby('kind')[['unfolded_rows','legacy_nonfinite','local_recoverable_rows','finite_disagreements','recoverable_start_marker_rows']].sum().to_dict('index'),note='Source label semantics inspected only for mapping audit. A recovered start is not automatically an independent/new ground-truth boundary; freeze a new label version before any future evaluation change.'))
    print(read(out/'summary.json'))


if __name__=='__main__':main()
