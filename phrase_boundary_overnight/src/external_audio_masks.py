"""Conservative external reference masks; no audio features or model training."""
from pathlib import Path
import hashlib,time
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .slice_energy_features import start_labels
from .external_audio_candidates import ASAP
from .external_audio_projection import midi_timing,seconds_to_quarters

OUT=ROOT/'reports/external_audio_masks';ART=ROOT/'artifacts/external_audio_masks'


def snap_coordinate(value):
    value=float(value);near=round(value*2)/2
    return near if abs(value-near)<1e-5 else value


def continuous_runs(intervals):
    result=[]
    for a,b in sorted((snap_coordinate(a),snap_coordinate(b)) for a,b in intervals):
        assert a<b
        if result and a<result[-1][1]-1e-4:raise ValueError('overlapping unfolded intervals')
        if result and abs(a-result[-1][1])<1e-4:result[-1]=(result[-1][0],b)
        else:result.append((a,b))
    return result


def build_masks(intervals,targets,n):
    runs=continuous_runs(intervals);grid=np.arange(n)
    labels=np.zeros(n,np.float32);mask=np.zeros(n,np.float32)
    run_ids=np.full(n,-1,np.int32);coverage=np.zeros(n,bool);records=[]
    targets=sorted(set(snap_coordinate(t) for t in targets))
    for run,(a,b) in enumerate(runs):
        inside=(grid>=a)&(grid<b);coverage|=inside;run_ids[inside]=run
        local=[t for t in targets if a<=t<b]
        y,m,rows=start_labels(local,n)
        safe=(grid-.5>=a-1e-5)&(grid+.5<=b+1e-5)
        m*=safe.astype(np.float32)
        assert not ((mask>0)&(m>0)).any()
        labels=np.maximum(labels,y*m);mask=np.maximum(mask,m)
        for row in rows:
            idx=row['beat_index']
            records.append(dict(run=run,run_start=a,run_stop=b,**row,
                effective_positive=bool(idx is not None and y[idx]>0 and m[idx]>0)))
    assert ((labels==0)|(labels==1)).all() and (labels<=mask).all()
    assert not ((mask>0)&~coverage).any()
    return labels,mask,run_ids,coverage,records,runs


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    previous=ROOT/'reports/external_audio_projection';assert read(previous/'completion_audit.json')['status']=='reference_projection_audited_not_training_ready'
    paths=[previous/'verified_bar_intervals.csv',previous/'projected_start_references.csv',
        ROOT/'reports/external_audio_candidates/audio_candidates.csv',ASAP/'asap_annotations.json',
        Path(__file__),ROOT/'src/slice_energy_features.py',ROOT/'src/data.py',OUT/'PROTOCOL.md',ROOT/'tests/test_external_audio_masks.py']
    hashes={}
    for folder in ('external_audio_projection','xml_dcml_identity','external_audio_candidates'):
        hashes.update(read(ROOT/'reports'/folder/'source_hashes.json'))
    assert all(sha(p)==h for p,h in hashes.items())
    for p in paths:hashes[str(p)]=sha(p)
    regions=pd.read_csv(paths[0]);points=pd.read_csv(paths[1]);candidates=pd.read_csv(paths[2]).set_index('asap_performance');ann=read(paths[3])
    summary=[];mapping=[];positive=[];gridfiles=[];used=set();timing={}
    for performance,group in regions.groupby('performance'):
        assert time.monotonic()-began<120
        candidate=candidates.loc[performance];pid=group.piece_id.iloc[0];folder=group.folder.iloc[0]
        assert pid==candidate.dcml_piece_id and group.piece_id.nunique()==1
        source=points[points.performance==performance].copy()
        n=int(np.ceil(group.score_quarter_end.max()+1e-5))+1
        y,m,rid,covered,rows,runs=build_masks(group[['score_quarter_start','score_quarter_end']].to_numpy(),source.score_quarter.to_numpy(),n)
        if folder not in timing:timing[folder]=midi_timing(ASAP/folder/'midi_score.mid')[0]
        a=ann[performance];sq=seconds_to_quarters(a['midi_score_beats'],timing[folder]);pb=np.asarray(a['performance_beats'])
        grid=np.arange(n);time_valid=(grid>=sq[0]-1e-5)&(grid<=sq[-1]+1e-5)
        seconds=np.full(n,np.nan);seconds[time_valid]=np.interp(grid[time_valid],sq,pb)+float(candidate.audio_offset_seconds)
        assert (time_valid[m>0]).all() and np.isfinite(seconds[m>0]).all()
        assert ((seconds[m>0]>=0)&(seconds[m>0]<float(candidate.audio_seconds))).all()
        token=hashlib.sha256(performance.encode()).hexdigest()[:16];assert token not in used;used.add(token)
        destination=ART/f'{token}.npz'
        np.savez_compressed(destination,labels=y,label_mask=m,verified_coverage=covered,run_id=rid,
            audio_seconds=seconds,valid_audio_time=time_valid,quarterbeat=grid)
        with np.load(destination,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['labels'],y);np.testing.assert_array_equal(z['label_mask'],m)
            np.testing.assert_allclose(z['audio_seconds'],seconds,rtol=0,atol=0,equal_nan=True)
        gridfiles.append(dict(performance=performance,piece_id=pid,npz=str(destination),sha256=sha(destination),audio_path=candidate.audio_path))
        for row in rows:
            mapping.append(dict(performance=performance,piece_id=pid,**row))
            if row['effective_positive']:
                distances=np.abs(source.score_quarter-row['target_qb']);s=source.loc[distances.idxmin()];assert distances.min()<1e-5
                positive.append(dict(performance=performance,piece_id=pid,source_mc=int(s.source_mc),
                    source_onset_quarters=float(s.source_onset_quarters),score_quarter=row['target_qb'],beat_index=row['beat_index']))
        summary.append(dict(performance=performance,piece_id=pid,composer=candidate.composer,sonata=int(candidate.sonata),
            source_start_occurrences=len(source),continuous_verified_runs=len(runs),known_grid_positions=int(m.sum()),
            positive_grid_positions=int(y.sum()),negative_grid_positions=int((m-y).sum()),full_grid_positions=n,
            verified_grid_positions=int(covered.sum()),ambiguous_start_occurrences=sum(r['status']=='ambiguous_tie' for r in rows),
            runs_with_two_starts=sum(sum(r['run']==i for r in rows)>=2 for i in range(len(runs))),
            longest_verified_run_quarters=max(b-a for a,b in runs)))
    summary=pd.DataFrame(summary);positive=pd.DataFrame(positive)
    summary.to_csv(OUT/'performance_audit.csv',index=False);pd.DataFrame(mapping).to_csv(OUT/'label_quantization.csv',index=False)
    positive.to_csv(OUT/'effective_positive_references.csv',index=False);pd.DataFrame(gridfiles).to_csv(OUT/'grid_manifest.csv',index=False)
    usable=summary[(summary.positive_grid_positions>0)&(summary.negative_grid_positions>0)]
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    result=dict(status='conservative_reference_masks_prepared_no_training',performances_examined=len(summary),
        performances_with_positive_and_negative=len(usable),movements_with_positive_and_negative=int(usable.piece_id.nunique()),
        sonatas_with_positive_and_negative=len(usable[['composer','sonata']].drop_duplicates()),
        known_grid_positions=int(summary.known_grid_positions.sum()),positive_grid_positions=int(summary.positive_grid_positions.sum()),
        negative_grid_positions=int(summary.negative_grid_positions.sum()),
        effective_source_start_occurrences=len(positive),unique_effective_source_starts=len(positive[['piece_id','source_mc','source_onset_quarters']].drop_duplicates()) if len(positive) else 0,
        unknown_fraction=float(1-summary.known_grid_positions.sum()/summary.full_grid_positions.sum()),
        quantization_tie_occurrences=int(summary.ambiguous_start_occurrences.sum()),training_runs=0,
        masks_reloaded=True,source_hashes_unchanged=True,seconds=time.monotonic()-began,
        caution='Reference anchors, conservative partial masks, no audio feature extraction or split audit yet. Not Chopin F1. Never supply label_mask as model input.')
    write(OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
