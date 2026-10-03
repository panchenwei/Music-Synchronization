"""Apply unchanged conservative mask semantics to the one-tick projection v2."""
from pathlib import Path
import hashlib,time
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .external_audio_candidates import ASAP
from .external_audio_projection import midi_timing,seconds_to_quarters
from .external_audio_masks import build_masks


def main():
    began=time.monotonic();out=ROOT/'reports/external_audio_masks_v2';art=ROOT/'artifacts/external_audio_masks_v2'
    out.mkdir(parents=True,exist_ok=True);art.mkdir(parents=True,exist_ok=True);previous=ROOT/'reports/external_audio_projection_v2'
    assert read(previous/'completion_audit.json')['status']=='reference_projection_audited_not_training_ready'
    rp=previous/'verified_bar_intervals.csv';pp=previous/'projected_start_references.csv';cp=ROOT/'reports/external_audio_candidates/audio_candidates.csv'
    regions=pd.read_csv(rp);points=pd.read_csv(pp);candidates=pd.read_csv(cp).set_index('asap_performance');ann=read(ASAP/'asap_annotations.json')
    oldmanifest=pd.read_csv(ROOT/'reports/external_audio_masks/grid_manifest.csv').set_index('performance')
    hashes=read(previous/'source_hashes.json');hashes.update(read(ROOT/'reports/external_audio_masks/source_hashes.json'))
    for p in (Path(__file__),rp,pp,cp,ROOT/'src/external_audio_masks.py'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());rows=[];mapping=[];positives=[];manifest=[];timings={}
    for performance,group in regions.groupby('performance'):
        assert time.monotonic()-began<120
        candidate=candidates.loc[performance];pid=candidate.dcml_piece_id;folder=candidate.asap_folder
        source=points[points.performance==performance];n=int(np.ceil(group.score_quarter_end.max()+1e-5))+1
        y,m,rid,covered,labelrows,runs=build_masks(group[['score_quarter_start','score_quarter_end']].to_numpy(),source.score_quarter.to_numpy(),n)
        if folder not in timings:timings[folder]=midi_timing(ASAP/folder/'midi_score.mid')[0]
        a=ann[performance];sq=seconds_to_quarters(a['midi_score_beats'],timings[folder]);pb=np.asarray(a['performance_beats'])
        grid=np.arange(n);valid=(grid>=sq[0]-1e-5)&(grid<=sq[-1]+1e-5);seconds=np.full(n,np.nan)
        seconds[valid]=np.interp(grid[valid],sq,pb)+float(candidate.audio_offset_seconds)
        assert valid[m>0].all() and np.isfinite(seconds[m>0]).all()
        assert ((seconds[m>0]>=0)&(seconds[m>0]<candidate.audio_seconds)).all()
        oldpath=Path(oldmanifest.loc[performance,'npz']);assert sha(oldpath)==oldmanifest.loc[performance,'sha256'];hashes[str(oldpath)]=sha(oldpath)
        with np.load(oldpath,allow_pickle=False) as old:
            take=np.flatnonzero(old['label_mask']);assert m[take].all();np.testing.assert_array_equal(y[take],old['labels'][take])
            previous_positive=int(old['labels'].sum());previous_known=int(old['label_mask'].sum())
        token=hashlib.sha256(performance.encode()).hexdigest()[:16];path=art/f'{token}.npz'
        np.savez_compressed(path,labels=y,label_mask=m,verified_coverage=covered,run_id=rid,audio_seconds=seconds,valid_audio_time=valid,quarterbeat=grid)
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['labels'],y);np.testing.assert_array_equal(z['label_mask'],m)
            np.testing.assert_allclose(z['audio_seconds'],seconds,atol=0,rtol=0,equal_nan=True)
        manifest.append(dict(performance=performance,piece_id=pid,npz=str(path),sha256=sha(path),audio_path=candidate.audio_path))
        for row in labelrows:
            mapping.append(dict(performance=performance,piece_id=pid,**row))
            if row['effective_positive']:
                distances=abs(source.score_quarter-row['target_qb']);s=source.loc[distances.idxmin()];assert distances.min()<1e-5
                positives.append(dict(performance=performance,piece_id=pid,source_mc=int(s.source_mc),source_onset_quarters=s.source_onset_quarters,score_quarter=row['target_qb'],beat_index=row['beat_index']))
        rows.append(dict(performance=performance,piece_id=pid,composer=candidate.composer,sonata=candidate.sonata,
            known_grid_positions=int(m.sum()),positive_grid_positions=int(y.sum()),negative_grid_positions=int((m-y).sum()),full_grid_positions=n,
            old_known_grid_positions=previous_known,old_positive_grid_positions=previous_positive,continuous_verified_runs=len(runs)))
    frame=pd.DataFrame(rows);p=pd.DataFrame(positives);good=frame[(frame.positive_grid_positions>0)&(frame.negative_grid_positions>0)]
    frame.to_csv(out/'performance_audit.csv',index=False);p.to_csv(out/'effective_positive_references.csv',index=False)
    pd.DataFrame(mapping).to_csv(out/'label_quantization.csv',index=False);pd.DataFrame(manifest).to_csv(out/'grid_manifest.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items());write(out/'source_hashes.json',hashes)
    result=dict(status='conservative_reference_masks_prepared_no_training',version=2,performances_examined=len(frame),
        performances_with_positive_and_negative=len(good),movements_with_positive_and_negative=int(good.piece_id.nunique()),
        sonatas_with_positive_and_negative=len(good[['composer','sonata']].drop_duplicates()),known_grid_positions=int(frame.known_grid_positions.sum()),
        positive_grid_positions=int(frame.positive_grid_positions.sum()),negative_grid_positions=int(frame.negative_grid_positions.sum()),
        unique_effective_source_starts=len(p[['piece_id','source_mc','source_onset_quarters']].drop_duplicates()),
        unknown_fraction=float(1-frame.known_grid_positions.sum()/frame.full_grid_positions.sum()),
        old_known_labels_and_masks_preserved=True,masks_reloaded=True,source_hashes_unchanged=True,training_runs=0,seconds=time.monotonic()-began,
        caution='No audio features or model training. Reference anchors and partial masks; group split/source-audio audit still required.')
    write(out/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
