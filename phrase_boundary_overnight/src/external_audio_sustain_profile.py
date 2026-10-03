"""Inspect early MIDI releases behind the stricter full-duration tie check."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .external_audio_candidates import ASAP
from .external_audio_tie_audit import xml_sound_events,midi_intervals


def main():
    began=time.monotonic();out=ROOT/'reports/external_audio_sustain_profile';out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'reports/external_audio_rejection_profile/mismatch_examples.csv';table=pd.read_csv(path)
    selected=table[table.category=='attack_equal_but_sustain_or_rounding_failed']
    detail=pd.read_csv(ROOT/'reports/external_audio_tie_audit/interval_diagnosis.csv').set_index(['performance','occurrence'])
    hashes=read(ROOT/'reports/external_audio_rejection_profile/source_hashes.json');hashes[str(Path(__file__))]=sha(Path(__file__));hashes[str(path)]=sha(path)
    assert all(sha(p)==h for p,h in hashes.items());cache={};notes=[];rows=[]
    for r in selected.itertuples():
        assert time.monotonic()-began<120
        if r.folder not in cache:cache[r.folder]=(xml_sound_events(ASAP/r.folder/'xml_score.musicxml'),midi_intervals(ASAP/r.folder/'midi_score.mid'))
        events,intervals=cache[r.folder];q0=detail.loc[(r.performance,r.occurrence),'score_quarter_start'];active_all=True;full_all=True;count=0
        for onset,pitch,duration in events[r.xml_index]['continuations']:
            t=q0+onset;active=intervals[(intervals[:,2]==pitch)&(intervals[:,0]<t-1e-4)&(intervals[:,1]>t+1e-4)]
            has=len(active)>0;active_all &= has;gap=float(t+duration-active[:,1].max()) if has else np.nan
            full=bool(has and gap<=1e-3);full_all &= full;count+=1
            notes.append(dict(performance=r.performance,piece_id=r.piece_id,xml_index=r.xml_index,occurrence=r.occurrence,
                pitch=pitch,onset_quarter=t,notated_duration=duration,active_at_tie_onset=has,
                whole_notated_duration_covered=full,early_release_quarters=gap,relative_early_release=gap/duration if has else np.nan))
        rows.append(dict(performance=r.performance,piece_id=r.piece_id,xml_index=r.xml_index,occurrence=r.occurrence,
            continuation_count=count,all_active_at_tie_onset=active_all,all_full_notated_duration=full_all,source_start_rows=r.source_start_rows))
    n=pd.DataFrame(notes);r=pd.DataFrame(rows);n.to_csv(out/'note_release_diagnosis.csv',index=False);r.to_csv(out/'interval_summary.csv',index=False)
    failed=n[n.active_at_tie_onset&~n.whole_notated_duration_covered]
    quant=failed[['early_release_quarters','relative_early_release']].quantile([0,.25,.5,.75,.9,.99,1]);quant.to_csv(out/'release_quantiles.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(out/'source_hashes.json',hashes)
    result=dict(status='diagnostic_complete_no_labels_changed',intervals_examined=len(r),
        all_continuations_active_at_onset=int(r.all_active_at_tie_onset.sum()),
        all_onset_active_source_start_occurrences=int(r.loc[r.all_active_at_tie_onset,'source_start_rows'].sum()),
        full_duration_failed_notes=len(failed),no_sound_at_continuation_notes=int((~n.active_at_tie_onset).sum()),
        training_runs=0,source_hashes_unchanged=True,seconds=time.monotonic()-began,
        caution='Early releases observed, but no approximate duration gate adopted or audio manually reviewed.')
    write(out/'completion_audit.json',result);print(result,flush=True);print(quant.to_string(),flush=True)


if __name__=='__main__':main()
