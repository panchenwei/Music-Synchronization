"""Read-only shape of remaining note-attack mismatches; no fuzzy remapping."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .external_audio_candidates import ASAP
from .external_audio_projection import midi_timing
from .external_audio_tie_audit import xml_sound_events


def main():
    began=time.monotonic();out=ROOT/'reports/external_audio_rejection_profile';out.mkdir(parents=True,exist_ok=True)
    inp=ROOT/'reports/external_audio_tie_audit/interval_diagnosis.csv'
    table=pd.read_csv(inp);failed=table[~table.tie_aware_pass];rows=[];cache={}
    hashes=read(ROOT/'reports/external_audio_tie_audit/source_hashes.json')
    for p in (Path(__file__),inp):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    for r in failed.itertuples():
        assert time.monotonic()-began<120
        if r.folder not in cache:cache[r.folder]=(xml_sound_events(ASAP/r.folder/'xml_score.musicxml'),midi_timing(ASAP/r.folder/'midi_score.mid')[1])
        views,notes=cache[r.folder];v=views[r.xml_index];q0=r.score_quarter_start;q1=r.score_quarter_end
        selected=notes[(notes[:,0]>=q0-1e-4)&(notes[:,0]<q1-1e-4)]
        left=Counter((round(o*96),p) for o,p,d in v['attacks']);right=Counter((round((o-q0)*96),int(p)) for o,p in selected)
        missing=left-right;extra=right-left;pl=Counter();pr=Counter()
        for (o,p),n in left.items():pl[p]+=n
        for (o,p),n in right.items():pr[p]+=n
        if v['conditional']:category='conditional_sound_tie'
        elif left==right:category='attack_equal_but_sustain_or_rounding_failed'
        elif set(left)==set(right):category='identical_onset_pitch_sets_different_multiplicity'
        elif pl==pr:category='same_pitch_multiset_different_onsets'
        elif set(left).issubset(set(right)):category='all_expected_attack_pairs_present_plus_midi_extras'
        elif set(right).issubset(set(left)):category='all_midi_attack_pairs_present_plus_xml_extras'
        else:category='both_missing_and_extra_attack_pairs'
        rows.append(dict(performance=r.performance,piece_id=r.piece_id,folder=r.folder,xml_index=r.xml_index,
            occurrence=r.occurrence,category=category,expected_attacks=sum(left.values()),midi_attacks=sum(right.values()),
            missing_count=sum(missing.values()),extra_count=sum(extra.values()),source_start_rows=r.source_start_rows,
            missing_examples=str(sorted(missing.items())[:6]),extra_examples=str(sorted(extra.items())[:6])))
    result=pd.DataFrame(rows);result.to_csv(out/'mismatch_examples.csv',index=False)
    groups=result.groupby('category').agg(intervals=('category','size'),start_occurrences=('source_start_rows','sum'),
        score_versions=('piece_id','nunique'));groups.to_csv(out/'category_summary.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(out/'source_hashes.json',hashes)
    write(out/'completion_audit.json',dict(status='diagnostic_complete',failed_intervals_examined=len(result),training_runs=0,
        automatic_remapping=False,source_hashes_unchanged=True,seconds=time.monotonic()-began,
        caution='Counter-pattern descriptions, not verified causal explanations or authorization to weaken identity checks.'))
    print(groups.to_string(),flush=True)


if __name__=='__main__':main()
