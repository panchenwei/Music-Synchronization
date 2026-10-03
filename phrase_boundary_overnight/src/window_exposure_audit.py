"""Replay sampler decisions without tensors; verify RNG against saved checkpoints."""
import copy
import numpy as np
from .phase6_models import window_starts


def exposure(data, seed, steps, batch_size=32, window=64):
    pieces=sorted(data);rng=np.random.default_rng(seed)
    starts={p:window_starts(len(data[p]['labels']),window,max(window//2,1)) for p in pieces}
    assert all(len(data[p]['curves'])>0 for p in pieces)
    score_seen=set();performance_seen=set();valid_beats=0.;piece_draws={p:0 for p in pieces}
    for _ in range(steps*batch_size):
        p=str(rng.choice(pieces));perf=int(rng.integers(len(data[p]['curves'])));start=int(rng.choice(starts[p]))
        score_seen.add((p,start));performance_seen.add((p,perf,start));piece_draws[p]+=1
        valid_beats+=float(data[p]['label_mask'][start:start+window].sum())
    nscore=sum(len(v) for v in starts.values())
    nperf=sum(len(starts[p])*len(data[p]['curves']) for p in pieces)
    nlabels=sum(float(data[p]['label_mask'].sum()) for p in pieces)
    nperflabels=sum(float(data[p]['label_mask'].sum())*len(data[p]['curves']) for p in pieces)
    return dict(steps=int(steps),window_draws=int(steps*batch_size),valid_loss_beat_exposures=valid_beats,
                unique_score_windows=len(score_seen),possible_score_windows=nscore,score_window_coverage=len(score_seen)/nscore,
                unique_performance_windows=len(performance_seen),possible_performance_windows=nperf,
                performance_window_coverage=len(performance_seen)/nperf,
                score_beat_pass_equivalent=valid_beats/nlabels,performance_beat_pass_equivalent=valid_beats/nperflabels,
                piece_draws=piece_draws,sampler=dict(rng=copy.deepcopy(rng.bit_generator.state)))
