"""Fixed, label-free start/end pairing. No phrase-length or cadence oracle."""
import numpy as np
from .phase2_models import nms_probabilities


def decode(start, end, start_threshold, end_threshold):
    start, end = np.asarray(start, float), np.asarray(end, float)
    if start.ndim != 1 or start.shape != end.shape:
        raise ValueError('Equal one-dimensional probabilities required')
    if not (np.isfinite(start).all() and np.isfinite(end).all() and
            ((start >= 0) & (start <= 1)).all() and ((end >= 0) & (end <= 1)).all()):
        raise ValueError('Invalid probabilities')
    if not (0 < start_threshold < 1 and 0 < end_threshold < 1):
        raise ValueError('Thresholds must be strictly between zero and one')
    sp = nms_probabilities(start) >= start_threshold
    ep = nms_probabilities(end) >= end_threshold
    def weights(p, threshold):
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.maximum(np.log(p/(1-p))-np.log(threshold/(1-threshold)), 0) + 1e-8
    sw, ew = weights(start, start_threshold), weights(end, end_threshold)
    # State 0 outside, 1 inside. Every transition reads the PREVIOUS beat state.
    value = np.array([0., -np.inf]); back = []
    for t in range(len(start)):
        new = value.copy(); links = [(0, ''), (1, '')]
        transitions = []
        if sp[t]: transitions.append((0, 1, sw[t], 'S'))
        if ep[t]: transitions.append((1, 0, ew[t], 'E'))
        if sp[t] and ep[t]: transitions.append((1, 1, sw[t]+ew[t], 'ES'))
        for old, dest, gain, action in transitions:
            candidate = value[old] + gain
            if candidate > new[dest]: new[dest] = candidate; links[dest] = (old, action)
        value = new; back.append(links)
    state = 0; actions = []
    for t in range(len(start)-1, -1, -1):
        state, action = back[t][state]
        if action: actions.append((t, action))
    assert state == 0
    spans = []; opened = None
    for t, action in reversed(actions):
        if 'E' in action:
            assert opened is not None and opened < t
            spans.append((opened, t)); opened = None
        if 'S' in action: opened = t
    assert opened is None
    scores = np.zeros_like(start)
    for s, e in spans:
        assert sp[s] and ep[e]
        scores[s] = start[s]
    return scores, spans, float(value[0])
