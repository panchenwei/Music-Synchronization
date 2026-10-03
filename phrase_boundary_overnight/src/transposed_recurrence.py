"""Fixed-shift ordered recurrence; no label access, no per-beat shift selection."""
import numpy as np
from .motif_recurrence_features import SCALES, staff_sequences


def shift_chroma(x, shift):
    """Shift each onset/occupancy chroma in each subtick, not the time axis."""
    x = np.asarray(x)
    assert x.ndim == 2 and x.shape[1] == 96
    return np.roll(x.reshape(len(x), 4, 2, 12), int(shift), axis=-1).reshape(x.shape)


def window_cross_similarity(s, w):
    n = len(s); m = n-w+1
    if m <= 0: return np.empty((0, 0), float)
    out = np.zeros((m, m), float)
    for offset in range(w): out += s[offset:offset+m, offset:offset+m]
    return np.clip(out/w, 0, 1)


def transposed_stream(x, shifts=tuple(range(12)), scales=SCALES):
    """Maximize cues over candidate AND a single consistent context shift.

    Forward and backward values in restart must use the same candidate/shift.
    This is not independently transposing every beat or both context sides.
    """
    n = len(x); out = np.zeros((n, len(scales)*4), float)
    assert shifts and len(set(int(s)%12 for s in shifts)) == len(shifts)
    for shift in shifts:
        s = x @ shift_chroma(x, shift).T
        for k, w in enumerate(scales):
            sim = window_cross_similarity(s, w); positions = np.arange(w, n-w+1)
            for b in positions:
                js = positions[abs(positions-b) >= 2*w]
                if len(js) == 0 or b >= n: continue
                f = sim[b, js]; back = sim[b-w, js-w]
                out[b, k*4:k*4+4] = np.maximum(out[b, k*4:k*4+4],
                    [f.max(), back.max(), np.max(f*(1-back)), 1.])
    return out.astype(np.float32)


def features(events, n, shifts=tuple(range(12))):
    return np.concatenate([transposed_stream(x, shifts) for x in staff_sequences(events, n)], axis=1)
