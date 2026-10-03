"""Label-free multi-scale score change cues, aligned before beat b."""
import numpy as np

SCALES = (3, 6, 12, 24)


def unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite [beats,channels]')
    den = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, den, out=np.zeros_like(x), where=den > 0)


def change_curve(x, scale):
    """Even-size Gaussian checkerboard on dot-product SSM, no O(n^2) storage.

    Left [b-L,b), right [b,b+L); normalized kernel has absolute sum one.
    A missing full side returns zero, and a separate availability flag.
    """
    z = unit_rows(x)
    n = len(z)
    if scale < 1:
        raise ValueError('scale must be positive')
    weights = np.exp(-.5*((np.arange(scale)+.5)/(scale/2.))**2)
    weights /= weights.sum()
    novelty, available = np.zeros(n), np.zeros(n)
    for b in range(scale, n-scale+1):
        left = weights[::-1] @ z[b-scale:b]
        right = weights @ z[b:b+scale]
        novelty[b] = np.dot(left-right, left-right) / 4.
        available[b] = 1.
    return novelty.astype(np.float32), available.astype(np.float32)


def onset_rhythm(events, n):
    """Six notated-duration bins plus onset density; no phrase/harmony labels."""
    x = np.zeros((n, 7), dtype=np.float64)
    for onset, duration, *_ in events:
        if duration <= 0:
            continue
        b = int(np.floor(onset + 1e-9))
        if 0 <= b < n:
            i = np.searchsorted([.25, .5, 1., 2., 4.], duration, side='left')
            x[b, i] += 1
    counts = x[:, :6].sum(axis=1)
    x[:, :6] = np.divide(x[:, :6], counts[:, None], out=np.zeros_like(x[:, :6]), where=counts[:, None] > 0)
    x[:, 6] = np.log1p(counts)
    return x.astype(np.float32)


def score_novelty(pitch_profiles, events):
    p = np.asarray(pitch_profiles)
    if p.ndim != 2 or p.shape[1] != 24:
        raise ValueError('Expected cached [beats,24] pitch profiles')
    n = len(p)
    streams = [p[:, :12], p[:, 12:], onset_rhythm(events, n)]
    out = np.zeros((n, 24), dtype=np.float32)
    for stream_index, x in enumerate(streams):
        for scale_index, scale in enumerate(SCALES):
            novelty, available = change_curve(x, scale)
            out[:, stream_index*4+scale_index] = novelty
            out[:, 12+scale_index] = available
    # Last eight columns deliberately zero to keep the frozen 58-D control model.
    return out
