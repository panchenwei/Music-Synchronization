"""Annotation-free global-tonic relative profiles; fixed KS weights and temperature."""
import numpy as np

MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR = np.array([6.33,2.68,3.52,5.38,2.6,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
TEMPERATURE = .1


def duration_histogram(events, n):
    h = np.zeros(12, dtype=np.float64)
    for o,d,p,*_ in events:
        if not np.isfinite([o,d,p]).all():
            raise ValueError('nonfinite note')
        duration = max(0., min(float(n), o+d)-max(0., o)) if d > 0 else 0.
        h[int(p)%12] += duration
    return h


def key_correlations(hist):
    h = np.asarray(hist, dtype=np.float64)
    if h.shape != (12,) or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError('finite nonnegative 12-bin histogram required')
    h = h-h.mean()
    weights = np.stack([np.roll(p,t) for p in (MAJOR,MINOR) for t in range(12)])
    weights -= weights.mean(axis=1, keepdims=True)
    den = np.linalg.norm(h)*np.linalg.norm(weights,axis=1)
    return np.divide(weights@h,den,out=np.zeros(24),where=den>1e-12).reshape(2,12)


def tonic_weights(hist, kind):
    scores = key_correlations(hist)
    if kind == 'K':
        w = np.isclose(scores, scores.max(), rtol=0., atol=1e-12).astype(float)
    elif kind == 'S':
        w = np.exp((scores-scores.max())/TEMPERATURE)
    else:
        raise ValueError('kind must be K or S')
    return (w/w.sum()).sum(axis=0)


def relative_profiles(profiles, hist, kind):
    x = np.asarray(profiles,dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 24 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('nonnegative [beats,24] required')
    w = tonic_weights(hist,kind)
    y = sum(w[t]*np.concatenate([np.roll(x[:,:12],-t,axis=1),np.roll(x[:,12:],-t,axis=1)],axis=1) for t in range(12))
    return y.astype(np.float32)
