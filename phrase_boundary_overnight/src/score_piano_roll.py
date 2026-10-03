"""Label-free beat-anchored piano rolls; no time shifting, edge deletion or velocity inference."""
import numpy as np


def piano_roll(events, n, subdivisions=4):
    """Return [beat, occupancy/onset, MIDI pitch, within-beat tick].

    Occupancy is the exact union-duration fraction within each tick, not note
    count or acoustic energy. Onsets are binary existence. Grace notes with
    nonpositive duration are excluded. Intervals are half-open [onset, end).
    """
    if n < 1 or subdivisions < 1 or int(n) != n or int(subdivisions) != subdivisions:
        raise ValueError('Positive integer lengths required')
    n, subdivisions = int(n), int(subdivisions)
    grid = np.zeros((2, 128, n*subdivisions), np.float32)
    by_pitch = {}
    for o, d, p, *_ in events:
        o, d = float(o), float(d)
        if not np.isfinite([o, d, p]).all() or not 0 <= p <= 127 or int(p) != p:
            raise ValueError('Invalid note event')
        p = int(p)
        if d <= 0:
            continue
        if 0 <= o < n:
            tick = min(int(np.floor(o*subdivisions+1e-9)), n*subdivisions-1)
            grid[1, p, tick] = 1
        a, b = max(0., o), min(float(n), o+d)
        if a < b:
            by_pitch.setdefault(p, []).append((a, b))
    for pitch, intervals in by_pitch.items():
        merged = []
        for a, b in sorted(intervals):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        for a, b in merged:
            first = max(0, int(np.floor(a*subdivisions)))
            last = min(n*subdivisions, int(np.ceil(b*subdivisions)))
            for tick in range(first, last):
                overlap = max(0., min(b, (tick+1)/subdivisions)-max(a, tick/subdivisions))
                grid[0, pitch, tick] += overlap*subdivisions
    grid[0] = np.clip(grid[0], 0, 1)
    result = grid.reshape(2,128,n,subdivisions).transpose(2,0,1,3).copy()
    assert result.shape == (n,2,128,subdivisions) and np.isfinite(result).all()
    return result
