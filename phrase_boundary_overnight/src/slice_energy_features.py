"""Versioned phrase-start labels and label-free score/tempo features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import choose_measure_path, to_float, _nearest_beat

SCALES = (1, 3, 6, 12, 24)

def tempo_hierarchy(tempo, time_mask=None):
    """Li2016 eq5.2-5.4 adaptation, NOT acoustic signal energy."""
    tempo = np.asarray(tempo, dtype=float)
    if tempo.ndim != 1 or not len(tempo) or not np.isfinite(tempo).all() or (tempo <= 0).any():
        raise ValueError('tempo must be finite, positive, nonempty and one-dimensional')
    time_mask = np.ones_like(tempo) if time_mask is None else np.asarray(time_mask)
    reference = np.median(tempo)
    levels = []
    quality = np.zeros_like(tempo)
    for scale in SCALES:
        level = np.zeros_like(tempo)
        for a in range(0, len(tempo), scale):
            b = min(a + scale, len(tempo)); block = tempo[a:b] / reference
            e = np.sqrt(np.mean(block ** 2)) - np.std(block, ddof=0)
            level[a:b] = np.log(max(e, 1e-8))
            if scale == SCALES[-1]: quality[a:b] = time_mask[a:b].mean()
        levels.append(level)
    features = np.column_stack(levels + [quality]).astype(np.float32)
    log_score = -2 * np.sum(levels, axis=0)
    relative = np.exp(log_score - log_score.max())
    probability = relative / relative.sum()
    return features, relative.astype(np.float32), probability

def start_labels(targets, n):
    """Unknown prefix/suffix and equidistant quantization are not negative labels."""
    targets = sorted(set(float(x) for x in targets if 0 <= x < n))
    labels = np.zeros(n, np.float32); mask = np.zeros(n, np.float32)
    rows = []
    if len(targets) >= 2:
        # Only score-time positions strictly after first known start, through last.
        grid = np.arange(n)
        mask[(grid > targets[0]) & (grid <= targets[-1])] = 1
    for j, t in enumerate(targets):
        status, index, distance = _nearest_beat(t, n, .5)
        if j > 0 and index is not None: labels[index] = 1
        if status == 'ambiguous_tie':
            for i in {int(np.floor(t)), int(np.ceil(t))}:
                if 0 <= i < n: mask[i] = 0
        rows.append({'target_qb':t,'beat_index':index,'distance':distance,'status':status,
                     'first_start':j == 0})
    # Sub-beat rounded final starts may fall just outside the interior interval.
    for r in rows[1:]:
        if r['beat_index'] is not None: mask[r['beat_index']] = 1
    # Reinstate ambiguity exclusions even if a neighbouring start mapped there.
    for r in rows:
        if r['status'] == 'ambiguous_tie':
            for i in {int(np.floor(r['target_qb'])), int(np.ceil(r['target_qb']))}:
                if 0 <= i < n: mask[i] = 0
    return labels, mask, rows

def map_starts(piece, n):
    measures = pd.read_csv(piece.measures_path, sep='\t')
    harmonies = pd.read_csv(piece.harmony_path, sep='\t')
    traversal, mode, error = choose_measure_path(measures, n)
    if not traversal: raise ValueError('score traversal mismatch: ' + piece.piece_id)
    folded = {int(r.mc):to_float(r.quarterbeats) for r in measures.itertuples()}
    starts = harmonies[harmonies.phraseend.fillna('').str.contains('{', regex=False)]
    targets = []; provenance = []
    for occurrence in traversal:
        mc = int(occurrence['mc'])
        for r in starts[starts.mc == mc].itertuples():
            t = occurrence['start_qb'] + to_float(r.quarterbeats) - folded[mc]
            targets.append(t)
            provenance.append({'piece_id':piece.piece_id,'mc':mc,'visit':occurrence['visit'],
                               'folded_qb':to_float(r.quarterbeats),'target_qb':t,'marker':r.phraseend})
    labels, mask, rows = start_labels(targets, n)
    unique = sorted(set(t for t in targets if 0 <= t < n))
    segments = [{'piece_id':piece.piece_id,'start_qb':a,'stop_qb':b,'length_beats':b-a,
                 'interval':'[start,stop)','status':'annotation_derived_not_manually_reviewed'}
                for a,b in zip(unique[:-1],unique[1:])]
    return labels, mask, rows, segments, provenance, mode, error

def voiced_events(piece, n):
    notes = pd.read_csv(piece.notes_path, sep='\t')
    measures = pd.read_csv(piece.measures_path, sep='\t')
    traversal, mode, error = choose_measure_path(measures, n)
    if not traversal: raise ValueError('No valid traversal')
    folded = {int(r.mc):to_float(r.quarterbeats) for r in measures.itertuples()}
    events = []
    for occurrence in traversal:
        mc = int(occurrence['mc'])
        for r in notes[notes.mc == mc].itertuples():
            onset = occurrence['start_qb'] + to_float(r.quarterbeats) - folded[mc]
            duration = max(to_float(r.duration_qb), 0)
            if np.isfinite(onset) and np.isfinite(duration):
                events.append((onset, duration, int(r.midi), int(r.staff), int(r.voice)))
    return events

def rest_before(events, n):
    output = np.zeros(n)
    for b in range(n):
        # Include a note ending exactly at b: there is no preceding silence.
        if any(o < b and o + d >= b - 1e-9 and d > 0 for o,d,*_ in events): continue
        ends = [o+d for o,d,*_ in events if d > 0 and o+d <= b]
        if ends: output[b] = np.clip(b-max(ends),0,8)/8
    return output

def voice_track(events, n, voice, lowest=False):
    selected = [e for e in events if e[3:5] == voice and e[1] > 0]
    delta = np.zeros(n); duration_change = np.zeros(n); direction = np.zeros(n)
    previous_pitch = previous_duration = previous_sign = None
    for b in range(n):
        attacks = [e for e in selected if b <= e[0] < b+1]
        if not attacks: continue
        # First onset within a beat anchors the transition; fixed score voice.
        first = min(e[0] for e in attacks)
        chord = [e for e in attacks if abs(e[0]-first) < 1e-8]
        note = (min if lowest else max)(chord, key=lambda e:e[2])
        pitch,duration = note[2],note[1]
        if previous_pitch is not None:
            delta[b] = pitch-previous_pitch
            sign = np.sign(delta[b])
            direction[b] = float(sign != 0 and previous_sign is not None and previous_sign != 0 and sign != previous_sign)
            if sign != 0: previous_sign = sign
            duration_change[b] = abs(np.log1p(duration)-np.log1p(previous_duration))
        previous_pitch,previous_duration = pitch,duration
    return delta,duration_change,direction

def corrected_score(old, events):
    n = len(old); rest = old.copy(); rest[:,4] = rest_before(events,n)
    fixed = rest.copy()
    voices = sorted(set(e[3:5] for e in events))
    if not voices: raise ValueError('No score voices')
    melody = (1,1) if (1,1) in voices else voices[0]
    bottom = max(v[0] for v in voices)
    bass = max([v for v in voices if v[0] == bottom],
               key=lambda v:sum(e[3:5] == v for e in events))
    topdelta,dur,turn = voice_track(events,n,melody)
    bassdelta,_,_ = voice_track(events,n,bass,lowest=True)
    fixed[:,6] = dur; fixed[:,7] = np.clip(topdelta/12,-4,4)
    fixed[:,8] = np.clip(abs(topdelta)/12,0,4); fixed[:,9] = turn
    fixed[:,13] = np.clip(abs(bassdelta)/12,0,4)
    # Recompute the old heuristic grouping-strength channel with corrected inputs.
    for score in [rest,fixed]:
        pitch = score[:,8]; timing = np.maximum(score[:,3],score[:,4])
        pc = np.zeros(n); tc = np.zeros(n)
        pc[1:] = abs(np.diff(pitch))/np.maximum(abs(pitch[1:])+abs(pitch[:-1]),1e-6)
        tc[1:] = abs(np.diff(timing))/np.maximum(abs(timing[1:])+abs(timing[:-1]),1e-6)
        score[:,15] = np.clip(.25*(pc+tc)+.5*np.maximum(pitch,timing),0,4)
    return rest,fixed,{'melody_staff_voice':list(melody),'bass_staff_voice':list(bass),
                       'voices':list(map(list,voices)),'melody_is_heuristic':True}
