from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mido
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .data import build_curve_features
from .models import BeatBoundaryTCN, Normalizer
from .phase3_features import SCORE_CUE_NAMES, _relative_change, _safe_cosine_novelty
from .phase3_models import ResidualGatedFusionTCN


def midi_note_events(midi_path: Path) -> tuple[list[tuple[float, float, int]], int]:
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    absolute = 0
    active: dict[tuple[int, int], list[int]] = {}
    events: list[tuple[float, float, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        absolute += message.time
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault((message.channel, message.note), []).append(absolute)
        elif message.type in {"note_off", "note_on"}:
            key = (message.channel, message.note)
            if active.get(key):
                start = active[key].pop(0)
                events.append((start / ticks_per_beat, max((absolute - start) / ticks_per_beat, 0.0), int(message.note)))
    return events, ticks_per_beat


def midi_score_cues(midi_path: Path, beats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create the Phase-3 schema from deployable MIDI and score beat ticks."""
    events, ticks_per_beat = midi_note_events(midi_path)
    positions = pd.to_numeric(beats["score_tick"], errors="coerce").to_numpy(float) / ticks_per_beat
    n = len(positions)
    step = np.diff(positions, append=positions[-1] + (np.median(np.diff(positions)) if n > 1 else 1.0))
    groups, active = [], []
    for position, width in zip(positions, step):
        groups.append([(o, d, m) for o, d, m in events if position <= o < position + width])
        active.append([(o, d, m) for o, d, m in events if o < position + width and o + d > position])
    onset = np.asarray([len(x) for x in groups], float)
    active_count = np.asarray([len(x) for x in active], float)
    top = np.asarray([max((m for _, _, m in x), default=np.nan) for x in groups])
    bass = np.asarray([min((m for _, _, m in x), default=np.nan) for x in groups])
    durations = np.asarray([np.mean([d for _, d, _ in x]) if x else np.nan for x in groups])
    ranges = np.asarray([(max(m for _, _, m in x) - min(m for _, _, m in x)) if x else 0 for x in groups], float)
    top_fill = pd.Series(top).interpolate(limit_direction="both").fillna(60).to_numpy()
    bass_fill = pd.Series(bass).interpolate(limit_direction="both").fillna(48).to_numpy()
    dur_fill = pd.Series(durations).interpolate(limit_direction="both").fillna(0).to_numpy()
    top_delta = np.diff(top_fill, prepend=top_fill[0]); bass_delta = np.diff(bass_fill, prepend=bass_fill[0])
    direction = np.sign(top_delta); direction_change = np.zeros(n)
    direction_change[1:] = ((direction[1:] != 0) & (direction[:-1] != 0) & (direction[1:] != direction[:-1])).astype(float)
    pc = np.zeros((n, 12)); novelty = np.zeros(n); novelty_valid = np.zeros(n)
    for i, group in enumerate(groups):
        for _, _, pitch in group: pc[i, pitch % 12] += 1
        if pc[i].sum(): pc[i] /= pc[i].sum()
        if i:
            novelty[i], valid = _safe_cosine_novelty(pc[i - 1], pc[i]); novelty_valid[i] = valid
    onset_gap = np.zeros(n); rest = np.zeros(n)
    for i, position in enumerate(positions):
        po = [o for o, _, _ in events if o <= position]
        pe = [o + d for o, d, _ in events if o + d <= position]
        if po: onset_gap[i] = min(max(position - max(po), 0), 8) / 8
        if pe: rest[i] = min(max(position - max(pe), 0), 8) / 8
    beat_in_measure = pd.to_numeric(beats.get("beat_in_measure", pd.Series(np.ones(n))), errors="coerce").fillna(1).to_numpy(float) - 1
    numerator = pd.to_numeric(beats.get("numerator", pd.Series(np.ones(n))), errors="coerce").fillna(1).to_numpy(float)
    phase = np.divide(beat_in_measure, np.maximum(numerator, 1))
    pitch_mag = np.abs(top_delta) / 12; time_mag = np.maximum(onset_gap, rest)
    pc_change = np.zeros(n); time_change = np.zeros(n)
    for i in range(1, n):
        pc_change[i] = _relative_change(pitch_mag[i - 1], pitch_mag[i]); time_change[i] = _relative_change(time_mag[i - 1], time_mag[i])
    lbdm = np.clip(.25 * (pc_change + time_change) + .5 * np.maximum(pitch_mag, time_mag), 0, 4)
    cues = np.column_stack([
        np.sin(2*np.pi*phase), np.cos(2*np.pi*phase), (beat_in_measure == 0).astype(float), onset_gap, rest,
        np.maximum(active_count-onset, 0)/np.maximum(active_count, 1),
        np.abs(np.diff(np.log1p(dur_fill), prepend=np.log1p(dur_fill[0]))),
        np.clip(top_delta/12, -4, 4), np.clip(np.abs(top_delta)/12, 0, 4), direction_change,
        np.log1p(onset), np.abs(np.diff(np.log1p(active_count), prepend=np.log1p(active_count[0]))),
        np.clip(ranges/24, 0, 4), np.clip(np.abs(bass_delta)/12, 0, 4), novelty, lbdm,
    ]).astype(np.float32)
    quality = np.ones_like(cues); valid_onset = (onset > 0).astype(np.float32)
    quality[:, 6:10] = valid_onset[:, None]; quality[:, 12:15] = np.column_stack([valid_onset, valid_onset, novelty_valid])
    quality[0, 6:10] = 0; quality[0, 13:16] = 0
    return cues, quality


def beat_interval_energy(wav_path: Path, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    audio, sample_rate = sf.read(wav_path, always_2d=True)
    mono = audio.mean(axis=1).astype(float)
    energy = np.zeros(len(times), dtype=float); valid = np.zeros(len(times), dtype=float)
    finite_intervals = np.diff(times[np.isfinite(times)])
    fallback = float(np.median(finite_intervals[finite_intervals > 0])) if np.any(finite_intervals > 0) else 0.5
    for i, start in enumerate(times):
        end = times[i + 1] if i + 1 < len(times) and np.isfinite(times[i + 1]) else start + fallback
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        a = max(0, int(round(start * sample_rate))); b = min(len(mono), int(round(end * sample_rate)))
        if b > a:
            energy[i] = 20 * np.log10(np.sqrt(np.mean(mono[a:b] ** 2)) + 1e-8); valid[i] = 1
    if valid.any():
        energy = pd.Series(np.where(valid > 0, energy, np.nan)).interpolate(limit_direction="both").fillna(0).to_numpy()
    return energy, valid


def load_m2_model(curve_checkpoint_path: Path, m2_checkpoint_path: Path, device: torch.device):
    curve_state = torch.load(curve_checkpoint_path, map_location=device, weights_only=False)
    curve = BeatBoundaryTCN(int(curve_state["input_dim"]), 32, [1, 2, 4, 8], 3, 0.2)
    curve.load_state_dict(curve_state["model"])
    model = ResidualGatedFusionTCN(curve).to(device)
    state = torch.load(m2_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"]); model.eval()
    curve_norm = Normalizer(np.asarray(state["curve_normalizer_mean"]), np.asarray(state["curve_normalizer_std"]))
    score_norm = Normalizer(np.asarray(state["score_normalizer_mean"]), np.asarray(state["score_normalizer_std"]))
    return model, curve_norm, score_norm, state


def run_frontend_smoke(
    predicted_beats_csv: Path,
    score_midi: Path,
    performance_wav: Path,
    curve_checkpoint: Path,
    m2_checkpoint: Path,
    out_csv: Path,
    out_json: Path,
) -> dict[str, Any]:
    beats = pd.read_csv(predicted_beats_csv)
    times = pd.to_numeric(beats["predicted_performance_time_sec"], errors="coerce").to_numpy(float)
    energy, energy_valid = beat_interval_energy(performance_wav, times)
    safe_times = pd.Series(times).interpolate(limit_direction="both").to_numpy(float)
    curves = build_curve_features(safe_times, energy)
    curves[:, 7] = np.isfinite(times).astype(np.float32); curves[:, 8] = energy_valid.astype(np.float32)
    score, quality = midi_score_cues(score_midi, beats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, curve_norm, score_norm, state = load_m2_model(curve_checkpoint, m2_checkpoint, device)
    c = torch.from_numpy(curve_norm.apply(curves).astype(np.float32)).unsqueeze(0).to(device)
    s = torch.from_numpy(score_norm.apply(score).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, curve_logits, residual, gate = model(c, s, return_parts=True)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    output = pd.DataFrame({
        "score_beat_index": beats["beat_index"].astype(int),
        "measure": beats.get("measure_number_guess"),
        "beat": beats.get("beat_in_measure"),
        "boundary_probability": probs,
        "predicted_performance_time_sec": times,
        "alignment_valid": beats["inside_warping_range"].astype(bool) & np.isfinite(times),
        "curve_feature_valid_fraction": np.column_stack([curves[:, 7], curves[:, 8]]).mean(axis=1),
        "score_feature_quality_fraction": quality.mean(axis=1),
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True); output.to_csv(out_csv, index=False)
    payload = {
        "status": "implemented_smoke_tested",
        "not_ground_truth_evaluated": True,
        "model": "phase3_M2_fold0_seed42",
        "threshold": float(state["threshold"]),
        "gate": float(gate.cpu()),
        "input_rows": len(output),
        "alignment_valid_rows": int(output.alignment_valid.sum()),
        "score_cue_names": SCORE_CUE_NAMES,
        "predictions": output.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
