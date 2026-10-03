from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DcmlPiece, choose_measure_path, to_float


SCORE_CUE_NAMES = [
    "metric_phase_sin",
    "metric_phase_cos",
    "strong_beat",
    "onset_gap_before",
    "rest_duration_before",
    "sustain_ratio",
    "duration_change",
    "top_voice_interval",
    "absolute_leap",
    "direction_change",
    "onset_density_log",
    "active_density_change",
    "pitch_range",
    "bass_motion",
    "tonal_novelty",
    "lbdm_strength",
]


@dataclass(frozen=True)
class ScoreCueResult:
    cues: np.ndarray
    quality: np.ndarray
    names: list[str]
    traversal_mode: str
    length_error_beats: float


def _safe_cosine_novelty(left: np.ndarray, right: np.ndarray) -> tuple[float, bool]:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0, False
    return float(np.clip(1.0 - np.dot(left, right) / denom, 0.0, 2.0)), True


def _relative_change(left: float, right: float) -> float:
    return float(abs(right - left) / max(abs(left) + abs(right), 1e-6))


def _events_from_dcml(piece: DcmlPiece, n_beats: int) -> tuple[list[tuple[float, float, int]], str, float]:
    notes = pd.read_csv(piece.notes_path, sep="\t")
    measures = pd.read_csv(piece.measures_path, sep="\t")
    traversal, mode, error = choose_measure_path(measures, n_beats)
    if not traversal:
        raise ValueError(f"{piece.piece_id}: score/beat length mismatch ({error:.3f})")
    folded_starts = {int(row.mc): to_float(row.quarterbeats) for row in measures.itertuples(index=False)}
    notes_by_mc = {int(mc): group for mc, group in notes.groupby(notes.mc.astype(int))}
    events: list[tuple[float, float, int]] = []
    for item in traversal:
        mc = int(item["mc"])
        group = notes_by_mc.get(mc)
        if group is None:
            continue
        for row in group.itertuples(index=False):
            onset = float(item["start_qb"]) + to_float(row.quarterbeats) - folded_starts[mc]
            duration = max(to_float(row.duration_qb), 0.0)
            if np.isfinite(onset) and np.isfinite(duration):
                events.append((onset, duration, int(row.midi)))
    return events, mode, float(error)


def build_compact_score_cues(piece: DcmlPiece, measure_number: np.ndarray, beat_number: np.ndarray) -> ScoreCueResult:
    """Build deployable beat-onset cues from notes/measures only.

    Cue row ``i`` describes the score state at beat onset ``i`` and the transition
    into it.  Backward differences use ``x[i]-x[i-1]``; row zero is neutral.
    No phrase, harmony, piece, opus, or split information is read.
    """

    n_beats = len(beat_number)
    events, mode, error = _events_from_dcml(piece, n_beats)
    onsets: list[list[tuple[float, float, int]]] = []
    active: list[list[tuple[float, float, int]]] = []
    for beat in range(n_beats):
        onsets.append([(o, d, m) for o, d, m in events if beat <= o < beat + 1])
        active.append([(o, d, m) for o, d, m in events if o < beat + 1 and o + d > beat])

    onset_count = np.asarray([len(x) for x in onsets], dtype=float)
    active_count = np.asarray([len(x) for x in active], dtype=float)
    mean_duration = np.asarray([np.mean([d for _, d, _ in x]) if x else np.nan for x in onsets], dtype=float)
    top = np.asarray([max((m for _, _, m in x), default=np.nan) for x in onsets], dtype=float)
    bass = np.asarray([min((m for _, _, m in x), default=np.nan) for x in onsets], dtype=float)
    pitch_range = np.asarray([(max(m for _, _, m in x) - min(m for _, _, m in x)) if x else 0.0 for x in onsets], dtype=float)
    pc = np.zeros((n_beats, 12), dtype=float)
    for index, group in enumerate(onsets):
        for _, _, midi in group:
            pc[index, midi % 12] += 1.0
        if pc[index].sum() > 0:
            pc[index] /= pc[index].sum()

    # The beat tables use zero-based positions after possible pickup beats.  The
    # maximum observed position provides a deployable meter cycle without assuming 3/4.
    finite_beats = beat_number[np.isfinite(beat_number)]
    meter = max(int(np.nanmax(finite_beats)) + 1 if len(finite_beats) else 1, 1)
    phase = np.mod(np.nan_to_num(beat_number, nan=0.0), meter) / meter

    latest_onset = np.nan
    latest_end = np.nan
    onset_gap = np.zeros(n_beats, dtype=float)
    rest_before = np.zeros(n_beats, dtype=float)
    for index in range(n_beats):
        prior_onsets = [o for o, _, _ in events if o < index + 1e-9]
        prior_ends = [o + d for o, d, _ in events if o + d <= index + 1e-9]
        if prior_onsets:
            latest_onset = max(prior_onsets)
            onset_gap[index] = min(max(index - latest_onset, 0.0), 8.0) / 8.0
        if prior_ends:
            latest_end = max(prior_ends)
            rest_before[index] = min(max(index - latest_end, 0.0), 8.0) / 8.0

    duration_filled = pd.Series(mean_duration).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    top_filled = pd.Series(top).interpolate(limit_direction="both").fillna(60.0).to_numpy()
    bass_filled = pd.Series(bass).interpolate(limit_direction="both").fillna(48.0).to_numpy()
    top_delta = np.diff(top_filled, prepend=top_filled[0])
    bass_delta = np.diff(bass_filled, prepend=bass_filled[0])
    direction = np.sign(top_delta)
    direction_change = np.zeros(n_beats, dtype=float)
    direction_change[1:] = ((direction[1:] != 0) & (direction[:-1] != 0) & (direction[1:] != direction[:-1])).astype(float)
    duration_change = np.abs(np.diff(np.log1p(duration_filled), prepend=np.log1p(duration_filled[0])))
    active_log = np.log1p(active_count)
    active_change = np.abs(np.diff(active_log, prepend=active_log[0]))
    novelty = np.zeros(n_beats, dtype=float)
    novelty_valid = np.zeros(n_beats, dtype=float)
    for index in range(1, n_beats):
        novelty[index], valid = _safe_cosine_novelty(pc[index - 1], pc[index])
        novelty_valid[index] = float(valid)

    pitch_interval_mag = np.abs(top_delta) / 12.0
    time_interval_mag = np.maximum(onset_gap, rest_before)
    pitch_change = np.zeros(n_beats, dtype=float)
    time_change = np.zeros(n_beats, dtype=float)
    for index in range(1, n_beats):
        pitch_change[index] = _relative_change(pitch_interval_mag[index - 1], pitch_interval_mag[index])
        time_change[index] = _relative_change(time_interval_mag[index - 1], time_interval_mag[index])
    lbdm = np.clip(0.25 * (pitch_change + time_change) + 0.5 * np.maximum(pitch_interval_mag, time_interval_mag), 0.0, 4.0)

    sustain = np.divide(np.maximum(active_count - onset_count, 0.0), np.maximum(active_count, 1.0))
    cues = np.column_stack(
        [
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
            (np.mod(np.nan_to_num(beat_number, nan=-1.0), meter) == 0).astype(float),
            onset_gap,
            rest_before,
            sustain,
            duration_change,
            np.clip(top_delta / 12.0, -4.0, 4.0),
            np.clip(np.abs(top_delta) / 12.0, 0.0, 4.0),
            direction_change,
            np.log1p(onset_count),
            active_change,
            np.clip(pitch_range / 24.0, 0.0, 4.0),
            np.clip(np.abs(bass_delta) / 12.0, 0.0, 4.0),
            novelty,
            lbdm,
        ]
    ).astype(np.float32)
    quality = np.ones_like(cues, dtype=np.float32)
    onset_valid = (onset_count > 0).astype(np.float32)
    quality[:, 6:10] = onset_valid[:, None]
    quality[:, 12:15] = np.column_stack([onset_valid, onset_valid, novelty_valid])
    quality[0, 6:10] = 0.0
    quality[0, 13:16] = 0.0
    if not np.isfinite(cues).all():
        raise FloatingPointError(f"Non-finite compact score cues for {piece.piece_id}")
    if cues.shape != (n_beats, len(SCORE_CUE_NAMES)):
        raise AssertionError("Compact score cue shape mismatch")
    return ScoreCueResult(cues, quality, list(SCORE_CUE_NAMES), mode, error)


def save_phase3_piece_cache(source_cache: Path, destination: Path, result: ScoreCueResult) -> dict[str, object]:
    with np.load(source_cache, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **payload,
        score_phase3=result.cues,
        score_phase3_quality=result.quality,
        score_phase3_names=np.asarray(result.names),
    )
    return {
        "piece_id": str(payload["piece_id"]),
        "beats": int(len(payload["labels"])),
        "performances": int(len(payload["performance_ids"])),
        "boundaries": int(payload["labels"].sum()),
        "score_cues": int(result.cues.shape[1]),
        "quality_valid_fraction": float(result.quality.mean()),
        "traversal_mode": result.traversal_mode,
        "length_error_beats": result.length_error_beats,
        "source_cache": str(source_cache),
        "destination_cache": str(destination),
    }
