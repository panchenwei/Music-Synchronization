from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import tomllib
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURVE_NAMES = [
    "log_tempo",
    "tempo_d1",
    "tempo_d2",
    "dynamics",
    "dynamics_d1",
    "tempo_robust_z",
    "dynamics_robust_z",
    "time_observed",
    "dynamics_observed",
]
SCORE_NAMES = [
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
TEMPO_SCALES = (1, 3, 6, 12, 24)
MULTISCALE_NAMES = [f"tempo_rmsq_s{scale}" for scale in TEMPO_SCALES] + [
    "tempo_rmsq_quality_s24",
    "tempo_lag_available",
    "tempo_direction_zero_control",
    "tempo_direction_available",
]
MOTIF_SCALES = (3, 6, 12)
MOTIF_NAMES = [
    f"{staff}_s{scale}_{kind}"
    for staff in ("staff_first", "staff_last")
    for scale in MOTIF_SCALES
    for kind in ("forward_repeat", "backward_repeat", "start_contrast", "available")
]
FEATURE_NAMES = CURVE_NAMES + SCORE_NAMES + MULTISCALE_NAMES + MOTIF_NAMES
assert len(FEATURE_NAMES) == 58


class AlignmentError(RuntimeError):
    """The score cannot be mapped without guessing."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return math.nan
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return float(text)


def _is_read(mode: Any) -> bool:
    if isinstance(mode, str):
        return not any(flag in mode for flag in "wax+")
    return True


class SourceAccessGuard:
    """Reject forbidden project/data reads and retain an auditable access set."""

    def __init__(self, dcml_root: Path, mazurka_root: Path, legacy_root: Path):
        self.dcml_root = dcml_root.resolve()
        self.mazurka_root = mazurka_root.resolve()
        self.legacy_root = legacy_root.resolve()
        self.seen: Counter[str] = Counter()
        self.roles: dict[str, str] = {}
        self._installed = False

    @staticmethod
    def _under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def validate_path(self, path: Path) -> str | None:
        path = path.resolve()
        if self._under(path, self.legacy_root):
            raise PermissionError(f"Legacy project access is forbidden in Goal 1: {path}")
        if self._under(path, self.dcml_root):
            relative = path.relative_to(self.dcml_root)
            allowed = len(relative.parts) == 2 and relative.parts[0] in {"notes", "measures"} and path.suffix == ".tsv"
            if not allowed:
                raise PermissionError(f"Only raw DCML notes/measures are allowed: {path}")
            return "dcml_notes" if relative.parts[0] == "notes" else "dcml_measures"
        if self._under(path, self.mazurka_root):
            relative = path.relative_to(self.mazurka_root)
            allowed = len(relative.parts) == 2 and relative.parts[0] in {"beat_time", "beat_dyn"} and path.suffix == ".csv"
            if not allowed:
                raise PermissionError(f"Only MazurkaBL beat_time/beat_dyn are allowed: {path}")
            return "mazurkabl_beat_time" if relative.parts[0] == "beat_time" else "mazurkabl_beat_dyn"
        return None

    def install(self) -> None:
        if self._installed:
            return

        def audit(event: str, args: tuple[Any, ...]) -> None:
            if event != "open" or not args or not isinstance(args[0], (str, bytes)) or not _is_read(args[1] if len(args) > 1 else "r"):
                return
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
            try:
                role = self.validate_path(path)
            except (OSError, ValueError):
                return
            if role:
                key = str(path.resolve())
                self.seen[key] += 1
                self.roles[key] = role

        sys.addaudithook(audit)
        self._installed = True

    def access_frame(self) -> pd.DataFrame:
        rows = []
        for value, count in sorted(self.seen.items()):
            path = Path(value)
            rows.append(
                {
                    "path": value,
                    "role": self.roles[value],
                    "open_events": count,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        return pd.DataFrame(rows, columns=["path", "role", "open_events", "bytes", "sha256"])


@dataclass(frozen=True)
class PieceSources:
    piece_id: str
    code: str
    notes: Path
    measures: Path
    beat_time: Path
    beat_dyn: Path


def read_frame(path: Path, separator: str, guard: SourceAccessGuard) -> pd.DataFrame:
    guard.validate_path(path)
    frame = pd.read_csv(path, sep=separator)
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]


def _score_key(path: Path) -> tuple[int, int] | None:
    match = re.search(r"op0*(\d+)-(\d+)", path.stem, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _mazurka_key(path: Path) -> tuple[int, int] | None:
    match = re.match(r"M0*(\d+)-(\d+)beat_(?:time|dyn)", path.stem, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def discover_sources(dcml_root: Path, mazurka_root: Path) -> tuple[list[PieceSources], list[dict[str, Any]]]:
    notes = {_score_key(path): path for path in (dcml_root / "notes").glob("*.notes.tsv") if _score_key(path)}
    measures = {_score_key(path): path for path in (dcml_root / "measures").glob("*.measures.tsv") if _score_key(path)}
    beat_time = {_mazurka_key(path): path for path in (mazurka_root / "beat_time").glob("*beat_time.csv") if _mazurka_key(path)}
    beat_dyn = {_mazurka_key(path): path for path in (mazurka_root / "beat_dyn").glob("*beat_dyn*.csv") if _mazurka_key(path)}
    all_keys = sorted(set(notes) | set(measures) | set(beat_time) | set(beat_dyn))
    sources, inventory = [], []
    for opus, number in all_keys:
        key = (opus, number)
        present = {"notes": key in notes, "measures": key in measures, "beat_time": key in beat_time, "beat_dyn": key in beat_dyn}
        piece_id = f"chopin_op{opus:02d}_no{number}"
        if all(present.values()):
            sources.append(PieceSources(piece_id, f"M{opus:02d}-{number}", notes[key], measures[key], beat_time[key], beat_dyn[key]))
            status = "candidate"
        else:
            status = "unmatched"
        inventory.append({"piece_id": piece_id, "opus": opus, "number": number, **present, "status": status})
    return sources, inventory


def _next_candidates(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [int(token) for token in re.findall(r"-?\d+", str(value))]


def unfold_measure_path(measures: pd.DataFrame, maximum_steps: int = 10000) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = {"mc", "duration_qb", "next"}
    if not required.issubset(measures.columns):
        raise AlignmentError(f"Missing measure columns: {sorted(required - set(measures.columns))}")
    if measures.mc.duplicated().any():
        raise AlignmentError("Measure mc values are not unique")
    table = {int(row.mc): row for row in measures.itertuples(index=False)}
    if not table:
        raise AlignmentError("No measures")
    current, unfolded_qb = min(table), 0.0
    visits: Counter[int] = Counter()
    path, decisions = [], []
    for sequence_index in range(maximum_steps):
        if current not in table:
            raise AlignmentError(f"Measure graph points to missing mc={current}")
        row = table[current]
        visits[current] += 1
        duration = to_float(row.duration_qb)
        if not np.isfinite(duration) or duration <= 0:
            raise AlignmentError(f"Invalid duration_qb at mc={current}: {row.duration_qb}")
        candidates = _next_candidates(row.next)
        visit = visits[current]
        if len(candidates) > 1 and visit > len(candidates):
            raise AlignmentError(f"Cannot uniquely select next edge at mc={current}, visit={visit}, candidates={candidates}")
        selected = candidates[min(visit - 1, len(candidates) - 1)] if candidates else None
        path.append(
            {
                "sequence_index": sequence_index,
                "mc": current,
                "visit": visit,
                "start_qb": unfolded_qb,
                "duration_qb": duration,
                "next_mc": selected,
                "next_candidates": ",".join(map(str, candidates)),
            }
        )
        if len(candidates) > 1:
            decisions.append({"mc": current, "visit": visit, "candidates": candidates, "selected": selected})
        unfolded_qb += duration
        if selected is None or selected == -1:
            break
        current = selected
    else:
        raise AlignmentError(f"Measure traversal exceeded {maximum_steps} steps")
    return path, decisions


def length_error(path: list[dict[str, Any]], beats: int) -> tuple[float, float]:
    score_length = float(sum(row["duration_qb"] for row in path))
    error = min(abs(score_length - beats), abs(score_length - (beats - 1)))
    return score_length, error


def positioned_events(notes: pd.DataFrame, path: list[dict[str, Any]], beats: int) -> tuple[list[tuple[float, float, int, int, int]], list[int | None], pd.DataFrame]:
    required = {"mc", "mc_onset", "duration_qb", "midi", "staff", "voice"}
    if not required.issubset(notes.columns):
        raise AlignmentError(f"Missing note columns: {sorted(required - set(notes.columns))}")
    parsed_onsets = notes.mc_onset.map(to_float)
    if not np.isfinite(parsed_onsets).all():
        rows = notes.index[~np.isfinite(parsed_onsets)].tolist()[:10]
        raise AlignmentError(f"Missing/non-finite mc_onset rows: {rows}")
    groups = {int(mc): group for mc, group in notes.groupby(notes.mc.astype(int))}
    events: list[tuple[float, float, int, int, int]] = []
    ties: list[int | None] = []
    mapping: list[dict[str, Any]] = []
    for occurrence in path:
        group = groups.get(int(occurrence["mc"]))
        if group is None:
            continue
        for source_index, row in group.iterrows():
            local_qb = 4.0 * to_float(row.mc_onset)
            if local_qb < -1e-7 or local_qb > float(occurrence["duration_qb"]) + 1e-7:
                raise AlignmentError(
                    f"mc_onset outside measure at row={source_index}, mc={occurrence['mc']}, local_qb={local_qb}"
                )
            onset = float(occurrence["start_qb"]) + local_qb
            duration = to_float(row.duration_qb)
            if not np.isfinite(duration) or duration < 0:
                raise AlignmentError(f"Invalid note duration at row={source_index}: {row.duration_qb}")
            tie_value = row.get("tied", None)
            tie = None if pd.isna(tie_value) or str(tie_value).strip() == "" else int(float(tie_value))
            if tie not in {None, -1, 0, 1}:
                raise AlignmentError(f"Invalid tied value at row={source_index}: {tie_value}")
            event = (onset, duration, int(row.midi), int(row.staff), int(row.voice))
            if onset < beats and onset + duration > 0:
                event_index = len(events)
                events.append(event)
                ties.append(tie)
                status = "mapped"
            else:
                event_index = None
                status = "outside_beat_grid"
            mapping.append(
                {
                    "source_row": int(source_index),
                    "mc": int(occurrence["mc"]),
                    "visit": int(occurrence["visit"]),
                    "occurrence_start_qb": float(occurrence["start_qb"]),
                    "mc_onset": str(row.mc_onset),
                    "local_qb": local_qb,
                    "global_qb": onset,
                    "beat_index": int(math.floor(onset)) if 0 <= onset < beats else None,
                    "subtick": int(math.floor(onset * 4 + 1e-8)) % 4 if 0 <= onset < beats else None,
                    "duration_qb": duration,
                    "midi": int(row.midi),
                    "staff": int(row.staff),
                    "voice": int(row.voice),
                    "tie": tie,
                    "event_index": event_index,
                    "status": status,
                }
            )
    if not events:
        raise AlignmentError("No note events overlap the beat grid")
    return events, ties, pd.DataFrame(mapping)


def _predecessors(events: list[tuple[float, float, int, int, int]], ties: list[int | None], index: int) -> list[int]:
    q, _, pitch, staff, voice = events[index]
    pool = [
        j
        for j, (event, tie) in enumerate(zip(events, ties))
        if tie in {0, 1} and event[1] > 0 and event[2] == pitch and abs(event[0] + event[1] - q) < 1e-7 and event[0] < q
    ]
    for candidates in (
        [j for j in pool if events[j][3:] == (staff, voice)],
        [j for j in pool if events[j][3] == staff],
        pool,
    ):
        if candidates:
            return candidates
    return []


def merge_unambiguous_ties(events: list[tuple[float, float, int, int, int]], ties: list[int | None]) -> tuple[list[tuple[float, float, int, int, int, float]], dict[str, int]]:
    links, ambiguous, orphan = {}, 0, 0
    for index, tie in enumerate(ties):
        if tie not in {0, -1}:
            continue
        candidates = _predecessors(events, ties, index)
        if len(candidates) == 1:
            links[index] = candidates[0]
        elif len(candidates) > 1:
            ambiguous += 1
        else:
            orphan += 1
    counts = Counter(links.values())
    links = {child: parent for child, parent in links.items() if counts[parent] == 1}
    groups: dict[int, list[int]] = {}
    for index, event in enumerate(events):
        if event[1] <= 0:
            continue
        root = index
        visited = set()
        while root in links:
            if root in visited:
                raise AlignmentError("Tie cycle detected")
            visited.add(root)
            root = links[root]
        groups.setdefault(root, []).append(index)
    rows = []
    for root, indices in groups.items():
        onset, _, pitch, staff, voice = events[root]
        end = max(events[index][0] + events[index][1] for index in indices)
        rows.append((onset, end - onset, pitch, staff, voice, float(ties[root] in {0, -1})))
    return sorted(rows), {"connected": len(links), "ambiguous": ambiguous, "orphan": orphan}


def _rolling_robust_z(values: np.ndarray, radius: int = 4) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for index in range(len(values)):
        local = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        median = np.nanmedian(local)
        mad = np.nanmedian(np.abs(local - median))
        output[index] = (values[index] - median) / max(1.4826 * mad, 1e-4)
    return np.clip(output, -8.0, 8.0)


def _interpolate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values, dtype=np.float32), valid.astype(np.float32)
    return pd.Series(values).interpolate(limit_direction="both").to_numpy(np.float32), valid.astype(np.float32)


def build_curve_features(times: np.ndarray, dynamics: np.ndarray) -> tuple[np.ndarray, int]:
    times, time_mask = _interpolate(np.asarray(times, float))
    dynamics, dynamics_mask = _interpolate(np.asarray(dynamics, float))
    interval = np.diff(times, append=np.nan)
    positive = interval[np.isfinite(interval) & (interval > 0)]
    fallback = float(np.median(positive)) if len(positive) else 0.6
    interval[-1] = interval[-2] if len(interval) > 1 and interval[-2] > 0 else fallback
    bad = ~np.isfinite(interval) | (interval <= 0)
    bad_count = int(bad.sum())
    interval[bad] = fallback
    log_tempo = np.log(np.clip(60.0 / interval, 20.0, 400.0)).astype(np.float32)
    d1 = np.diff(log_tempo, prepend=log_tempo[0]).astype(np.float32)
    d2 = np.diff(d1, prepend=d1[0]).astype(np.float32)
    dynamics_d1 = np.diff(dynamics, prepend=dynamics[0]).astype(np.float32)
    features = np.column_stack(
        [
            log_tempo,
            d1,
            d2,
            dynamics,
            dynamics_d1,
            _rolling_robust_z(log_tempo),
            _rolling_robust_z(dynamics),
            time_mask,
            dynamics_mask,
        ]
    ).astype(np.float32)
    if features.shape[1] != 9 or not np.isfinite(features).all():
        raise AlignmentError("Invalid performance curve features")
    return features, bad_count


def align_dynamics(beat_frame: pd.DataFrame, dynamics_frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["measure_number", "beat_number"]
    if not set(keys).issubset(beat_frame.columns) or not set(keys).issubset(dynamics_frame.columns):
        raise AlignmentError("beat_time/beat_dyn missing measure_number or beat_number")
    if beat_frame.duplicated(keys).any() or dynamics_frame.duplicated(keys).any():
        raise AlignmentError("beat_time/beat_dyn coordinates are not one-to-one")
    left = beat_frame[keys].copy()
    left["__order"] = np.arange(len(left))
    return left.merge(dynamics_frame, on=keys, how="left", validate="one_to_one", sort=False).sort_values("__order").drop(columns="__order")


def build_performance_curves(beat_frame: pd.DataFrame, dynamics_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    aligned = align_dynamics(beat_frame, dynamics_frame)
    reserved = {"measure_number", "beat_number"}
    performance_ids = [column for column in beat_frame.columns if column not in reserved]
    curves, kept, invalid_intervals = [], [], 0
    for performance_id in performance_ids:
        times = pd.to_numeric(beat_frame[performance_id], errors="coerce").to_numpy(float)
        if np.isfinite(times).mean() < 0.8:
            continue
        dynamics = pd.to_numeric(aligned[performance_id], errors="coerce").to_numpy(float) if performance_id in aligned else np.full(len(times), np.nan)
        curve, bad = build_curve_features(times, dynamics)
        curves.append(curve)
        kept.append(performance_id)
        invalid_intervals += bad
    if not curves:
        raise AlignmentError("No performance has at least 80% finite beat times")
    return np.stack(curves), np.asarray(kept), {"nonpositive_or_missing_intervals_replaced": invalid_intervals}


def tempo_multiscale(curves: np.ndarray) -> np.ndarray:
    result = np.zeros((*curves.shape[:2], 9), dtype=np.float32)
    for performance, curve in enumerate(curves):
        tempo = np.exp(curve[:, 0].astype(np.float64))
        reference = float(np.median(tempo))
        levels = []
        quality = np.zeros(len(tempo), dtype=np.float32)
        for scale in TEMPO_SCALES:
            level = np.zeros(len(tempo), dtype=np.float64)
            for start in range(0, len(tempo), scale):
                stop = min(start + scale, len(tempo))
                block = tempo[start:stop] / reference
                energy = np.sqrt(np.mean(block**2)) - np.std(block, ddof=0)
                level[start:stop] = np.log(max(float(energy), 1e-8))
                if scale == TEMPO_SCALES[-1]:
                    quality[start:stop] = curve[start:stop, 7].mean()
            levels.append(level.astype(np.float32))
        observed = curve[:, 7] > 0.5
        interval_observed = np.zeros(len(curve), bool)
        interval_observed[:-1] = observed[:-1] & observed[1:]
        lag_available = np.zeros(len(curve), np.float32)
        lag_available[1:] = interval_observed[:-1]
        direction_available = np.zeros(len(curve), np.float32)
        for beat in range(1, len(curve)):
            indices = np.r_[np.arange(max(0, beat - 3), beat), np.arange(beat, min(len(curve), beat + 3))]
            if len(indices):
                direction_available[beat] = interval_observed[indices].mean()
        result[performance] = np.column_stack(levels + [quality, lag_available, np.zeros(len(curve)), direction_available])
    if not np.isfinite(result).all():
        raise AlignmentError("Non-finite multiscale tempo features")
    return result


def _relative_change(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left) + abs(right), 1e-6)


def _voice_track(events: list[tuple[float, float, int, int, int]], beats: int, voice: tuple[int, int], lowest: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [event for event in events if event[3:5] == voice and event[1] > 0]
    delta, duration_change, direction_change = np.zeros(beats), np.zeros(beats), np.zeros(beats)
    previous_pitch = previous_duration = previous_sign = None
    for beat in range(beats):
        attacks = [event for event in selected if beat <= event[0] < beat + 1]
        if not attacks:
            continue
        first = min(event[0] for event in attacks)
        chord = [event for event in attacks if abs(event[0] - first) < 1e-8]
        note = (min if lowest else max)(chord, key=lambda event: event[2])
        pitch, duration = note[2], note[1]
        if previous_pitch is not None:
            delta[beat] = pitch - previous_pitch
            sign = np.sign(delta[beat])
            direction_change[beat] = float(sign != 0 and previous_sign not in {None, 0} and sign != previous_sign)
            if sign != 0:
                previous_sign = sign
            duration_change[beat] = abs(np.log1p(duration) - np.log1p(previous_duration))
        previous_pitch, previous_duration = pitch, duration
    return delta, duration_change, direction_change


def build_score_cues(events: list[tuple[float, float, int, int, int]], beat_number: np.ndarray, beats: int) -> np.ndarray:
    onsets = [[event for event in events if beat <= event[0] < beat + 1] for beat in range(beats)]
    active = [[event for event in events if event[0] < beat + 1 and event[0] + event[1] > beat] for beat in range(beats)]
    onset_count = np.asarray([len(group) for group in onsets], float)
    active_count = np.asarray([len(group) for group in active], float)
    mean_duration = np.asarray([np.mean([event[1] for event in group]) if group else np.nan for group in onsets])
    pitch_range = np.asarray([(max(event[2] for event in group) - min(event[2] for event in group)) if group else 0.0 for group in onsets])
    pitch_class = np.zeros((beats, 12), float)
    for index, group in enumerate(onsets):
        for event in group:
            pitch_class[index, event[2] % 12] += 1
        if pitch_class[index].sum():
            pitch_class[index] /= pitch_class[index].sum()
    finite_beats = beat_number[np.isfinite(beat_number)]
    meter = max(int(np.nanmax(finite_beats)) + 1 if len(finite_beats) else 1, 1)
    phase = np.mod(np.nan_to_num(beat_number, nan=0.0), meter) / meter
    onset_gap, rest_before = np.zeros(beats), np.zeros(beats)
    for beat in range(beats):
        prior_onsets = [event[0] for event in events if event[0] <= beat + 1e-9]
        if prior_onsets:
            onset_gap[beat] = min(max(beat - max(prior_onsets), 0.0), 8.0) / 8.0
        if any(onset < beat and onset + duration >= beat - 1e-9 and duration > 0 for onset, duration, *_ in events):
            continue
        prior_ends = [onset + duration for onset, duration, *_ in events if duration > 0 and onset + duration <= beat]
        if prior_ends:
            rest_before[beat] = min(max(beat - max(prior_ends), 0.0), 8.0) / 8.0
    duration_filled = pd.Series(mean_duration).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    active_log = np.log1p(active_count)
    active_change = np.abs(np.diff(active_log, prepend=active_log[0]))
    novelty = np.zeros(beats)
    for index in range(1, beats):
        denominator = np.linalg.norm(pitch_class[index - 1]) * np.linalg.norm(pitch_class[index])
        if denominator > 1e-12:
            novelty[index] = np.clip(1.0 - np.dot(pitch_class[index - 1], pitch_class[index]) / denominator, 0.0, 2.0)
    voices = sorted(set(event[3:5] for event in events))
    if not voices:
        raise AlignmentError("No score voices")
    melody = (1, 1) if (1, 1) in voices else voices[0]
    bottom_staff = max(voice[0] for voice in voices)
    bass = max([voice for voice in voices if voice[0] == bottom_staff], key=lambda voice: sum(event[3:5] == voice for event in events))
    top_delta, duration_change, direction_change = _voice_track(events, beats, melody)
    bass_delta, _, _ = _voice_track(events, beats, bass, lowest=True)
    pitch_interval = np.abs(top_delta) / 12.0
    timing_interval = np.maximum(onset_gap, rest_before)
    pitch_change, time_change = np.zeros(beats), np.zeros(beats)
    for index in range(1, beats):
        pitch_change[index] = _relative_change(pitch_interval[index - 1], pitch_interval[index])
        time_change[index] = _relative_change(timing_interval[index - 1], timing_interval[index])
    lbdm = np.clip(0.25 * (pitch_change + time_change) + 0.5 * np.maximum(pitch_interval, timing_interval), 0.0, 4.0)
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
    if cues.shape != (beats, 16) or not np.isfinite(cues).all():
        raise AlignmentError("Invalid score cue matrix")
    return cues


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def staff_sequences(events: Iterable[tuple[float, float, int, int, int, float]], beats: int) -> np.ndarray:
    events = list(events)
    staffs = sorted(set(float(event[3]) for event in events))
    if not staffs:
        raise AlignmentError("No staff events")
    result = np.zeros((2, beats, 4, 24), np.float64)
    for stream, staff in enumerate((staffs[0], staffs[-1])):
        for onset, duration, pitch, event_staff, *_ in events:
            if event_staff != staff or duration <= 0:
                continue
            pitch_class = int(pitch) % 12
            tick = int(np.floor(onset * 4 + 1e-8))
            if 0 <= tick < beats * 4:
                result[stream, tick // 4, tick % 4, pitch_class] = 1.0
            for occupied in range(max(0, int(np.floor(onset * 4))), min(beats * 4, int(np.ceil((onset + duration) * 4)))):
                coverage = max(0.0, min((occupied + 1) / 4, onset + duration) - max(occupied / 4, onset)) * 4
                result[stream, occupied // 4, occupied % 4, 12 + pitch_class] = max(
                    result[stream, occupied // 4, occupied % 4, 12 + pitch_class], coverage
                )
    return _normalize_rows(result.reshape(2, beats, 96))


def _window_similarity(values: np.ndarray, width: int) -> np.ndarray:
    count = len(values) - width + 1
    if count <= 0:
        return np.empty((0, 0))
    similarity = values @ values.T
    output = np.zeros((count, count))
    for offset in range(width):
        output += similarity[offset : offset + count, offset : offset + count]
    return np.clip(output / width, 0.0, 1.0)


def motif_recurrence(events: Iterable[tuple[float, float, int, int, int, float]], beats: int) -> tuple[np.ndarray, np.ndarray]:
    sequences = staff_sequences(events, beats)
    output = []
    for values in sequences:
        stream = np.zeros((beats, len(MOTIF_SCALES) * 4), np.float32)
        for scale_index, width in enumerate(MOTIF_SCALES):
            similarity = _window_similarity(values, width)
            positions = np.arange(width, beats - width + 1)
            for beat in positions:
                candidates = positions[np.abs(positions - beat) >= 2 * width]
                if not len(candidates) or beat >= beats:
                    continue
                forward = similarity[beat, candidates]
                backward = similarity[beat - width, candidates - width]
                stream[beat, scale_index * 4 : scale_index * 4 + 4] = [
                    forward.max(),
                    backward.max(),
                    np.max(forward * (1.0 - backward)),
                    1.0,
                ]
        output.append(stream)
    features = np.concatenate(output, axis=1)
    if features.shape != (beats, 24) or not np.isfinite(features).all() or features.min() < 0 or features.max() > 1:
        raise AlignmentError("Invalid motif recurrence features")
    return features, sequences.reshape(2, beats, 4, 24)


def window_starts(length: int, window: int = 64, stride: int = 32) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    next_start = starts[-1] + stride
    if next_start < length:
        starts.append(next_start)
    return starts


def feature_schema() -> pd.DataFrame:
    rows = []
    groups = [(CURVE_NAMES, "performance_curve", "one beat"), (SCORE_NAMES, "score_cue", "one beat plus past transition"), (MULTISCALE_NAMES, "tempo_multiscale", "1/3/6/12/24-beat blocks"), (MOTIF_NAMES, "ordered_motif", "3/6/12-beat windows")]
    offset = 0
    for names, group, support in groups:
        for name in names:
            rows.append({"index": offset, "name": name, "group": group, "time_support": support, "coordinate": "quarterbeat row i represents [i,i+1)", "missing_rule": "explicit observed/available columns; numeric values finite"})
            offset += 1
    return pd.DataFrame(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_piece(source: PieceSources, guard: SourceAccessGuard, cache_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    beat_frame = read_frame(source.beat_time, ",", guard)
    dynamics_frame = read_frame(source.beat_dyn, ",", guard)
    measures = read_frame(source.measures, "\t", guard)
    notes = read_frame(source.notes, "\t", guard)
    beats = len(beat_frame)
    path, decisions = unfold_measure_path(measures)
    score_length, error = length_error(path, beats)
    if error > 2.0:
        raise AlignmentError(f"Score/beat length mismatch: score={score_length}, beat_rows={beats}, error={error}")
    events, ties, event_mapping = positioned_events(notes, path, beats)
    merged, tie_audit = merge_unambiguous_ties(events, ties)
    beat_number = pd.to_numeric(beat_frame.beat_number, errors="coerce").to_numpy(float)
    measure_number = pd.to_numeric(beat_frame.measure_number, errors="coerce").to_numpy(float)
    if not np.isfinite(beat_number).all() or not np.isfinite(measure_number).all():
        raise AlignmentError("Non-finite MazurkaBL beat coordinates")
    curves, performance_ids, curve_audit = build_performance_curves(beat_frame, dynamics_frame)
    score = build_score_cues(events, beat_number, beats)
    multiscale = tempo_multiscale(curves)
    motif, subticks = motif_recurrence(merged, beats)
    score_broadcast = np.broadcast_to(score, (*curves.shape[:2], 16))
    motif_broadcast = np.broadcast_to(motif, (*curves.shape[:2], 24))
    inputs = np.concatenate([curves, score_broadcast, multiscale, motif_broadcast], axis=-1).astype(np.float32)
    if inputs.shape != (len(performance_ids), beats, 58) or not np.isfinite(inputs).all():
        raise AlignmentError(f"Invalid 58-D input shape: {inputs.shape}")
    path_frame = pd.DataFrame(path)
    path_frame.insert(0, "piece_id", source.piece_id)
    event_mapping.insert(0, "piece_id", source.piece_id)
    cache_path = cache_dir / f"{source.piece_id}.npz"
    np.savez_compressed(
        cache_path,
        piece_id=np.asarray(source.piece_id),
        performance_ids=performance_ids,
        quarterbeat=np.arange(beats, dtype=np.float32),
        measure_number=measure_number.astype(np.float32),
        beat_number=beat_number.astype(np.float32),
        curves=curves,
        score_cues=score,
        tempo_multiscale=multiscale,
        motif_recurrence=motif,
        motif_subticks=subticks.astype(np.float32),
        inputs=inputs,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    durations = pd.to_numeric(measures.duration_qb, errors="coerce").dropna().to_numpy()
    first_duration = float(to_float(measures.iloc[0].duration_qb))
    typical_duration = float(np.median(durations)) if len(durations) else first_duration
    selected_by_mc: dict[int, set[int | None]] = {}
    for decision in decisions:
        selected_by_mc.setdefault(int(decision["mc"]), set()).add(decision["selected"])
    has_alternative = any(len(values) > 1 for values in selected_by_mc.values())
    audit = {
        "piece_id": source.piece_id,
        "status": "usable",
        "beats": beats,
        "performances": len(performance_ids),
        "input_dimensions": 58,
        "score_length_qb": score_length,
        "length_error_beats": error,
        "measure_occurrences": len(path),
        "repeated_measure_occurrences": len(path) - len({row["mc"] for row in path}),
        "branch_decisions": len(decisions),
        "has_pickup": first_duration < typical_duration - 1e-7,
        "has_repeat": len(path) > len({row["mc"] for row in path}),
        "has_alternative_ending": has_alternative,
        "mapped_events": len(events),
        "outside_grid_events": int((event_mapping.status == "outside_beat_grid").sum()),
        "tie_connected": tie_audit["connected"],
        "tie_ambiguous": tie_audit["ambiguous"],
        "tie_orphan": tie_audit["orphan"],
        **curve_audit,
        "cache_path": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "coordinate_invariant": "all four feature groups have identical beat axis [0,n)",
    }
    visual = {"path": path, "events": merged, "inputs": inputs[0], "subticks": subticks, "audit": audit}
    return audit, path_frame, event_mapping, visual


def plot_alignment(piece_id: str, payload: dict[str, Any], destination: Path) -> None:
    audit, path, events, values = payload["audit"], payload["path"], payload["events"], payload["inputs"]
    figure, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1.0, 2.2, 1.5]})
    seen: Counter[int] = Counter()
    for row in path:
        seen[int(row["mc"])] += 1
        color = "#f59e0b" if seen[int(row["mc"])] > 1 else "#3b82f6"
        axes[0].barh(0, row["duration_qb"], left=row["start_qb"], height=0.55, color=color, edgecolor="white")
    axes[0].set_yticks([])
    axes[0].set_ylabel("measure\npath")
    axes[0].set_title(
        f"{piece_id} | pickup={audit['has_pickup']} repeat={audit['has_repeat']} alternative={audit['has_alternative_ending']}"
    )
    if events:
        axes[1].scatter([event[0] for event in events], [event[2] for event in events], s=4, c=[event[3] for event in events], cmap="viridis")
    axes[1].set_ylabel("MIDI pitch")
    axes[1].grid(axis="x", color="#e5e7eb", linewidth=0.4)
    group_slices = {"curves 9": slice(0, 9), "score 16": slice(9, 25), "tempo-scale 9": slice(25, 34), "motif 24": slice(34, 58)}
    for label, group in group_slices.items():
        signal = np.mean(np.abs(values[:, group]), axis=1)
        maximum = max(float(signal.max()), 1e-8)
        axes[2].plot(np.arange(len(signal)), signal / maximum, label=label, linewidth=1.2)
    axes[2].set(xlabel="unfolded quarterbeat / beat row", ylabel="group magnitude\n(display-normalized)")
    axes[2].legend(ncol=4, fontsize=8, loc="upper right")
    axes[2].set_ylim(-0.05, 1.15)
    for axis in axes:
        axis.set_facecolor("white")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def run(project_root: Path, config_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    dcml_root = Path(config["raw"]["dcml_chopin"])
    mazurka_root = Path(config["raw"]["mazurkabl"])
    legacy_root = Path(config["legacy_read_only"]["experiment_root"])
    guard = SourceAccessGuard(dcml_root, mazurka_root, legacy_root)
    guard.install()
    cache_dir = project_root / "cache" / "unlabeled58"
    manifest_dir = project_root / "manifests" / "goal1"
    report_dir = project_root / "reports" / "goal1"
    visual_dir = report_dir / "visualizations"
    log_dir = project_root / "logs" / "goal1"
    for path in (cache_dir, manifest_dir, visual_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    sources, inventory = discover_sources(dcml_root, mazurka_root)
    piece_rows, measure_frames, event_frames, visuals, failures = [], [], [], {}, []
    for source in sources:
        try:
            audit, measure_mapping, event_mapping, visual = process_piece(source, guard, cache_dir)
            piece_rows.append(audit)
            measure_frames.append(measure_mapping)
            event_frames.append(event_mapping)
            visuals[source.piece_id] = visual
            print(f"USABLE {source.piece_id}: {audit['performances']} performances x {audit['beats']} beats x 58", flush=True)
        except Exception as error:
            record = {"piece_id": source.piece_id, "status": "excluded", "reason": str(error), "error_type": type(error).__name__}
            piece_rows.append(record)
            failures.append(record)
            print(f"EXCLUDED {source.piece_id}: {error}", flush=True)
    piece_audit = pd.DataFrame(piece_rows).sort_values("piece_id")
    pd.DataFrame(inventory).to_csv(manifest_dir / "source_inventory.csv", index=False)
    piece_audit.to_csv(manifest_dir / "piece_audit.csv", index=False)
    pd.concat(measure_frames, ignore_index=True).to_csv(manifest_dir / "measure_mapping.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(manifest_dir / "event_mapping.csv.gz", index=False)
    feature_schema().to_csv(manifest_dir / "feature_schema.csv", index=False)
    usable = piece_audit[piece_audit.status == "usable"]
    windows = []
    full_sequences = []
    for row in usable.itertuples(index=False):
        with np.load(row.cache_path, allow_pickle=False) as cache:
            performances = [str(value) for value in cache["performance_ids"]]
            beats = len(cache["quarterbeat"])
        for performance_id in performances:
            full_sequences.append(
                {
                    "piece_id": row.piece_id,
                    "performance_id": performance_id,
                    "cache_path": row.cache_path,
                    "start_beat": 0,
                    "stop_beat_exclusive": beats,
                    "valid_length": beats,
                    "sequence_mode": "full_piece_variable_length",
                    "batch_padding": "pad_to_longest_sequence_in_batch",
                    "requires_padding_mask": True,
                }
            )
            for start in window_starts(beats, 64, 32):
                valid = min(64, beats - start)
                windows.append(
                    {
                        "piece_id": row.piece_id,
                        "performance_id": performance_id,
                        "cache_path": row.cache_path,
                        "start_beat": start,
                        "stop_beat_exclusive": start + valid,
                        "valid_length": valid,
                        "window_length": 64,
                        "stride": 32,
                        "padded": valid < 64,
                    }
                )
    window_frame = pd.DataFrame(windows)
    window_frame.to_csv(manifest_dir / "window_manifest.csv.gz", index=False)
    full_sequence_frame = pd.DataFrame(full_sequences)
    full_sequence_frame.to_csv(manifest_dir / "full_sequence_manifest.csv.gz", index=False)
    preferred = [piece_id for piece_id, payload in visuals.items() if all(payload["audit"][flag] for flag in ("has_pickup", "has_repeat", "has_alternative_ending"))]
    if len(preferred) < 3:
        preferred = sorted(visuals, key=lambda piece_id: sum(bool(visuals[piece_id]["audit"][flag]) for flag in ("has_pickup", "has_repeat", "has_alternative_ending")), reverse=True)
    visualized = preferred[:3]
    for piece_id in visualized:
        plot_alignment(piece_id, visuals[piece_id], visual_dir / f"{piece_id}_alignment.png")
    access = guard.access_frame()
    access.to_csv(manifest_dir / "source_access.csv", index=False)
    failures_path = log_dir / "failures.jsonl"
    failures_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")
    cache_checks = []
    for row in usable.itertuples(index=False):
        with np.load(row.cache_path, allow_pickle=False) as cache:
            keys = set(cache.files)
            shapes = {
                "curves": list(cache["curves"].shape),
                "score_cues": list(cache["score_cues"].shape),
                "tempo_multiscale": list(cache["tempo_multiscale"].shape),
                "motif_recurrence": list(cache["motif_recurrence"].shape),
                "inputs": list(cache["inputs"].shape),
            }
            cache_checks.append(
                {
                    "piece_id": row.piece_id,
                    "no_label_fields": not any("label" in key.lower() for key in keys),
                    "shape_ok": shapes["inputs"] == [row.performances, row.beats, 58]
                    and shapes["curves"] == [row.performances, row.beats, 9]
                    and shapes["score_cues"] == [row.beats, 16]
                    and shapes["tempo_multiscale"] == [row.performances, row.beats, 9]
                    and shapes["motif_recurrence"] == [row.beats, 24],
                    "finite": bool(all(np.isfinite(cache[key]).all() for key in ("curves", "score_cues", "tempo_multiscale", "motif_recurrence", "inputs"))),
                }
            )
    cache_check_frame = pd.DataFrame(cache_checks)
    cache_check_frame.to_csv(report_dir / "cache_checks.csv", index=False)
    forbidden = access[~access.role.isin({"dcml_notes", "dcml_measures", "mazurkabl_beat_time", "mazurkabl_beat_dyn"})]
    completion = {
        "status": "complete" if len(usable) and len(visualized) >= 3 and cache_check_frame[["no_label_fields", "shape_ok", "finite"]].all().all() and forbidden.empty else "incomplete",
        "scientific_status": "implemented_and_locally_reproduced_data_pipeline",
        "candidate_pieces": len(sources),
        "usable_pieces": len(usable),
        "excluded_pieces": len(piece_audit) - len(usable),
        "performances": int(usable.performances.sum()),
        "windows": len(window_frame),
        "full_piece_sequences": len(full_sequence_frame),
        "sequence_protocols": ["fixed_64_stride_32", "full_piece_variable_length"],
        "feature_dimensions": 58,
        "visualizations": visualized,
        "forbidden_source_accesses": len(forbidden),
        "model_training_runs": 0,
        "length_compatibility_rule": "min(|score_qb-beat_rows|, |score_qb-(beat_rows-1)|) <= 2",
        "seconds": time.monotonic() - started,
    }
    write_json(report_dir / "completion_audit.json", completion)
    excluded_lines = "\n".join(f"- `{row.piece_id}`: {row.reason}" for row in piece_audit[piece_audit.status == "excluded"].itertuples()) or "- 无"
    report = f"""# Goal 1：纯无标签数据与谱面对齐审计

状态：**{completion['status']} / implemented and locally reproduced**。本轮训练次数：**0**。

## 结果

- 候选配对作品：{len(sources)}；可用：{len(usable)}；停止并排除：{len(piece_audit) - len(usable)}。
- 可用演奏：{int(usable.performances.sum())}；64-beat/32-beat窗口：{len(window_frame)}；整曲变长序列：{len(full_sequence_frame)}。
- 同一份58维缓存同时提供两种互不覆盖的数据视图：固定64-beat滑窗，以及整曲变长序列。变长序列训练时按批次最长序列补齐，补齐位置必须由padding mask排除损失和评价。
- 每个演奏输入严格为：9维演奏曲线 + 16维谱面线索 + 9维多尺度速度代理 + 24维有序动机关系 = 58维。
- 谱面位置统一使用 `occurrence_start_qb + 4 × mc_onset`；beat行 `i` 对应 `[i,i+1)`。
- 所有四组特征共享同一个 `quarterbeat = 0..n-1` 轴；逐缓存形状、有限值及无标签字段检查见 `cache_checks.csv`。
- 原始输入访问仅包括DCML notes/measures与MazurkaBL beat_time/beat_dyn；禁止访问数：{len(forbidden)}。完整路径和SHA-256见 `source_access.csv`。
- 没有读取 harmony、乐句标签、label mask、旧缓存、旧模型、旧预测或句距先验。

## 对齐规则

`next`列按每次访问顺序选择第1、第2……个出口；访问次数超过已列出口、指向不存在小节或超过安全步数即停止该作品。长度兼容规则沿用beat网格端点差异容差：`min(|score_qb-N|, |score_qb-(N-1)|) ≤ 2`。超过容差不回退到折叠谱，也不猜测。

音符起点按 `[i,i+1)`落入beat；持续音只要与区间相交就计为活动。动机分支保留每beat四个子位置；只合并唯一可确定的延音线，歧义延音不静默合并。

## 被停止的作品

{excluded_lines}

## 可视化

{chr(10).join(f'- `{piece_id}` → `visualizations/{piece_id}_alignment.png`' for piece_id in visualized)}

橙色小节块表示重复访问；音符散点、四组输入幅度与展开后的quarterbeat共用横轴。图仅用于对齐审计，不是模型结果。

## 产物

- `manifests/goal1/measure_mapping.csv`
- `manifests/goal1/event_mapping.csv.gz`
- `manifests/goal1/window_manifest.csv.gz`
- `manifests/goal1/full_sequence_manifest.csv.gz`
- `manifests/goal1/feature_schema.csv`
- `manifests/goal1/source_access.csv`
- `cache/unlabeled58/*.npz`
- `reports/goal1/completion_audit.json`
"""
    (report_dir / "data_audit.md").write_text(report, encoding="utf-8")
    status = f"""# Project Status

- 状态：`GOAL 1 {completion['status'].upper()}`
- 已完成：纯无标签原始数据读取、DCML next展开、quarterbeat映射、58维输入、固定窗口与整曲变长序列清单、访问审计、3首可视化
- 可用作品：{len(usable)}；排除作品：{len(piece_audit) - len(usable)}；演奏：{int(usable.performances.sum())}；64/32窗口：{len(window_frame)}；整曲序列：{len(full_sequence_frame)}
- 模型训练：`0`（本Goal禁止训练）
- 下一步：Goal 2 Transformer自监督遮挡任务；尚未启动

详情见 `reports/goal1/data_audit.md`。
"""
    (project_root / "STATUS.md").write_text(status, encoding="utf-8")
    if completion["status"] != "complete":
        raise RuntimeError(f"Goal 1 completion audit failed: {completion}")
    return completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_paths.toml")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    completion = run(project_root, (project_root / args.config).resolve())
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
