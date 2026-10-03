from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    extends = config.pop("extends", None)
    if not extends:
        return config
    base = load_config((path.parent / extends).resolve())

    def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        result = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    return merge(base, config)


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


def canonical_piece_id(opus: Any, number: Any) -> str | None:
    match = re.search(r"(?:op\.?)?\s*0*(\d+)", str(opus), flags=re.IGNORECASE)
    if not match or pd.isna(number):
        return None
    return f"chopin_op{int(match.group(1)):02d}_no{int(float(number))}"


def mazurka_code_to_piece_id(code: str) -> str | None:
    match = re.fullmatch(r"[Mm]0*(\d+)-(\d+)", code.strip())
    if not match:
        return None
    return f"chopin_op{int(match.group(1)):02d}_no{int(match.group(2))}"


def read_csv_clean(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]


@dataclass(frozen=True)
class DcmlPiece:
    piece_id: str
    dcml_key: str
    opus: int
    number: int
    harmony_path: Path
    notes_path: Path
    measures_path: Path
    folded_length_qb: float
    unfolded_length_qb: float


def discover_dcml_pieces(root: Path) -> dict[str, DcmlPiece]:
    metadata = pd.read_csv(root / "metadata.tsv", sep="\t")
    pieces: dict[str, DcmlPiece] = {}
    for row in metadata.itertuples(index=False):
        piece_id = canonical_piece_id(getattr(row, "workNumber", None), getattr(row, "movementNumber", None))
        if piece_id is None:
            continue
        op_match = re.search(r"(\d+)", str(row.workNumber))
        opus = int(op_match.group(1))
        number = int(float(row.movementNumber))
        key = str(row.piece)
        harmony = root / "harmonies" / f"{key}.harmonies.tsv"
        notes = root / "notes" / f"{key}.notes.tsv"
        measures = root / "measures" / f"{key}.measures.tsv"
        if harmony.exists() and notes.exists() and measures.exists():
            pieces[piece_id] = DcmlPiece(
                piece_id=piece_id,
                dcml_key=key,
                opus=opus,
                number=number,
                harmony_path=harmony,
                notes_path=notes,
                measures_path=measures,
                folded_length_qb=float(row.length_qb),
                unfolded_length_qb=float(row.length_qb_unfolded),
            )
    return pieces


def discover_mazurka_files(root: Path) -> dict[str, dict[str, Path | None]]:
    result: dict[str, dict[str, Path | None]] = {}
    for beat_path in sorted((root / "beat_time").glob("*beat_time.csv")):
        code = re.sub(r"beat_time$", "", beat_path.stem, flags=re.IGNORECASE)
        piece_id = mazurka_code_to_piece_id(code)
        if piece_id is None:
            continue
        dyn_candidates = list((root / "beat_dyn").glob(f"{code}beat_dyn*.csv"))
        if not dyn_candidates:
            dyn_candidates = [p for p in (root / "beat_dyn").glob("*.csv") if p.name.lower().startswith(code.lower() + "beat_dyn")]
        xml_candidates = [p for p in (root / "xml_scores").glob("*.xml") if re.search(rf"mazurka0*{int(code[1:].split('-')[0])}-{code.split('-')[1]}\.xml$", p.name, re.IGNORECASE)]
        result[piece_id] = {
            "code": code,
            "beat_time": beat_path,
            "beat_dyn": dyn_candidates[0] if dyn_candidates else None,
            "xml_score": xml_candidates[0] if xml_candidates else None,
        }
    return result


def _next_candidates(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    candidates = []
    for token in re.findall(r"-?\d+", str(value)):
        candidates.append(int(token))
    return candidates


def unfold_measure_path(measures: pd.DataFrame, max_steps: int = 10000) -> list[dict[str, float | int]]:
    """Follow DCML's deterministic `next` graph, selecting successive repeat exits."""
    table = {int(row.mc): row for row in measures.itertuples(index=False)}
    if not table:
        return []
    current = min(table)
    visits: dict[int, int] = {}
    unfolded_qb = 0.0
    path: list[dict[str, float | int]] = []
    active_stop_marker: str | None = None
    for _ in range(max_steps):
        if current not in table:
            raise ValueError(f"Measure graph points to missing mc={current}")
        row = table[current]
        visits[current] = visits.get(current, 0) + 1
        duration = to_float(row.duration_qb)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration for mc={current}: {row.duration_qb}")
        path.append({"mc": current, "start_qb": unfolded_qb, "duration_qb": duration, "visit": visits[current]})
        unfolded_qb += duration
        marker = str(getattr(row, "markers", "")).strip().lower()
        if active_stop_marker and marker == active_stop_marker:
            break
        play_until = str(getattr(row, "play_until", "")).strip().lower()
        jump_backward = str(getattr(row, "jump_bwd", "")).strip().lower()
        if play_until and play_until != "nan" and jump_backward and jump_backward != "nan":
            active_stop_marker = play_until
        candidates = _next_candidates(getattr(row, "next"))
        if not candidates:
            break
        stop_candidates = [candidate for candidate in candidates if candidate in table and str(getattr(table[candidate], "markers", "")).strip().lower() == active_stop_marker]
        if active_stop_marker and stop_candidates:
            selected = stop_candidates[0]
        else:
            choice_index = min(visits[current] - 1, len(candidates) - 1)
            selected = candidates[choice_index]
        if selected == -1:
            break
        current = selected
    else:
        raise RuntimeError("DCML measure traversal exceeded safety limit")
    return path


def folded_measure_path(measures: pd.DataFrame) -> list[dict[str, float | int]]:
    path = []
    position = 0.0
    for row in measures.sort_values("mc").itertuples(index=False):
        duration = to_float(row.duration_qb)
        if not np.isfinite(duration) or duration <= 0:
            continue
        path.append({"mc": int(row.mc), "start_qb": position, "duration_qb": duration, "visit": 1})
        position += duration
    return path


def choose_measure_path(measures: pd.DataFrame, n_beats: int, maximum_length_error: float = 2.0) -> tuple[list[dict[str, float | int]], str, float]:
    candidates: list[tuple[str, list[dict[str, float | int]]]] = [("folded", folded_measure_path(measures))]
    try:
        candidates.append(("dcml_next_graph", unfold_measure_path(measures)))
    except RuntimeError:
        pass
    scored = []
    for name, path in candidates:
        length = sum(float(item["duration_qb"]) for item in path)
        # A beat grid may contain either every beat onset (< duration) or a final endpoint.
        error = min(abs(length - n_beats), abs(length - (n_beats - 1)))
        scored.append((error, 0 if name == "dcml_next_graph" else 1, name, path))
    error, _, name, path = min(scored, key=lambda item: (item[0], item[1]))
    if error > maximum_length_error:
        return [], "score_length_mismatch", float(error)
    return path, name, float(error)


def _nearest_beat(target: float, n_beats: int, maximum_distance: float) -> tuple[str, int | None, float]:
    if not np.isfinite(target):
        return "invalid_coordinate", None, math.nan
    if target < -maximum_distance or target > (n_beats - 1) + maximum_distance:
        return "out_of_range", None, math.nan
    lower = math.floor(target)
    upper = math.ceil(target)
    if lower != upper and abs(target - lower) == abs(upper - target):
        return "ambiguous_tie", None, abs(target - lower)
    index = int(round(target))
    distance = abs(target - index)
    if index < 0 or index >= n_beats or distance > maximum_distance:
        return "out_of_range", None, distance
    return "mapped", index, distance


def map_piece_boundaries(piece: DcmlPiece, beat_frame: pd.DataFrame, maximum_distance: float = 0.5) -> tuple[pd.DataFrame, dict[str, Any]]:
    harmonies = pd.read_csv(piece.harmony_path, sep="\t")
    measures = pd.read_csv(piece.measures_path, sep="\t")
    traversal, traversal_mode, length_error = choose_measure_path(measures, len(beat_frame))
    if not traversal:
        label_rows = harmonies[harmonies["phraseend"].fillna("").astype(str).str.contains("}", regex=False)]
        frame = pd.DataFrame(
            [{"piece_id": piece.piece_id, "dcml_key": piece.dcml_key, "label_row": int(index), "mc": int(row.mc), "mn": row.get("mn", np.nan), "mn_onset": row.get("mn_onset", np.nan), "quarterbeats": row.get("quarterbeats", np.nan), "phraseend": row.get("phraseend", ""), "unfolded_visit": np.nan, "target_unfolded_qb": np.nan, "beat_index": np.nan, "distance_beats": np.nan, "status": "score_length_mismatch"} for index, row in label_rows.iterrows()]
        )
        return frame, {
            "piece_id": piece.piece_id, "dcml_phrase_rows": int(len(label_rows)), "unfolded_candidates": int(len(frame)), "mapped_candidates": 0,
            "mapped_unique_beats": 0, "ambiguous_candidates": 0, "out_of_range_candidates": 0, "invalid_coordinate_candidates": 0,
            "mazurka_beats": int(len(beat_frame)), "traversal_length_qb": np.nan, "expected_unfolded_length_qb": piece.unfolded_length_qb,
            "traversal_mode": traversal_mode, "length_error_beats": length_error, "mapping_eligible": False,
        }
    occurrences: dict[int, list[dict[str, float | int]]] = {}
    for item in traversal:
        occurrences.setdefault(int(item["mc"]), []).append(item)
    measure_starts = {int(row.mc): to_float(row.quarterbeats) for row in measures.itertuples(index=False)}
    label_rows = harmonies[harmonies["phraseend"].fillna("").astype(str).str.contains("}", regex=False)]
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, float]] = set()
    for label_index, row in label_rows.iterrows():
        mc = int(row.mc)
        if mc not in measure_starts:
            records.append({"piece_id": piece.piece_id, "label_row": int(label_index), "mc": mc, "status": "missing_measure"})
            continue
        offset = to_float(row.quarterbeats) - measure_starts[mc]
        for occurrence in occurrences.get(mc, []):
            target = float(occurrence["start_qb"]) + offset
            dedupe_key = (int(occurrence["visit"]), round(target, 8))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            status, beat_index, distance = _nearest_beat(target, len(beat_frame), maximum_distance)
            record = {
                "piece_id": piece.piece_id,
                "dcml_key": piece.dcml_key,
                "label_row": int(label_index),
                "mc": mc,
                "mn": row.get("mn", np.nan),
                "mn_onset": row.get("mn_onset", np.nan),
                "quarterbeats": row.get("quarterbeats", np.nan),
                "phraseend": row.get("phraseend", ""),
                "unfolded_visit": int(occurrence["visit"]),
                "target_unfolded_qb": target,
                "beat_index": beat_index,
                "distance_beats": distance,
                "status": status,
            }
            if beat_index is not None:
                record["measure_number"] = beat_frame.iloc[beat_index]["measure_number"]
                record["beat_number"] = beat_frame.iloc[beat_index]["beat_number"]
            records.append(record)
    frame = pd.DataFrame(records)
    mapped_unique = frame.loc[frame.status == "mapped", "beat_index"].dropna().astype(int).nunique() if not frame.empty else 0
    diagnostics = {
        "piece_id": piece.piece_id,
        "dcml_phrase_rows": int(len(label_rows)),
        "unfolded_candidates": int(len(frame)),
        "mapped_candidates": int((frame.status == "mapped").sum()) if not frame.empty else 0,
        "mapped_unique_beats": int(mapped_unique),
        "ambiguous_candidates": int((frame.status == "ambiguous_tie").sum()) if not frame.empty else 0,
        "out_of_range_candidates": int((frame.status == "out_of_range").sum()) if not frame.empty else 0,
        "invalid_coordinate_candidates": int((frame.status == "invalid_coordinate").sum()) if not frame.empty else 0,
        "mazurka_beats": int(len(beat_frame)),
        "traversal_length_qb": float(sum(float(x["duration_qb"]) for x in traversal)),
        "expected_unfolded_length_qb": piece.unfolded_length_qb,
        "traversal_mode": traversal_mode,
        "length_error_beats": length_error,
        "mapping_eligible": True,
    }
    return frame, diagnostics


def make_piece_splits(piece_ids: Iterable[str], seed: int = 42, folds: int = 5, validation_count: int = 6) -> dict[int, dict[str, list[str]]]:
    piece_ids = np.array(sorted(set(piece_ids)), dtype=object)
    if len(piece_ids) < folds:
        raise ValueError("Not enough pieces for requested folds")
    outer = KFold(n_splits=folds, shuffle=True, random_state=seed)
    result: dict[int, dict[str, list[str]]] = {}
    for fold, (remaining_idx, test_idx) in enumerate(outer.split(piece_ids)):
        remaining = piece_ids[remaining_idx].copy()
        rng = np.random.default_rng(seed + 1000 + fold)
        rng.shuffle(remaining)
        validation = sorted(remaining[:validation_count].tolist())
        train = sorted(remaining[validation_count:].tolist())
        test = sorted(piece_ids[test_idx].tolist())
        assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
        assert set(train) | set(validation) | set(test) == set(piece_ids)
        result[fold] = {"train": train, "validation": validation, "test": test}
    return result


def _rolling_robust_z(values: np.ndarray, radius: int = 4) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for index in range(len(values)):
        local = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        median = np.nanmedian(local)
        mad = np.nanmedian(np.abs(local - median))
        output[index] = float((values[index] - median) / max(1.4826 * mad, 1e-4))
    return np.clip(output, -8.0, 8.0)


def _interpolate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values, dtype=np.float32), valid.astype(np.float32)
    series = pd.Series(values).interpolate(limit_direction="both")
    return series.to_numpy(dtype=np.float32), valid.astype(np.float32)


def build_curve_features(times: np.ndarray, dynamics: np.ndarray) -> np.ndarray:
    times, time_mask = _interpolate(times.astype(float))
    dynamics, dyn_mask = _interpolate(dynamics.astype(float))
    interval = np.diff(times, append=np.nan)
    positive = interval[np.isfinite(interval) & (interval > 0)]
    fallback = float(np.median(positive)) if len(positive) else 0.6
    interval[-1] = interval[-2] if len(interval) > 1 and interval[-2] > 0 else fallback
    bad = ~np.isfinite(interval) | (interval <= 0)
    interval[bad] = fallback
    tempo = np.clip(60.0 / interval, 20.0, 400.0)
    log_tempo = np.log(tempo).astype(np.float32)
    d1 = np.diff(log_tempo, prepend=log_tempo[0]).astype(np.float32)
    d2 = np.diff(d1, prepend=d1[0]).astype(np.float32)
    ddyn = np.diff(dynamics, prepend=dynamics[0]).astype(np.float32)
    return np.column_stack(
        [log_tempo, d1, d2, dynamics, ddyn, _rolling_robust_z(log_tempo), _rolling_robust_z(dynamics), time_mask, dyn_mask]
    ).astype(np.float32)


def align_dynamics_to_beats(beat_frame: pd.DataFrame, dynamics_frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["measure_number", "beat_number"]
    if dynamics_frame.empty or not set(keys).issubset(dynamics_frame.columns):
        return pd.DataFrame(index=np.arange(len(beat_frame)))
    if dynamics_frame.duplicated(keys).any():
        raise ValueError("Dynamics coordinates are not unique within a piece")
    left = beat_frame[keys].copy()
    left["__beat_order"] = np.arange(len(left))
    aligned = left.merge(dynamics_frame, on=keys, how="left", sort=False, validate="one_to_one")
    return aligned.sort_values("__beat_order").drop(columns=["__beat_order"])


def build_score_features(piece: DcmlPiece, beat_frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    notes = pd.read_csv(piece.notes_path, sep="\t")
    measures = pd.read_csv(piece.measures_path, sep="\t")
    traversal, traversal_mode, length_error = choose_measure_path(measures, len(beat_frame))
    if not traversal:
        raise ValueError(f"{piece.piece_id}: score/beat length mismatch ({length_error:.1f} beats)")
    folded_starts = {int(row.mc): to_float(row.quarterbeats) for row in measures.itertuples(index=False)}
    events: list[tuple[float, float, int]] = []
    notes_by_mc = {mc: group for mc, group in notes.groupby(notes.mc.astype(int))}
    for item in traversal:
        mc = int(item["mc"])
        group = notes_by_mc.get(mc)
        if group is None:
            continue
        for row in group.itertuples(index=False):
            onset = float(item["start_qb"]) + to_float(row.quarterbeats) - folded_starts[mc]
            duration = max(to_float(row.duration_qb), 0.0)
            midi = int(row.midi)
            events.append((onset, duration, midi))
    n_beats = len(beat_frame)
    names = [f"chroma_{i}" for i in range(12)] + [
        "meter_position", "strong_beat", "onset_count", "note_density", "rest_fraction",
        "duration_mean", "duration_std", "duration_max", "top_pitch", "bass_pitch", "pitch_range", "melodic_interval",
    ]
    features = np.zeros((n_beats, len(names)), dtype=np.float32)
    top_track = np.full(n_beats, np.nan)
    for beat in range(n_beats):
        onset_events = [(o, d, m) for o, d, m in events if beat <= o < beat + 1]
        active_events = [(o, d, m) for o, d, m in events if o < beat + 1 and o + d > beat]
        for _, _, midi in onset_events:
            features[beat, midi % 12] += 1.0
        chroma_sum = features[beat, :12].sum()
        if chroma_sum:
            features[beat, :12] /= chroma_sum
        beat_number = to_float(beat_frame.iloc[beat]["beat_number"])
        max_beat = max(float(pd.to_numeric(beat_frame["beat_number"], errors="coerce").max()), 1.0)
        features[beat, 12] = beat_number / max_beat
        features[beat, 13] = float(beat_number == 0)
        features[beat, 14] = len(onset_events)
        features[beat, 15] = len(onset_events)
        features[beat, 16] = float(not active_events)
        if onset_events:
            durations = np.array([d for _, d, _ in onset_events], dtype=float)
            midis = np.array([m for _, _, m in onset_events], dtype=float)
            features[beat, 17:20] = [durations.mean(), durations.std(), durations.max()]
            features[beat, 20:23] = [midis.max() / 127.0, midis.min() / 127.0, (midis.max() - midis.min()) / 127.0]
            top_track[beat] = midis.max()
    top_track = pd.Series(top_track).interpolate(limit_direction="both").fillna(60.0).to_numpy()
    features[:, 23] = np.diff(top_track, prepend=top_track[0]) / 24.0
    return features, names


def build_piece_cache(
    piece: DcmlPiece,
    maz_info: dict[str, Path | None],
    boundary_rows: pd.DataFrame,
    cache_path: Path,
) -> dict[str, Any]:
    beat_frame = read_csv_clean(Path(maz_info["beat_time"]))
    dyn_path = maz_info.get("beat_dyn")
    dyn_frame = read_csv_clean(Path(dyn_path)) if dyn_path else pd.DataFrame()
    aligned_dyn = align_dynamics_to_beats(beat_frame, dyn_frame)
    score, score_names = build_score_features(piece, beat_frame)
    reserved = {"measure_number", "beat_number"}
    performance_ids = [c for c in beat_frame.columns if c not in reserved and not c.lower().startswith("index")]
    curve_list = []
    kept_ids = []
    for performance_id in performance_ids:
        times = pd.to_numeric(beat_frame[performance_id], errors="coerce").to_numpy(float)
        if performance_id in aligned_dyn:
            dynamics = pd.to_numeric(aligned_dyn[performance_id], errors="coerce").to_numpy(float)
        else:
            dynamics = np.full(len(beat_frame), np.nan)
        if np.isfinite(times).mean() < 0.8:
            continue
        curve_list.append(build_curve_features(times, dynamics))
        kept_ids.append(performance_id)
    curves = np.stack(curve_list) if curve_list else np.empty((0, len(beat_frame), 9), dtype=np.float32)
    labels = np.zeros(len(beat_frame), dtype=np.float32)
    label_mask = np.ones(len(beat_frame), dtype=np.float32)
    selected = boundary_rows[boundary_rows.piece_id == piece.piece_id]
    for row in selected.itertuples(index=False):
        if row.status == "mapped" and pd.notna(row.beat_index):
            labels[int(row.beat_index)] = 1.0
        elif row.status == "ambiguous_tie" and pd.notna(row.target_unfolded_qb):
            target = float(row.target_unfolded_qb)
            for index in {math.floor(target), math.ceil(target)}:
                if 0 <= index < len(label_mask):
                    label_mask[index] = 0.0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        piece_id=piece.piece_id,
        score=score,
        curves=curves,
        performance_ids=np.array(kept_ids),
        labels=labels,
        label_mask=label_mask,
        measure_number=pd.to_numeric(beat_frame.measure_number, errors="coerce").to_numpy(float),
        beat_number=pd.to_numeric(beat_frame.beat_number, errors="coerce").to_numpy(float),
        score_feature_names=np.array(score_names),
        curve_feature_names=np.array(["log_tempo", "tempo_d1", "tempo_d2", "dynamics", "dynamics_d1", "tempo_robust_z", "dynamics_robust_z", "time_mask", "dynamics_mask"]),
    )
    return {"piece_id": piece.piece_id, "beats": len(beat_frame), "performances": len(kept_ids), "boundaries": int(labels.sum()), "masked_beats": int((label_mask == 0).sum())}


def load_piece_cache(cache_root: Path, piece_id: str) -> dict[str, np.ndarray]:
    path = cache_root / "piece_features" / f"{piece_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
