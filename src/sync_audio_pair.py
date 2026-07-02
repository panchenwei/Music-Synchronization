from pathlib import Path
import json

import librosa
import numpy as np
import pandas as pd
import scipy.interpolate

from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import (
    make_path_strictly_monotonic,
    compute_optimal_chroma_shift,
    shift_chroma_vectors,
)
from synctoolbox.feature.chroma import (
    pitch_to_chroma,
    quantize_chroma,
    quantized_chroma_to_CENS,
)
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning


DEFAULT_FS = 22050
DEFAULT_FEATURE_RATE = 1000
DEFAULT_STEP_WEIGHTS = np.array([1.5, 1.5, 2.0], dtype=float)
DEFAULT_THRESHOLD_REC = 10 ** 6
DEFAULT_BOUNDARY_HOP_LENGTH = 2048
DEFAULT_BOUNDARY_MARGIN_SEC = 1.5
DEFAULT_BOUNDARY_QUERY_FRACTION = 0.2
DEFAULT_BOUNDARY_MIN_QUERY_SEC = 8.0
DEFAULT_BOUNDARY_MAX_QUERY_SEC = 45.0
DEFAULT_BOUNDARY_MIN_CROP_RATIO = 0.65
DEFAULT_MAX_BOUNDARY_COST = 0.28


def _subsequence_match_range(query_chroma, performance_chroma):
    cost_matrix, warping_path = librosa.sequence.dtw(
        X=query_chroma,
        Y=performance_chroma,
        metric="euclidean",
        subseq=True,
        backtrack=True,
    )

    performance_frames = warping_path[:, 1].astype(int)
    start_frame = int(np.min(performance_frames))
    end_frame = int(np.max(performance_frames))
    endpoint_frame = int(performance_frames[0])
    normalized_cost = float(cost_matrix[-1, endpoint_frame] / max(len(warping_path), 1))

    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "endpoint_frame": endpoint_frame,
        "normalized_cost": normalized_cost,
    }


def locate_performance_bounds(
    y_score,
    y_performance,
    sr,
    hop_length=DEFAULT_BOUNDARY_HOP_LENGTH,
    margin_sec=DEFAULT_BOUNDARY_MARGIN_SEC,
    query_fraction=DEFAULT_BOUNDARY_QUERY_FRACTION,
    min_query_sec=DEFAULT_BOUNDARY_MIN_QUERY_SEC,
    max_query_sec=DEFAULT_BOUNDARY_MAX_QUERY_SEC,
    min_crop_ratio=DEFAULT_BOUNDARY_MIN_CROP_RATIO,
    max_boundary_cost=DEFAULT_MAX_BOUNDARY_COST,
    pin_start=False,
    pin_end=False,
):
    """先粗找 performance 里对得上的那一段。"""

    score_chroma = librosa.feature.chroma_cens(y=y_score, sr=sr, hop_length=hop_length)
    performance_chroma = librosa.feature.chroma_cens(y=y_performance, sr=sr, hop_length=hop_length)

    score_energy = np.linalg.norm(score_chroma, axis=0)
    active_idx = np.flatnonzero(score_energy > 1e-8)

    if active_idx.size == 0:
        query_start_frame = 0
        query_end_frame = score_chroma.shape[1] - 1
        score_query = score_chroma
    else:
        query_start_frame = int(active_idx[0])
        query_end_frame = int(active_idx[-1])
        score_query = score_chroma[:, query_start_frame : query_end_frame + 1]

    if score_query.shape[1] < 2 or performance_chroma.shape[1] < 2:
        raise ValueError("coarse boundary localization needs at least 2 chroma frames")

    min_query_frames = max(2, int(np.ceil(min_query_sec * sr / hop_length)))
    max_query_frames = max(min_query_frames, int(np.ceil(max_query_sec * sr / hop_length)))
    query_len = int(np.ceil(score_query.shape[1] * query_fraction))
    query_len = max(min_query_frames, min(max_query_frames, query_len, score_query.shape[1]))

    head_query = score_query[:, :query_len]
    tail_query = score_query[:, -query_len:]
    head_match = _subsequence_match_range(head_query, performance_chroma)
    tail_match = _subsequence_match_range(tail_query, performance_chroma)

    margin_frames = int(np.ceil(margin_sec * sr / hop_length))
    crop_start_frame = max(0, head_match["start_frame"] - margin_frames)
    crop_end_frame = min(performance_chroma.shape[1] - 1, tail_match["end_frame"] + margin_frames)

    fallback_reason = None
    if crop_end_frame <= crop_start_frame:
        fallback_reason = "inverted_match_range"

    start_sample = int(crop_start_frame * hop_length)
    end_sample = int(min(len(y_performance), (crop_end_frame + 1) * hop_length))

    if pin_start:
        start_sample = 0
    if pin_end:
        end_sample = len(y_performance)

    if end_sample <= start_sample:
        fallback_reason = fallback_reason or "empty_crop"

    score_duration_sec = float(len(y_score) / sr)
    performance_duration_sec = float(len(y_performance) / sr)
    cropped_performance = y_performance[start_sample:end_sample] if end_sample > start_sample else y_performance
    cropped_duration_sec = float(len(cropped_performance) / sr)

    if cropped_duration_sec < score_duration_sec * min_crop_ratio:
        fallback_reason = fallback_reason or "crop_too_short_for_score"

    normalized_cost = float((head_match["normalized_cost"] + tail_match["normalized_cost"]) / 2.0)
    if normalized_cost > max_boundary_cost:
        fallback_reason = fallback_reason or "boundary_cost_too_high"

    if fallback_reason is not None:
        start_sample = 0
        end_sample = len(y_performance)
        cropped_performance = y_performance
        cropped_duration_sec = performance_duration_sec

    boundary_meta = {
        "method": "chroma_cens_subsequence_dtw_head_tail",
        "coarse_hop_length": int(hop_length),
        "margin_sec": round(float(margin_sec), 5),
        "query_fraction": round(float(query_fraction), 5),
        "max_boundary_cost": round(float(max_boundary_cost), 5),
        "query_frames": int(query_len),
        "pin_start": bool(pin_start),
        "pin_end": bool(pin_end),
        "query_score_start_frame": int(query_start_frame),
        "query_score_end_frame": int(query_end_frame),
        "head_match_start_frame": int(head_match["start_frame"]),
        "head_match_end_frame": int(head_match["end_frame"]),
        "tail_match_start_frame": int(tail_match["start_frame"]),
        "tail_match_end_frame": int(tail_match["end_frame"]),
        "head_match_cost": round(float(head_match["normalized_cost"]), 5),
        "tail_match_cost": round(float(tail_match["normalized_cost"]), 5),
        "performance_start_sample": int(start_sample),
        "performance_end_sample": int(end_sample),
        "performance_start_sec": round(float(start_sample / sr), 5),
        "performance_end_sec": round(float(end_sample / sr), 5),
        "original_performance_duration_sec": round(performance_duration_sec, 5),
        "cropped_performance_duration_sec": round(cropped_duration_sec, 5),
        "score_duration_sec": round(score_duration_sec, 5),
        "used_full_performance": bool(fallback_reason is not None),
        "fallback_reason": fallback_reason,
        "normalized_subsequence_boundary_cost": round(normalized_cost, 5),
    }

    return cropped_performance, start_sample, start_sample / sr, boundary_meta


def extract_highres_features(
    audio: np.ndarray,
    label: str,
    fs=DEFAULT_FS,
    feature_rate=DEFAULT_FEATURE_RATE,
):
    """先把高分辨率特征抽出来。"""

    print(f"[{label}] estimating tuning...")
    tuning_offset = estimate_tuning(audio, fs)
    print(f"[{label}] tuning_offset: {tuning_offset} cents")

    print(f"[{label}] extracting pitch features...")
    f_pitch = audio_to_pitch_features(
        f_audio=audio,
        Fs=fs,
        tuning_offset=tuning_offset,
        feature_rate=feature_rate,
        verbose=False,
    )

    print(f"[{label}] pitch -> chroma...")
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)

    print(f"[{label}] quantizing chroma...")
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

    print(f"[{label}] extracting pitch onset features...")
    f_pitch_onset = audio_to_pitch_onset_features(
        f_audio=audio,
        Fs=fs,
        tuning_offset=tuning_offset,
        verbose=False,
    )

    print(f"[{label}] onset -> DLNCO...")
    f_dlnco = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=f_chroma_quantized.shape[1],
        visualize=False,
    )

    print(f"[{label}] chroma shape: {f_chroma_quantized.shape}")
    print(f"[{label}] DLNCO shape:  {f_dlnco.shape}")

    return f_chroma_quantized, f_dlnco


def maybe_apply_chroma_shift(
    score_chroma,
    score_dlnco,
    perf_chroma,
    perf_dlnco,
    feature_rate=DEFAULT_FEATURE_RATE,
):
    """如果两段音频有点转调，就顺手矫正一下。"""

    print("[shift] computing optimal chroma shift...")

    score_cens_1hz = quantized_chroma_to_CENS(
        score_chroma,
        win_len_smooth=201,
        downsamp_smooth=50,
        input_feature_rate=feature_rate,
    )[0]

    perf_cens_1hz = quantized_chroma_to_CENS(
        perf_chroma,
        win_len_smooth=201,
        downsamp_smooth=50,
        input_feature_rate=feature_rate,
    )[0]

    opt_shift = compute_optimal_chroma_shift(score_cens_1hz, perf_cens_1hz)

    print(f"[shift] optimal chroma shift: {opt_shift} bins")

    if opt_shift != 0:
        perf_chroma = shift_chroma_vectors(perf_chroma, opt_shift)
        perf_dlnco = shift_chroma_vectors(perf_dlnco, opt_shift)

    return perf_chroma, perf_dlnco, opt_shift


def run_highres_sync(
    score_wav: Path,
    performance_wav: Path,
    out_dir: Path,
    fs=DEFAULT_FS,
    feature_rate=DEFAULT_FEATURE_RATE,
    step_weights=DEFAULT_STEP_WEIGHTS,
    threshold_rec=DEFAULT_THRESHOLD_REC,
    boundary_hop_length=DEFAULT_BOUNDARY_HOP_LENGTH,
    boundary_margin_sec=DEFAULT_BOUNDARY_MARGIN_SEC,
    boundary_query_fraction=DEFAULT_BOUNDARY_QUERY_FRACTION,
    boundary_min_query_sec=DEFAULT_BOUNDARY_MIN_QUERY_SEC,
    boundary_max_query_sec=DEFAULT_BOUNDARY_MAX_QUERY_SEC,
    boundary_min_crop_ratio=DEFAULT_BOUNDARY_MIN_CROP_RATIO,
    max_boundary_cost=DEFAULT_MAX_BOUNDARY_COST,
    pin_performance_start=False,
    pin_performance_end=False,
):
    """跑一遍高分辨率同步。"""

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading audio...")
    score_audio, sr_score = librosa.load(score_wav, sr=fs, mono=True)
    perf_audio, sr_perf = librosa.load(performance_wav, sr=fs, mono=True)

    if sr_score != fs or sr_perf != fs:
        raise ValueError("loaded audio sample rate does not match requested fs")

    y_perf_for_sync, perf_start_sample, perf_offset_sec, boundary_meta = locate_performance_bounds(
        score_audio,
        perf_audio,
        fs,
        hop_length=boundary_hop_length,
        margin_sec=boundary_margin_sec,
        query_fraction=boundary_query_fraction,
        min_query_sec=boundary_min_query_sec,
        max_query_sec=boundary_max_query_sec,
        min_crop_ratio=boundary_min_crop_ratio,
        max_boundary_cost=max_boundary_cost,
        pin_start=pin_performance_start,
        pin_end=pin_performance_end,
    )

    score_duration = len(score_audio) / fs
    perf_duration = len(perf_audio) / fs

    print("[performance bounds]")
    print("  performance_start_sec:", round(perf_offset_sec, 5))
    print("  performance_end_sec:", boundary_meta["performance_end_sec"])
    print("  normalized_subsequence_boundary_cost:", boundary_meta["normalized_subsequence_boundary_cost"])
    print(f"score_wav:       {score_wav}")
    print(f"performance_wav: {performance_wav}")
    print(f"score duration:       {score_duration:.5f} sec")
    print(f"performance duration: {perf_duration:.5f} sec")

    print("[2/6] extracting high-resolution features for score audio...")
    score_chroma, score_dlnco = extract_highres_features(
        score_audio,
        "score",
        fs=fs,
        feature_rate=feature_rate,
    )

    print("[3/6] extracting high-resolution features for performance audio...")
    perf_chroma, perf_dlnco = extract_highres_features(
        y_perf_for_sync,
        "performance",
        fs=fs,
        feature_rate=feature_rate,
    )

    print("[4/6] optional chroma shift correction...")
    perf_chroma, perf_dlnco, opt_shift = maybe_apply_chroma_shift(
        score_chroma,
        score_dlnco,
        perf_chroma,
        perf_dlnco,
        feature_rate=feature_rate,
    )

    print("[5/6] running MrMsDTW...")
    wp = sync_via_mrmsdtw(
        f_chroma1=score_chroma,
        f_onset1=score_dlnco,
        f_chroma2=perf_chroma,
        f_onset2=perf_dlnco,
        input_feature_rate=feature_rate,
        step_weights=np.asarray(step_weights, dtype=float),
        threshold_rec=threshold_rec,
        verbose=False,
    )

    print("raw warping_path shape:", wp.shape)

    print("[6/6] making warping path strictly monotonic...")
    wp = make_path_strictly_monotonic(wp)

    print("strict warping_path shape:", wp.shape)

    score_times = wp[0] / feature_rate
    performance_times = wp[1] / feature_rate + perf_offset_sec

    out_npy = out_dir / "warping_path_highres.npy"
    out_csv = out_dir / "warping_path_highres.csv"
    out_meta = out_dir / "sync_highres_meta.json"

    np.save(out_npy, wp)

    df = pd.DataFrame(
        {
            "score_time_sec": score_times,
            "performance_time_sec": performance_times,
        }
    )
    df.to_csv(out_csv, index=False, float_format="%.5f")

    meta = {
        "score_wav": str(score_wav),
        "performance_wav": str(performance_wav),
        "fs": fs,
        "feature_rate": feature_rate,
        "score_duration_sec": round(float(score_duration), 5),
        "performance_duration_sec": round(float(perf_duration), 5),
        "score_leading_silence_sec": 0.0,
        "performance_leading_silence_sec": round(float(perf_offset_sec), 5),
        "score_start_sample": 0,
        "performance_start_sample": int(perf_start_sample),
        "optimal_chroma_shift_bins": int(opt_shift),
        "warping_path_points": int(wp.shape[1]),
        "boundary_localization": boundary_meta,
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved:", out_npy)
    print("saved:", out_csv)
    print("saved:", out_meta)

    print("\nexample mapping:")
    mapper = scipy.interpolate.interp1d(
        score_times,
        performance_times,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    for t in [0, 5, 10, 20, 30, 40, 50]:
        if 0 <= t <= score_times[-1]:
            mapped = float(mapper(t))
            if np.isfinite(mapped):
                print(f"score_time {t:0.5f}s -> performance_time {mapped:0.5f}s")

    return out_csv


def main():
    src_dir = Path(__file__).resolve().parent
    example_name = "bwv_856"
    example_dir = src_dir / example_name

    score_wav = example_dir / "score_synth.wav"
    performance_wav = example_dir / "performance.wav"
    out_dir = example_dir

    if not score_wav.exists():
        raise FileNotFoundError(f"score wav not found: {score_wav}")

    if not performance_wav.exists():
        raise FileNotFoundError(f"performance wav not found: {performance_wav}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Using paths:")
    print("  score_wav       =", score_wav)
    print("  performance_wav =", performance_wav)
    print("  out_dir         =", out_dir)

    run_highres_sync(
        score_wav=score_wav,
        performance_wav=performance_wav,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
