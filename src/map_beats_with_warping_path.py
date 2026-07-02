from pathlib import Path
import json

import numpy as np
import pandas as pd
import scipy.interpolate


def build_mapper_from_warping_path(warping_csv: Path):
    """把 warping path 读进来，做个时间映射器。"""

    wp = pd.read_csv(warping_csv)

    required_cols = {"score_time_sec", "performance_time_sec"}
    missing = required_cols - set(wp.columns)
    if missing:
        raise ValueError(f"warping csv missing columns: {missing}")

    wp = wp[["score_time_sec", "performance_time_sec"]].dropna()
    wp["score_time_sec"] = wp["score_time_sec"].astype(float)
    wp["performance_time_sec"] = wp["performance_time_sec"].astype(float)

    wp = wp.sort_values("score_time_sec")

    # 同一个 score_time 如果撞车了，就把 performance_time 取平均。
    wp = wp.groupby("score_time_sec", as_index=False)["performance_time_sec"].mean()

    score_times = wp["score_time_sec"].to_numpy(dtype=float)
    performance_times = wp["performance_time_sec"].to_numpy(dtype=float)

    if len(score_times) < 2:
        raise ValueError("warping path needs at least 2 points")

    mapper = scipy.interpolate.interp1d(
        score_times,
        performance_times,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    return mapper, score_times, performance_times


def map_beats_with_warping_path(
    score_beats_csv: Path,
    warping_csv: Path,
    out_predicted_csv: Path,
    out_alignment_json: Path,
):
    """把 score beat 映到 performance 时间轴上。"""

    print("[1/4] loading score beats...")
    beats = pd.read_csv(score_beats_csv)

    if "score_time_sec" not in beats.columns:
        raise ValueError("score_beats csv must contain column: score_time_sec")

    beats["score_time_sec"] = beats["score_time_sec"].astype(float)

    print("[2/4] loading warping path and building mapper...")
    mapper, wp_score_times, wp_performance_times = build_mapper_from_warping_path(warping_csv)

    min_score_time = float(np.min(wp_score_times))
    max_score_time = float(np.max(wp_score_times))

    print("warping score_time range:")
    print("  min_score_time:", round(min_score_time, 5))
    print("  max_score_time:", round(max_score_time, 5))

    # 先标一下哪些 beat 在 warping path 覆盖范围里。
    beats["inside_warping_range"] = (
        (beats["score_time_sec"] >= min_score_time)
        & (beats["score_time_sec"] <= max_score_time)
    )

    outside_count = int((~beats["inside_warping_range"]).sum())
    if outside_count > 0:
        print(f"[warning] {outside_count} beats are outside warping path range; predicted time will be NaN.")

    print("[3/4] mapping beats...")
    beats["predicted_performance_time_sec"] = mapper(
        beats["score_time_sec"].to_numpy(dtype=float)
    )

    # 输出列顺手排一下，后面看着省事。
    preferred_cols = [
        "beat_index",
        "measure_number_guess",
        "beat_in_measure",
        "numerator",
        "denominator",
        "score_tick",
        "score_time_sec",
        "predicted_performance_time_sec",
        "inside_warping_range",
    ]
    cols = [c for c in preferred_cols if c in beats.columns]
    rest_cols = [c for c in beats.columns if c not in cols]
    beats = beats[cols + rest_cols]

    print("[4/4] saving outputs...")
    out_predicted_csv.parent.mkdir(parents=True, exist_ok=True)

    beats.to_csv(out_predicted_csv, index=False, float_format="%.5f")

    alignment = {
        "source": {
            "score_beats_csv": str(score_beats_csv),
            "warping_csv": str(warping_csv),
        },
        "warping_score_time_range": {
            "min_score_time_sec": round(float(min_score_time), 5),
            "max_score_time_sec": round(float(max_score_time), 5),
        },
        "beats": [
            {
                "beat_index": int(row.beat_index) if hasattr(row, "beat_index") else int(i + 1),
                "measure_number_guess": int(row.measure_number_guess) if hasattr(row, "measure_number_guess") else None,
                "beat_in_measure": int(row.beat_in_measure) if hasattr(row, "beat_in_measure") else None,
                "score_time_sec": round(float(row.score_time_sec), 5),
                "time_sec": None if not np.isfinite(row.predicted_performance_time_sec) else round(float(row.predicted_performance_time_sec), 5),
                "inside_warping_range": bool(row.inside_warping_range),
            }
            for i, row in enumerate(beats.itertuples(index=False))
        ],
    }

    with open(out_alignment_json, "w", encoding="utf-8") as f:
        json.dump(alignment, f, ensure_ascii=False, indent=2)

    print("saved:", out_predicted_csv)
    print("saved:", out_alignment_json)

    print("\n[predicted beats preview]")
    print(beats.head())
    print(beats.tail())

    return beats


def main():
    src_dir = Path(__file__).resolve().parent
    example_name = "bwv_856"
    example_dir = src_dir / example_name

    score_beats_csv = example_dir / "score_beats_from_midi.csv"
    warping_csv = example_dir / "warping_path_highres.csv"

    out_predicted_csv = example_dir / "predicted_beats.csv"
    out_alignment_json = example_dir / "alignment_beats.json"

    print("Using paths:")
    print("  score_beats_csv    =", score_beats_csv)
    print("  warping_csv        =", warping_csv)
    print("  out_predicted_csv  =", out_predicted_csv)
    print("  out_alignment_json =", out_alignment_json)

    if not score_beats_csv.exists():
        raise FileNotFoundError(f"score beats csv not found: {score_beats_csv}")

    if not warping_csv.exists():
        raise FileNotFoundError(f"warping csv not found: {warping_csv}")

    map_beats_with_warping_path(
        score_beats_csv=score_beats_csv,
        warping_csv=warping_csv,
        out_predicted_csv=out_predicted_csv,
        out_alignment_json=out_alignment_json,
    )


if __name__ == "__main__":
    main()
