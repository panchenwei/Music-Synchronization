from pathlib import Path
import json

import numpy as np
import pandas as pd


ASAP_KEY = "Bach/Prelude/bwv_856/LuoJ01M.mid"


def _as_float_array(values):
    if isinstance(values, list):
        return np.array(values, dtype=float)

    if isinstance(values, dict):
        items = []
        for k, v in values.items():
            try:
                sort_key = float(k)
            except Exception:
                sort_key = len(items)
            items.append((sort_key, float(v)))
        items.sort(key=lambda x: x[0])
        return np.array([v for _, v in items], dtype=float)

    raise TypeError(f"Unsupported ASAP beat format: {type(values)}")


def _normalize_beat_types(raw_beat_types, gt_times):
    if raw_beat_types is None:
        return [""] * len(gt_times)

    if isinstance(raw_beat_types, list):
        if len(raw_beat_types) == len(gt_times):
            return list(raw_beat_types)
        return list(raw_beat_types[:len(gt_times)]) + [""] * max(0, len(gt_times) - len(raw_beat_types))

    if isinstance(raw_beat_types, dict):
        keys = list(raw_beat_types.keys())

        try:
            int_keys = [int(float(k)) for k in keys]
            if set(int_keys) >= set(range(len(gt_times))):
                return [raw_beat_types[str(i)] if str(i) in raw_beat_types else raw_beat_types.get(i, "") for i in range(len(gt_times))]
        except Exception:
            pass

        beat_types = []
        numeric_items = []
        for k, v in raw_beat_types.items():
            try:
                numeric_items.append((float(k), v))
            except Exception:
                pass

        if numeric_items:
            key_times = np.array([x[0] for x in numeric_items], dtype=float)
            key_vals = [x[1] for x in numeric_items]

            for t in gt_times:
                idx = int(np.argmin(np.abs(key_times - t)))
                if abs(key_times[idx] - t) <= 1e-3:
                    beat_types.append(key_vals[idx])
                else:
                    beat_types.append("")
            return beat_types

    return [""] * len(gt_times)


def _load_gt_from_txt(performance_annotations_txt: Path):
    gt_times = []
    gt_types = []

    with open(performance_annotations_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            gt_times.append(float(parts[1]))
            raw_type = parts[2].strip() if len(parts) > 2 else ""
            gt_types.append(raw_type.split(",")[0] if raw_type else "")

    return np.array(gt_times, dtype=float), gt_types


def _load_gt_from_json(asap_annotations_json: Path, asap_key):
    with open(asap_annotations_json, "r", encoding="utf-8") as f:
        all_ann = json.load(f)

    if asap_key not in all_ann:
        raise KeyError(f"ASAP key not found: {asap_key}")

    ann = all_ann[asap_key]
    if "performance_beats" not in ann:
        raise KeyError("ASAP annotation missing field: performance_beats")

    gt_times = _as_float_array(ann["performance_beats"])
    gt_types = _normalize_beat_types(ann.get("performance_beats_type"), gt_times)
    return gt_times, gt_types


def _select_predicted_rows_by_time_dp(predicted: pd.DataFrame, gt_times: np.ndarray) -> tuple[pd.DataFrame, str]:
    pred_times = predicted["predicted_performance_time_sec"].to_numpy(dtype=float)
    m = len(pred_times)
    n = len(gt_times)

    dp = np.full((m + 1, n + 1), np.inf, dtype=float)
    take = np.zeros((m + 1, n + 1), dtype=np.uint8)
    dp[:, 0] = 0.0

    for i in range(1, m + 1):
        upper_j = min(i, n)
        for j in range(1, upper_j + 1):
            skip_cost = dp[i - 1, j]
            take_cost = dp[i - 1, j - 1] + (pred_times[i - 1] - gt_times[j - 1]) ** 2

            if take_cost <= skip_cost:
                dp[i, j] = take_cost
                take[i, j] = 1
            else:
                dp[i, j] = skip_cost

    if not np.isfinite(dp[m, n]):
        raise ValueError("dynamic beat matching failed")

    keep = []
    i = m
    j = n
    while j > 0:
        if i <= 0:
            raise ValueError("dynamic beat matching backtrack failed")
        if take[i, j] == 1:
            keep.append(i - 1)
            i -= 1
            j -= 1
        else:
            i -= 1

    keep.reverse()
    selected = predicted.iloc[keep].copy().reset_index(drop=True)
    return selected, f"time_dp_drop_{m - n}"


def _select_gt_indices_by_time_dp(pred_times: np.ndarray, gt_times: np.ndarray) -> np.ndarray:
    m = len(pred_times)
    n = len(gt_times)

    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    take = np.zeros((n + 1, m + 1), dtype=np.uint8)
    dp[:, 0] = 0.0

    for i in range(1, n + 1):
        upper_j = min(i, m)
        for j in range(1, upper_j + 1):
            skip_cost = dp[i - 1, j]
            take_cost = dp[i - 1, j - 1] + (gt_times[i - 1] - pred_times[j - 1]) ** 2

            if take_cost <= skip_cost:
                dp[i, j] = take_cost
                take[i, j] = 1
            else:
                dp[i, j] = skip_cost

    if not np.isfinite(dp[n, m]):
        raise ValueError("dynamic gt beat matching failed")

    keep = []
    i = n
    j = m
    while j > 0:
        if i <= 0:
            raise ValueError("dynamic gt beat matching backtrack failed")
        if take[i, j] == 1:
            keep.append(i - 1)
            i -= 1
            j -= 1
        else:
            i -= 1

    keep.reverse()
    return np.array(keep, dtype=int)


def _select_predicted_rows_for_asap(predicted: pd.DataFrame, gt_times: np.ndarray) -> tuple[pd.DataFrame, str, np.ndarray]:
    predicted = predicted.copy()
    gt_count = len(gt_times)

    if len(predicted) == gt_count:
        return predicted.reset_index(drop=True), "one_to_one", np.arange(gt_count, dtype=int)

    required_cols = {"numerator", "denominator", "beat_in_measure"}
    if required_cols.issubset(set(predicted.columns)):
        numerators = set(predicted["numerator"].dropna().astype(int).unique().tolist())
        denominators = set(predicted["denominator"].dropna().astype(int).unique().tolist())

        if numerators == {12} and denominators == {8}:
            selected = predicted[predicted["beat_in_measure"].astype(int).isin([1, 4, 7, 10])].copy()
            if len(selected) == gt_count:
                return selected.reset_index(drop=True), "compound_12_8_take_1_4_7_10", np.arange(gt_count, dtype=int)
            if len(selected) > gt_count:
                selected, _ = _select_predicted_rows_by_time_dp(selected, gt_times)
                if len(selected) == gt_count:
                    return selected.reset_index(drop=True), "compound_12_8_take_1_4_7_10_then_time_dp", np.arange(gt_count, dtype=int)

        compound_denominators = {8, 16}
        if denominators.issubset(compound_denominators):
            compound_mask = (
                predicted["numerator"].notna()
                & predicted["denominator"].notna()
                & (predicted["numerator"].astype(int) > 3)
                & (predicted["numerator"].astype(int) % 3 == 0)
            )
            selected = predicted[
                (~compound_mask)
                | (((predicted["beat_in_measure"].astype(int) - 1) % 3) == 0)
            ].copy()
            if len(selected) == gt_count:
                return selected.reset_index(drop=True), "compound_meter_take_every_3rd_subbeat", np.arange(gt_count, dtype=int)
            if len(selected) > gt_count:
                selected, _ = _select_predicted_rows_by_time_dp(selected, gt_times)
                if len(selected) == gt_count:
                    return selected.reset_index(drop=True), "compound_meter_take_every_3rd_subbeat_then_time_dp", np.arange(gt_count, dtype=int)

    ratio = len(predicted) / gt_count
    if ratio.is_integer():
        step = int(ratio)
        selected = predicted.iloc[::step].copy()
        if len(selected) == gt_count:
            return selected.reset_index(drop=True), f"downsample_every_{step}", np.arange(gt_count, dtype=int)

    if len(predicted) > gt_count:
        selected, rule = _select_predicted_rows_by_time_dp(predicted, gt_times)
        return selected, rule, np.arange(gt_count, dtype=int)

    if len(predicted) < gt_count:
        keep_idx = _select_gt_indices_by_time_dp(
            predicted["predicted_performance_time_sec"].to_numpy(dtype=float),
            gt_times,
        )
        return predicted.reset_index(drop=True), f"gt_time_dp_drop_{gt_count - len(predicted)}", keep_idx

    raise ValueError(
        "Cannot match predicted beats to ASAP beats. "
        f"predicted_count={len(predicted)}, asap_count={gt_count}. "
        "You may need a custom beat selection rule."
    )


def _compute_metrics(errors_sec: np.ndarray) -> dict:
    abs_errors = np.abs(errors_sec)

    return {
        "count": int(len(abs_errors)),
        "mae_sec": round(float(np.mean(abs_errors)), 5),
        "rmse_sec": round(float(np.sqrt(np.mean(errors_sec ** 2))), 5),
        "median_abs_error_sec": round(float(np.median(abs_errors)), 5),
        "max_abs_error_sec": round(float(np.max(abs_errors)), 5),
        "acc_10ms": round(float(np.mean(abs_errors <= 0.010)), 5),
        "acc_20ms": round(float(np.mean(abs_errors <= 0.020)), 5),
        "acc_50ms": round(float(np.mean(abs_errors <= 0.050)), 5),
        "acc_100ms": round(float(np.mean(abs_errors <= 0.100)), 5),
        "acc_200ms": round(float(np.mean(abs_errors <= 0.200)), 5),
        "acc_500ms": round(float(np.mean(abs_errors <= 0.500)), 5),
    }


def _filter_usable_predicted_rows(predicted: pd.DataFrame) -> pd.DataFrame:
    usable = predicted.copy()

    if "inside_warping_range" in usable.columns:
        inside = usable["inside_warping_range"]
        if inside.dtype != bool:
            inside = inside.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
        usable = usable[inside]

    pred_times = pd.to_numeric(
        usable["predicted_performance_time_sec"],
        errors="coerce",
    )
    usable = usable[np.isfinite(pred_times.to_numpy(dtype=float))].copy()
    usable["predicted_performance_time_sec"] = pred_times.loc[usable.index].to_numpy(dtype=float)
    usable = usable.reset_index(drop=True)

    return usable


def run_valid(
    predicted_beats_csv: Path,
    asap_annotations_json: Path,
    out_csv: Path,
    out_json: Path,
    asap_key=ASAP_KEY,
    performance_annotations_txt: Path | None = None,
):
    print("[1/4] loading predicted beats...")
    predicted = pd.read_csv(predicted_beats_csv)

    if "predicted_performance_time_sec" not in predicted.columns:
        raise ValueError("predicted_beats.csv must contain column: predicted_performance_time_sec")

    raw_predicted_count = len(predicted)
    predicted = _filter_usable_predicted_rows(predicted)
    usable_predicted_count = len(predicted)

    if usable_predicted_count == 0:
        raise ValueError("no usable predicted beats remain after filtering invalid or out-of-range rows")

    print("[2/4] loading beat-level ground truth...")
    if performance_annotations_txt is not None:
        gt_times, gt_types = _load_gt_from_txt(performance_annotations_txt)
        gt_source = str(performance_annotations_txt)
        gt_source_type = "performance_annotations_txt"
    else:
        gt_times, gt_types = _load_gt_from_json(asap_annotations_json, asap_key)
        gt_source = str(asap_annotations_json)
        gt_source_type = "asap_annotations_json"

    print("predicted rows:", raw_predicted_count)
    print("usable predicted rows:", usable_predicted_count)
    print("ASAP gt beats:", len(gt_times))

    print("[3/4] matching predicted beats to ASAP beats...")
    selected, match_rule, gt_keep_idx = _select_predicted_rows_for_asap(predicted, gt_times)
    print("match_rule:", match_rule)

    gt_times = gt_times[gt_keep_idx]
    gt_types = [gt_types[i] for i in gt_keep_idx]

    if len(selected) == 0 or len(gt_times) == 0:
        raise ValueError("no valid beat pairs remain after beat selection")

    pred_times = selected["predicted_performance_time_sec"].to_numpy(dtype=float)
    errors = pred_times - gt_times
    abs_errors = np.abs(errors)

    valid = selected.copy()
    valid.insert(0, "valid_index", np.arange(1, len(valid) + 1))
    valid["asap_beat_type"] = gt_types
    valid["gt_performance_time_sec"] = gt_times
    valid["error_sec"] = errors
    valid["abs_error_sec"] = abs_errors

    metrics = _compute_metrics(errors)

    summary = {
        "asap_key": asap_key,
        "predicted_beats_csv": str(predicted_beats_csv),
        "asap_annotations_json": str(asap_annotations_json),
        "gt_source": gt_source,
        "gt_source_type": gt_source_type,
        "match_rule": match_rule,
        "metrics": metrics,
    }

    print("[4/4] saving valid results...")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(out_csv, index=False, float_format="%.5f")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("saved:", out_csv)
    print("saved:", out_json)

    print("\n[metrics]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n[preview]")
    print(valid.head())
    print(valid.tail())

    return valid, summary


def main():
    src_dir = Path(__file__).resolve().parent
    example_name = "bwv_856"
    example_dir = src_dir / example_name

    predicted_beats_csv = example_dir / "predicted_beats.csv"
    asap_annotations_json = example_dir / "asap_annotations.json"

    out_csv = example_dir / "valid.csv"
    out_json = example_dir / "valid_metrics.json"

    print("Using paths:")
    print("  predicted_beats_csv  =", predicted_beats_csv)
    print("  asap_annotations_json =", asap_annotations_json)
    print("  out_csv               =", out_csv)
    print("  out_json              =", out_json)

    if not predicted_beats_csv.exists():
        raise FileNotFoundError(f"predicted beats csv not found: {predicted_beats_csv}")

    if not asap_annotations_json.exists():
        raise FileNotFoundError(f"asap_annotations.json not found: {asap_annotations_json}")

    run_valid(
        predicted_beats_csv=predicted_beats_csv,
        asap_annotations_json=asap_annotations_json,
        out_csv=out_csv,
        out_json=out_json,
    )


if __name__ == "__main__":
    main()
