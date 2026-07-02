from pathlib import Path
import contextlib
import gc
import io
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from shutil import copyfile

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from map_beats_with_warping_path import map_beats_with_warping_path
from parse_score_beats_from_midi import parse_beats_from_midi
from sync_audio_pair import run_highres_sync
from valid import run_valid


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

# dataset 相关
DATASET_ROOT = ROOT_DIR / "dataset" / "asap-dataset"
METADATA_CSV = DATASET_ROOT / "metadata.csv"
ASAP_ANNOTATIONS_JSON = DATASET_ROOT / "asap_annotations.json"
MAESTRO_ROOT = ROOT_DIR / "dataset" / "maestro-v2.0.0"
OUTPUT_ROOT = ROOT_DIR / "output"

# MuseScore 相关
MUSESCORE_EXE = Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")

# sync_audio_pair.py 的参数
SYNC_FS = 22050
SYNC_FEATURE_RATE = 1000
SYNC_STEP_WEIGHTS = [1.5, 1.5, 2.0]
SYNC_THRESHOLD_REC = 10 ** 6
BOUNDARY_HOP_LENGTH = 2048
BOUNDARY_MARGIN_SEC = 1.5
BOUNDARY_QUERY_FRACTION = 0.2
BOUNDARY_MIN_QUERY_SEC = 8.0
BOUNDARY_MAX_QUERY_SEC = 45.0
BOUNDARY_MIN_CROP_RATIO = 0.65
MAX_BOUNDARY_COST = 0.28

# maestro 裁音频用
CLIP_PADDING_SEC = 0.1
MAX_WORKERS = min(4, os.cpu_count() or 1)
BATCH_SIZE = max(MAX_WORKERS * 2, 16)
PIECE_SAMPLE_SIZE = 24
PIECE_SAMPLE_SEED = 42

# 根目录总表
TOTAL_VALID_CSV = OUTPUT_ROOT / "total_valid.csv"
TOTAL_FAILURES_CSV = OUTPUT_ROOT / "total_failures.csv"
SAMPLED_PIECES_CSV = OUTPUT_ROOT / "sampled_pieces.csv"
RESET_OUTPUT_FILENAMES = [
    "warping_path_highres.npy",
    "warping_path_highres.csv",
    "sync_highres_meta.json",
    "predicted_beats.csv",
    "alignment_beats.json",
    "valid.csv",
    "valid_metrics.json",
]
MUSESCORE_LOCK = OUTPUT_ROOT / ".musescore.lock"

TOTAL_VALID_COLUMNS = [
    "composer",
    "title",
    "folder",
    "performer",
    "xml_score",
    "audio_performance",
    "midi_performance",
    "performance_annotations",
    "asap_key",
    "score_and_performance_aligned",
    "output_dir",
]

TOTAL_FAILURE_COLUMNS = [
    "composer",
    "title",
    "folder",
    "performer",
    "xml_score",
    "audio_performance",
    "midi_performance",
    "asap_key",
    "output_dir",
    "stage",
    "error_message",
]


def run_musescore_export(src_path: Path, out_path: Path):
    lock_fd = None
    while lock_fd is None:
        try:
            lock_fd = os.open(str(MUSESCORE_LOCK), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            time.sleep(0.2)

    try:
        subprocess.run(
            [str(MUSESCORE_EXE), "-o", str(out_path), str(src_path)],
            check=True,
        )
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        MUSESCORE_LOCK.unlink(missing_ok=True)


def has_audio_value(value) -> bool:
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def to_optional_float(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def resolve_maestro_audio_path(row: pd.Series) -> Path:
    raw_path = row.get("maestro_audio_performance")
    if pd.isna(raw_path) or not str(raw_path).strip():
        raise FileNotFoundError("maestro_audio_performance is empty")

    path = Path(str(raw_path).replace("{maestro}", str(MAESTRO_ROOT)))
    if not path.exists():
        raise FileNotFoundError(f"maestro audio not found: {path}")

    return path


def clip_and_copy_audio(in_path: Path, out_path: Path, start=None, end=None, padding=CLIP_PADDING_SEC):
    if start is None and end is None:
        if in_path != out_path:
            copyfile(in_path, out_path)
        return

    s = 0.0 if start is None else float(start)
    dur = None if end is None else float(end) - s
    data, sr = librosa.load(
        str(in_path),
        sr=None,
        mono=False,
        offset=s,
        duration=dur,
    )

    if s > 0:
        samples = int(sr * min(float(padding), s))
        if data.ndim == 1:
            data = np.append(np.zeros(samples), data)
        else:
            zeros = np.zeros((data.shape[0], samples))
            data = np.asfortranarray(np.concatenate((zeros, data), axis=1))

    output = data.astype(np.float32)
    if output.ndim > 1:
        output = output.T
    sf.write(str(out_path), output, sr)


def prepare_performance_audio(row: pd.Series, paths: dict):
    start = to_optional_float(row.get("start"))
    end = to_optional_float(row.get("end"))

    if paths["performance_wav"].exists():
        return

    clip_and_copy_audio(
        paths["maestro_audio_src"],
        paths["performance_wav"],
        start=start,
        end=end,
        padding=CLIP_PADDING_SEC,
    )


def outputs_exist(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def run_stage_with_cache(stage_name: str, output_paths: list[Path], stage_fn, load_fn=None):
    if output_paths and outputs_exist(*output_paths):
        if load_fn is not None:
            return load_fn()
        return None
    return stage_fn()


def run_sync_stage(row: pd.Series, paths: dict):
    prepare_performance_audio(row, paths)
    pin_start = to_optional_float(row.get("start")) is not None
    pin_end = to_optional_float(row.get("end")) is not None
    return run_highres_sync(
        paths["score_wav"],
        paths["performance_wav"],
        paths["output_dir"],
        fs=SYNC_FS,
        feature_rate=SYNC_FEATURE_RATE,
        step_weights=SYNC_STEP_WEIGHTS,
        threshold_rec=SYNC_THRESHOLD_REC,
        boundary_hop_length=BOUNDARY_HOP_LENGTH,
        boundary_margin_sec=BOUNDARY_MARGIN_SEC,
        boundary_query_fraction=BOUNDARY_QUERY_FRACTION,
        boundary_min_query_sec=BOUNDARY_MIN_QUERY_SEC,
        boundary_max_query_sec=BOUNDARY_MAX_QUERY_SEC,
        boundary_min_crop_ratio=BOUNDARY_MIN_CROP_RATIO,
        max_boundary_cost=MAX_BOUNDARY_COST,
        pin_performance_start=pin_start,
        pin_performance_end=pin_end,
    )


def export_score_midi_with_fallback(paths: dict):
    try:
        run_musescore_export(paths["xml_score"], paths["score_midi"])
    except subprocess.CalledProcessError:
        if not paths["dataset_midi_score"].exists():
            raise
        copyfile(paths["dataset_midi_score"], paths["score_midi"])


def build_paths(row: pd.Series) -> dict:
    folder = str(row["folder"])
    midi_performance = str(row["midi_performance"])
    performer = Path(midi_performance).stem
    asap_key = midi_performance
    output_dir = OUTPUT_ROOT / Path(folder) / performer

    return {
        "folder": folder,
        "performer": performer,
        "asap_key": asap_key,
        "output_dir": output_dir,
        "xml_score": DATASET_ROOT / str(row["xml_score"]),
        "dataset_midi_score": DATASET_ROOT / str(row["midi_score"]),
        "performance_annotations_txt": DATASET_ROOT / str(row["performance_annotations"]),
        "maestro_audio_src": resolve_maestro_audio_path(row),
        "performance_wav": output_dir / "performance.wav",
        "score_midi": output_dir / "score.mid",
        "score_wav": output_dir / "score_synth.wav",
        "score_beats_csv": output_dir / "score_beats_from_midi.csv",
        "warping_npy": output_dir / "warping_path_highres.npy",
        "warping_csv": output_dir / "warping_path_highres.csv",
        "sync_meta_json": output_dir / "sync_highres_meta.json",
        "predicted_beats_csv": output_dir / "predicted_beats.csv",
        "alignment_json": output_dir / "alignment_beats.json",
        "valid_csv": output_dir / "valid.csv",
        "valid_metrics_json": output_dir / "valid_metrics.json",
    }


def make_context(row: pd.Series, paths: dict, annotations: dict) -> dict:
    asap_key = paths["asap_key"]
    ann = annotations.get(asap_key, {})

    return {
        "composer": row["composer"],
        "title": row["title"],
        "folder": paths["folder"],
        "performer": paths["performer"],
        "xml_score": str(row["xml_score"]),
        "audio_performance": str(row["audio_performance"]),
        "midi_performance": str(row["midi_performance"]),
        "performance_annotations": str(row["performance_annotations"]),
        "asap_key": asap_key,
        "score_and_performance_aligned": ann.get("score_and_performance_aligned"),
        "output_dir": str(paths["output_dir"]),
    }


def append_failure(failures: list, row: pd.Series, paths: dict, stage: str, error: Exception):
    failures.append(
        {
            "composer": row["composer"],
            "title": row["title"],
            "folder": paths["folder"],
            "performer": paths["performer"],
            "xml_score": str(row["xml_score"]),
            "audio_performance": str(row["audio_performance"]),
            "midi_performance": str(row["midi_performance"]),
            "asap_key": paths["asap_key"],
            "output_dir": str(paths["output_dir"]),
            "stage": stage,
            "error_message": str(error),
        }
    )


def build_failure_row(row: pd.Series, paths: dict, stage: str, error: Exception):
    return {
        "composer": row["composer"],
        "title": row["title"],
        "folder": paths["folder"],
        "performer": paths["performer"],
        "xml_score": str(row["xml_score"]),
        "audio_performance": str(row["audio_performance"]),
        "midi_performance": str(row["midi_performance"]),
        "asap_key": paths["asap_key"],
        "output_dir": str(paths["output_dir"]),
        "stage": stage,
        "error_message": str(error),
    }


def append_valid_frame(valid_df: pd.DataFrame, out_csv: Path, wrote_header: bool):
    if valid_df is None or valid_df.empty:
        return wrote_header, 0

    valid_df.to_csv(
        out_csv,
        mode="a",
        header=not wrote_header,
        index=False,
        float_format="%.5f",
    )
    row_count = len(valid_df)
    return True, row_count


def append_failure_row(failure: dict, out_csv: Path, wrote_header: bool):
    if not failure:
        return wrote_header, 0

    failure_df = pd.DataFrame([failure], columns=TOTAL_FAILURE_COLUMNS)
    failure_df.to_csv(
        out_csv,
        mode="a",
        header=not wrote_header,
        index=False,
    )
    row_count = len(failure_df)
    del failure_df
    return True, row_count


def sample_piece_folders(metadata: pd.DataFrame) -> list[str]:
    folders = sorted(metadata["folder"].dropna().astype(str).unique().tolist())
    if len(folders) <= PIECE_SAMPLE_SIZE:
        return folders

    sampled = (
        pd.Series(folders)
        .sample(n=PIECE_SAMPLE_SIZE, random_state=PIECE_SAMPLE_SEED, replace=False)
        .sort_values()
        .tolist()
    )
    return sampled


def reset_sample_output_files(rows: list[dict]):
    for row in rows:
        folder = str(row["folder"])
        performer = Path(str(row["midi_performance"])).stem
        output_dir = OUTPUT_ROOT / Path(folder) / performer
        for filename in RESET_OUTPUT_FILENAMES:
            (output_dir / filename).unlink(missing_ok=True)


def process_row(row: pd.Series, annotations: dict):
    paths = build_paths(row)
    context = make_context(row, paths, annotations)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    stages = [
        (
            "xml_to_midi",
            [paths["score_midi"]],
            lambda: export_score_midi_with_fallback(paths),
            None,
        ),
        (
            "midi_to_wav",
            [paths["score_wav"]],
            lambda: run_musescore_export(paths["score_midi"], paths["score_wav"]),
            None,
        ),
        (
            "parse_score_beats",
            [paths["score_beats_csv"]],
            lambda: parse_beats_from_midi(paths["score_midi"], paths["score_beats_csv"]),
            None,
        ),
        (
            "sync_audio_pair",
            [paths["warping_npy"], paths["warping_csv"], paths["sync_meta_json"]],
            lambda: run_sync_stage(row, paths),
            None,
        ),
        (
            "map_beats",
            [paths["predicted_beats_csv"], paths["alignment_json"]],
            lambda: map_beats_with_warping_path(
                score_beats_csv=paths["score_beats_csv"],
                warping_csv=paths["warping_csv"],
                out_predicted_csv=paths["predicted_beats_csv"],
                out_alignment_json=paths["alignment_json"],
            ),
            None,
        ),
        (
            "validate",
            [paths["valid_csv"], paths["valid_metrics_json"]],
            lambda: run_valid(
                predicted_beats_csv=paths["predicted_beats_csv"],
                asap_annotations_json=ASAP_ANNOTATIONS_JSON,
                out_csv=paths["valid_csv"],
                out_json=paths["valid_metrics_json"],
                asap_key=paths["asap_key"],
                performance_annotations_txt=paths["performance_annotations_txt"],
            ),
            lambda: (pd.read_csv(paths["valid_csv"]), None),
        ),
    ]

    stage_result = None
    for stage_name, output_paths, stage_fn, load_fn in stages:
        try:
            stage_result = run_stage_with_cache(stage_name, output_paths, stage_fn, load_fn)
        except Exception as error:
            return None, paths, context, stage_name, error

    valid_df, _summary = stage_result
    valid_df = valid_df.copy()

    for key, value in reversed(list(context.items())):
        valid_df.insert(0, key, value)

    return valid_df, paths, context, None, None


def process_sample(index: int, total: int, row: dict, annotations: dict):
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return index, row, process_row(row, annotations)
    except Exception as error:
        folder = str(row["folder"])
        performer = Path(str(row["midi_performance"])).stem
        output_dir = OUTPUT_ROOT / Path(folder) / performer
        paths = {
            "folder": folder,
            "performer": performer,
            "asap_key": str(row["midi_performance"]),
            "output_dir": output_dir,
        }
        context = {}
        return index, row, (None, paths, context, "sync_audio_pair", error)


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"dataset root not found: {DATASET_ROOT}")

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"metadata.csv not found: {METADATA_CSV}")

    if not ASAP_ANNOTATIONS_JSON.exists():
        raise FileNotFoundError(f"asap_annotations.json not found: {ASAP_ANNOTATIONS_JSON}")

    if not MAESTRO_ROOT.exists():
        raise FileNotFoundError(f"maestro root not found: {MAESTRO_ROOT}")

    if not MUSESCORE_EXE.exists():
        raise FileNotFoundError(f"MuseScore not found: {MUSESCORE_EXE}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TOTAL_VALID_CSV.unlink(missing_ok=True)
    TOTAL_FAILURES_CSV.unlink(missing_ok=True)
    SAMPLED_PIECES_CSV.unlink(missing_ok=True)

    metadata = pd.read_csv(METADATA_CSV)
    metadata = metadata[metadata["audio_performance"].apply(has_audio_value)].copy()
    sampled_folders = sample_piece_folders(metadata)
    metadata = metadata[metadata["folder"].astype(str).isin(sampled_folders)].copy()
    metadata = metadata.reset_index(drop=True)
    rows = metadata.to_dict("records")
    reset_sample_output_files(rows)

    sampled_pieces = (
        metadata[["composer", "title", "folder"]]
        .drop_duplicates()
        .sort_values(["composer", "title", "folder"])
        .reset_index(drop=True)
    )
    sampled_pieces.to_csv(SAMPLED_PIECES_CSV, index=False)

    with open(ASAP_ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    wrote_valid_header = False
    wrote_failure_header = False
    success_count = 0
    failure_count = 0
    total_valid_rows = 0
    progress = tqdm(total=len(rows), desc="pipeline2", unit="sample")
    try:
        for batch_start in range(0, len(rows), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(rows))
            batch_rows = rows[batch_start:batch_end]

            future_to_index = {}
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for offset, row in enumerate(batch_rows):
                    index = batch_start + offset
                    future = executor.submit(process_sample, index, len(rows), row, annotations)
                    future_to_index[future] = index

                for future in as_completed(future_to_index):
                    index, row, result = future.result()
                    valid_df, paths, _context, failed_stage, error = result

                    if error is not None:
                        failure_row = build_failure_row(row, paths, failed_stage, error)
                        wrote_failure_header, _ = append_failure_row(
                            failure_row,
                            TOTAL_FAILURES_CSV,
                            wrote_failure_header,
                        )
                        failure_count += 1
                        del failure_row
                        del result
                        del paths
                        progress.update(1)
                        gc.collect()
                        continue

                    wrote_valid_header, added_valid_rows = append_valid_frame(
                        valid_df,
                        TOTAL_VALID_CSV,
                        wrote_valid_header,
                    )
                    success_count += 1
                    total_valid_rows += added_valid_rows
                    del valid_df
                    del result
                    del paths
                    progress.update(1)
                    gc.collect()

            del batch_rows
            del future_to_index
            gc.collect()
    finally:
        progress.close()

    if not wrote_valid_header:
        pd.DataFrame(columns=TOTAL_VALID_COLUMNS).to_csv(TOTAL_VALID_CSV, index=False)

    if not wrote_failure_header:
        pd.DataFrame(columns=TOTAL_FAILURE_COLUMNS).to_csv(TOTAL_FAILURES_CSV, index=False)

    print(f"saved: {TOTAL_VALID_CSV}")
    print(f"saved: {TOTAL_FAILURES_CSV}")
    print(f"saved: {SAMPLED_PIECES_CSV}")
    print(f"success_count: {success_count}")
    print(f"failure_count: {failure_count}")
    print(f"total_valid_rows: {total_valid_rows}")


if __name__ == "__main__":
    main()
