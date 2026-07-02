from pathlib import Path
import subprocess

from map_beats_with_warping_path import map_beats_with_warping_path
from parse_score_beats_from_midi import parse_beats_from_midi
from sync_audio_pair import run_highres_sync
from valid import run_valid


SRC_DIR = Path(__file__).resolve().parent
EXAMPLE_NAME = "bwv_856"
EXAMPLE_DIR = SRC_DIR / EXAMPLE_NAME

# MuseScore 导出用的。
MUSESCORE_EXE = Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")

# 这组是当前样例自己的输入路径。
XML_SCORE = Path(r"C:\Users\pa1018\Desktop\Music Synchronization\dataset\asap-dataset\Bach\Prelude\bwv_856\xml_score.musicxml")
SCORE_MIDI = EXAMPLE_DIR / "score.mid"
SCORE_WAV = EXAMPLE_DIR / "score_synth.wav"
PERFORMANCE_WAV = EXAMPLE_DIR / "performance.wav"
ASAP_ANNOTATIONS_JSON = EXAMPLE_DIR / "asap_annotations.json"
PERFORMANCE_ANNOTATIONS_TXT = Path(r"C:\Users\pa1018\Desktop\Music Synchronization\dataset\asap-dataset\Bach\Prelude\bwv_856\LuoJ01M_annotations.txt")

# sync_audio_pair.py 用的参数。
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

# 各阶段输出文件。
SCORE_BEATS_CSV = EXAMPLE_DIR / "score_beats_from_midi.csv"
WARPING_PATH_CSV = EXAMPLE_DIR / "warping_path_highres.csv"
PREDICTED_BEATS_CSV = EXAMPLE_DIR / "predicted_beats.csv"
ALIGNMENT_BEATS_JSON = EXAMPLE_DIR / "alignment_beats.json"
VALID_CSV = EXAMPLE_DIR / "valid.csv"
VALID_METRICS_JSON = EXAMPLE_DIR / "valid_metrics.json"


def run_musescore_export(src_path: Path, out_path: Path):
    print("  src =", src_path)
    print("  out =", out_path)
    subprocess.run(
        [str(MUSESCORE_EXE), "-o", str(out_path), str(src_path)],
        check=True,
    )


def main():
    print("Using paths:")
    print("  example_dir            =", EXAMPLE_DIR)
    print("  musescore_exe          =", MUSESCORE_EXE)
    print("  xml_score              =", XML_SCORE)
    print("  score_midi             =", SCORE_MIDI)
    print("  score_wav              =", SCORE_WAV)
    print("  performance_wav        =", PERFORMANCE_WAV)
    print("  asap_annotations_json  =", ASAP_ANNOTATIONS_JSON)
    print("  performance_annotations_txt =", PERFORMANCE_ANNOTATIONS_TXT)
    print("  sync_fs                =", SYNC_FS)
    print("  sync_feature_rate      =", SYNC_FEATURE_RATE)
    print("  sync_step_weights      =", SYNC_STEP_WEIGHTS)
    print("  sync_threshold_rec     =", SYNC_THRESHOLD_REC)
    print("  boundary_hop_length    =", BOUNDARY_HOP_LENGTH)
    print("  boundary_margin_sec    =", BOUNDARY_MARGIN_SEC)
    print("  boundary_query_fraction =", BOUNDARY_QUERY_FRACTION)
    print("  boundary_min_query_sec =", BOUNDARY_MIN_QUERY_SEC)
    print("  boundary_max_query_sec =", BOUNDARY_MAX_QUERY_SEC)
    print("  boundary_min_crop_ratio =", BOUNDARY_MIN_CROP_RATIO)
    print("  max_boundary_cost     =", MAX_BOUNDARY_COST)

    if not MUSESCORE_EXE.exists():
        raise FileNotFoundError(f"MuseScore not found: {MUSESCORE_EXE}")

    if not XML_SCORE.exists():
        raise FileNotFoundError(f"xml score not found: {XML_SCORE}")

    if not PERFORMANCE_WAV.exists():
        raise FileNotFoundError(f"performance wav not found: {PERFORMANCE_WAV}")

    if not ASAP_ANNOTATIONS_JSON.exists():
        raise FileNotFoundError(f"asap_annotations.json not found: {ASAP_ANNOTATIONS_JSON}")

    if not PERFORMANCE_ANNOTATIONS_TXT.exists():
        raise FileNotFoundError(f"performance annotations txt not found: {PERFORMANCE_ANNOTATIONS_TXT}")

    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] xml -> midi")
    run_musescore_export(
        src_path=XML_SCORE,
        out_path=SCORE_MIDI,
    )

    print("\n[2/6] midi -> wav")
    run_musescore_export(
        src_path=SCORE_MIDI,
        out_path=SCORE_WAV,
    )

    print("\n[3/6] parse score beats")
    parse_beats_from_midi(
        score_midi=SCORE_MIDI,
        out_csv=SCORE_BEATS_CSV,
    )

    print("\n[4/6] sync audio pair")
    run_highres_sync(
        score_wav=SCORE_WAV,
        performance_wav=PERFORMANCE_WAV,
        out_dir=EXAMPLE_DIR,
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
    )

    print("\n[5/6] map beats")
    map_beats_with_warping_path(
        score_beats_csv=SCORE_BEATS_CSV,
        warping_csv=WARPING_PATH_CSV,
        out_predicted_csv=PREDICTED_BEATS_CSV,
        out_alignment_json=ALIGNMENT_BEATS_JSON,
    )

    print("\n[6/6] validate")
    run_valid(
        predicted_beats_csv=PREDICTED_BEATS_CSV,
        asap_annotations_json=ASAP_ANNOTATIONS_JSON,
        out_csv=VALID_CSV,
        out_json=VALID_METRICS_JSON,
        performance_annotations_txt=PERFORMANCE_ANNOTATIONS_TXT,
    )


if __name__ == "__main__":
    main()
