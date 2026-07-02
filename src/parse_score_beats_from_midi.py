from pathlib import Path
from fractions import Fraction

import mido
import pandas as pd


def read_midi_timeline(score_midi: Path):
    """把 MIDI 里的时间线拆出来。"""

    mid = mido.MidiFile(score_midi)
    ticks_per_beat = mid.ticks_per_beat

    merged = mido.merge_tracks(mid.tracks)

    abs_tick = 0
    end_tick = 0

    # 没写 tempo 就按 120 BPM。
    tempo_by_tick = {0: 500000}

    # 没写拍号就先当 4/4。
    ts_by_tick = {0: (4, 4)}

    for msg in merged:
        abs_tick += msg.time
        end_tick = max(end_tick, abs_tick)

        if msg.is_meta and msg.type == "set_tempo":
            tempo_by_tick[abs_tick] = msg.tempo

        if msg.is_meta and msg.type == "time_signature":
            ts_by_tick[abs_tick] = (msg.numerator, msg.denominator)

    tempo_events = sorted(tempo_by_tick.items(), key=lambda x: x[0])

    time_signature_events = [
        (tick, numerator, denominator)
        for tick, (numerator, denominator) in sorted(ts_by_tick.items(), key=lambda x: x[0])
    ]

    return ticks_per_beat, end_tick, tempo_events, time_signature_events


def tick_to_seconds(target_tick: int, tempo_events, ticks_per_beat: int) -> float:
    """把 tick 换成秒，tempo 变化也算进去。"""

    total_sec = 0.0
    last_tick = 0
    current_tempo = tempo_events[0][1]

    for event_tick, event_tempo in tempo_events[1:]:
        if target_tick <= event_tick:
            break

        delta_ticks = event_tick - last_tick
        total_sec += mido.tick2second(delta_ticks, ticks_per_beat, current_tempo)

        last_tick = event_tick
        current_tempo = event_tempo

    delta_ticks = target_tick - last_tick
    total_sec += mido.tick2second(delta_ticks, ticks_per_beat, current_tempo)

    return float(total_sec)


def parse_beats_from_midi(score_midi: Path, out_csv: Path) -> pd.DataFrame:
    """从 score.mid 拉一张 beat 表出来。"""

    ticks_per_beat, end_tick, tempo_events, ts_events = read_midi_timeline(score_midi)

    midi_duration_sec = tick_to_seconds(
        target_tick=end_tick,
        tempo_events=tempo_events,
        ticks_per_beat=ticks_per_beat,
    )

    print("[MIDI]")
    print("  score_midi:", score_midi)
    print("  ticks_per_beat:", ticks_per_beat)
    print("  end_tick:", end_tick)
    print("  midi_duration_sec:", round(midi_duration_sec, 5))
    print("  tempo_events:", tempo_events[:10])
    print("  time_signature_events:", ts_events)

    rows = []
    beat_index = 1
    global_measure_number = 1

    for i, (ts_tick, numerator, denominator) in enumerate(ts_events):
        segment_start = ts_tick
        segment_end = ts_events[i + 1][0] if i + 1 < len(ts_events) else end_tick

        # 分母变了，每拍占的 tick 也跟着变。
        beat_ticks = Fraction(ticks_per_beat * 4, denominator)
        measure_ticks = beat_ticks * numerator

        local_beat = 0
        tick_fraction = Fraction(segment_start, 1)

        while tick_fraction <= segment_end:
            score_tick = int(round(float(tick_fraction)))

            beat_in_measure = int(local_beat % numerator) + 1
            measure_number_guess = global_measure_number + int(local_beat // numerator)

            score_time_sec = tick_to_seconds(
                target_tick=score_tick,
                tempo_events=tempo_events,
                ticks_per_beat=ticks_per_beat,
            )

            rows.append(
                {
                    "beat_index": beat_index,
                    "measure_number_guess": measure_number_guess,
                    "beat_in_measure": beat_in_measure,
                    "numerator": numerator,
                    "denominator": denominator,
                    "score_tick": score_tick,
                    "score_time_sec": score_time_sec,
                }
            )

            beat_index += 1
            local_beat += 1
            tick_fraction += beat_ticks

        measures_in_segment = int(Fraction(segment_end - segment_start, 1) // measure_ticks)
        global_measure_number += measures_in_segment

    df = pd.DataFrame(rows)
    df = df[df["score_tick"] <= end_tick].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.5f")

    print("[saved]", out_csv)
    print()
    print("[head]")
    print(df.head())
    print()
    print("[tail]")
    print(df.tail())

    return df


def main():
    src_dir = Path(__file__).resolve().parent
    example_name = "bwv_856"
    example_dir = src_dir / example_name

    score_midi = example_dir / "score.mid"
    out_csv = example_dir / "score_beats_from_midi.csv"

    if not score_midi.exists():
        raise FileNotFoundError(f"score midi not found: {score_midi}")

    print("Using paths:")
    print("  score_midi =", score_midi)
    print("  out_csv    =", out_csv)

    parse_beats_from_midi(
        score_midi=score_midi,
        out_csv=out_csv,
    )


if __name__ == "__main__":
    main()
