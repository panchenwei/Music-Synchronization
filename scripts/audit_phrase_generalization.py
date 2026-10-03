"""Read-only frozen holdout diagnostic; never tunes models or decoder settings."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'phrase_boundary_overnight/reports/mentor_continuation_20260916/locked_holdout_test'


def main():
    rows = list(csv.DictReader((SOURCE / 'primary_per_work.csv').open(encoding='utf-8-sig')))
    summary = json.loads((SOURCE / 'primary_summary.json').read_text(encoding='utf-8'))
    assert len(rows) == summary['works'] == 17
    mean = sum(float(r['f1_tol1']) for r in rows) / len(rows)
    assert abs(mean - summary['macro_f1']) < 1e-10
    groups = {}
    for r in rows:
        groups.setdefault(r['opus'], []).append(r)
    result = []
    for opus, works in sorted(groups.items(), key=lambda x: int(x[0])):
        def avg(key):
            return sum(float(r[key]) for r in works) / len(works)
        result.append(dict(opus=opus, works=len(works), f1=avg('f1_tol1'),
                           exact_f1=avg('f1_tol0'), raw_ap=avg('raw_ap'),
                           prediction_to_truth_ratio=avg('predicted_boundaries') / avg('true_boundaries')))
    sampler = ROOT / 'phrase_boundary_overnight/src/mentor_sequence_models.py'
    print(json.dumps(dict(
        status='reproduced', new_training=0, model_selection=False,
        macro_f1=mean, groups=result,
        f1_group_range=max(r['f1'] for r in result)-min(r['f1'] for r in result),
        interpretation='Group variation is observed; causal distribution shift is not established. Frozen holdout cannot guide parameter tuning.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [SOURCE/'primary_per_work.csv', SOURCE/'primary_summary.json', sampler]},
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
