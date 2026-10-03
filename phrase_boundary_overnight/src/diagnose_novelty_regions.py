"""Post-hoc error decomposition; keeps full-piece matching and all labels."""
import numpy as np
import pandas as pd
from .score_novelty_study import ART, OUT, read, write
from .phase2_models import nms_probabilities
from .evaluation import one_to_one_counts


def decompose(probability, labels, valid, threshold):
    predicted = np.flatnonzero((nms_probabilities(probability) >= threshold) & valid)
    truth = np.flatnonzero((labels > .5) & valid)
    candidates = sorted((abs(int(p)-int(t)), int(p), int(t)) for p in predicted for t in truth if abs(p-t) <= 1)
    used_p, used_t = set(), set()
    for _, p, t in candidates:
        if p not in used_p and t not in used_t:
            used_p.add(p)
            used_t.add(t)
    expected = one_to_one_counts(predicted, truth, 1)
    assert (len(used_p), len(predicted)-len(used_p), len(truth)-len(used_t)) == (expected.tp, expected.fp, expected.fn)
    def edge(i):
        return i < 24 or i > len(labels)-24
    rows = []
    for region in ('edge', 'interior'):
        wanted = region == 'edge'
        rows.append(dict(region=region, valid_beats=sum(edge(i) == wanted for i in np.flatnonzero(valid)), truth=sum(edge(t) == wanted for t in truth), tp=sum(edge(t) == wanted for t in used_t), fn=sum(edge(t) == wanted for t in set(truth)-used_t), fp=sum(edge(p) == wanted for p in set(predicted)-used_p)))
    assert sum(r['tp'] for r in rows) == expected.tp
    assert sum(r['fp'] for r in rows) == expected.fp
    assert sum(r['fn'] for r in rows) == expected.fn
    return rows


def main():
    assert read(OUT / 'completion_audit.json')['status'] == 'complete'
    results = pd.read_csv(ART / 'summary.csv')
    rows = []
    for r in results.to_dict('records'):
        frame = pd.read_csv(ART / 'metrics' / f"{r['run_id']}_predictions.csv.gz")
        for (pid, perf), g in frame.groupby(['piece_id', 'performance_id']):
            g = g.sort_values('beat')
            for counts in decompose(g.probability.to_numpy(), g.label.to_numpy(), g.valid.to_numpy().astype(bool), r['threshold']):
                rows.append(dict(run_id=r['run_id'], kind=r['kind'], fold=r['fold'], seed=r['seed'], piece_id=pid, performance_id=perf, **counts))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'region_error_counts.csv', index=False)
    total = df.groupby(['kind', 'region'])[['truth', 'tp', 'fp', 'fn', 'valid_beats']].sum()
    total.to_csv(OUT / 'region_error_totals.csv')
    write(OUT / 'region_diagnostic.json', dict(status='complete', post_hoc=True, definition='edge iff largest 24-beat half-window lacks full support; else interior', full_piece_matching_before_region_assignment=True, label_filtering=False, new_training=0, test_used=False, warning='Totals count repeated performances and seeds, not independent musical boundaries. TP assigned by truth location, FP by unmatched prediction; regional precision is not a separate primary metric.'))
    print(total.to_string())


if __name__ == '__main__':
    main()
