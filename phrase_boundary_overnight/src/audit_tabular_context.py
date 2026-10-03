"""Post-round metric replay, data contract checks and fixed diagnostic comparison."""
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .tabular_context_study import ROOT, OUT, ART, KINDS, dataset, split_ids, normalizer, raw_predictions, read, write, sha, Normalizer
from .local_context_study import metrics
from .phase2_models import nms_probabilities


def load_raw(path, data):
    frame = pd.read_csv(path)
    assert set(frame.piece_id) == set(data)
    raw = {}
    for (pid, perf), g in frame.groupby(['piece_id', 'performance_id']):
        g = g.sort_values('beat')
        np.testing.assert_array_equal(g.beat, np.arange(len(data[pid]['labels'])))
        np.testing.assert_array_equal(g.label, data[pid]['labels'])
        np.testing.assert_array_equal(g.valid, data[pid]['label_mask'])
        assert np.isfinite(g.probability).all() and g.probability.between(0, 1).all()
        raw.setdefault(pid, {})[perf] = g.probability.to_numpy()
    for pid in data:
        assert set(raw[pid]) == set(data[pid]['performance_ids'].astype(str))
    return raw


def main():
    started = time.monotonic()
    state = read(OUT / 'STATE.json')
    assert state['status'] in ('training_complete', 'complete')
    results = pd.read_csv(ART / 'summary.csv')
    assert len(results) == 8
    assert set(zip(results.kind, results.fold, results.seed)) == {(k, f, s) for k in KINDS for f in (0, 1) for s in (42, 43)}
    hashes = read(OUT / 'contract.json')['hashes']
    assert all(sha(p) == h for p, h in hashes.items())
    before = {str(p): sha(p) for p in (ART / 'models').glob('*/*.joblib')}
    assert len(before) == 16
    rows, histories, disagreements = [], [], []
    baseline = pd.read_csv(ROOT / 'artifacts/relation_interval_study/summary.csv')
    with threadpool_limits(limits=2):
        for r in results.to_dict('records'):
            run = r['run_id']
            ids = split_ids(int(r['fold']))
            data = dataset(ids['validation'], 'B')
            raw = load_raw(ART / 'metrics' / f'{run}_predictions.csv.gz', data)
            _, _, recomputed = metrics(raw, data, r['threshold'])
            error = max(abs(recomputed[c]-r[c]) for c in ('macro_f1_tol0', 'macro_f1_tol1', 'macro_f1_tol2', 'raw_ap'))
            assert error < 1e-10
            saved = joblib.load(ART / 'models' / run / 'best.joblib')
            latest = joblib.load(ART / 'models' / run / 'latest.joblib')
            assert (saved['contract'], saved['kind'], saved['fold'], saved['seed'], saved['step']) == (state['contract'], r['kind'], r['fold'], r['seed'], r['best_step'])
            assert latest['step'] == 150 and latest['contract'] == state['contract']
            model = saved['model']
            assert not model.do_early_stopping_ and model.n_iter_ == r['best_step']
            assert model.n_features_in_ == r['features']
            norm = Normalizer(saved['mean'], saved['std'])
            train = dataset(ids['train'], 'B')
            fitted = normalizer(train)
            np.testing.assert_array_equal(norm.mean, fitted.mean)
            np.testing.assert_array_equal(norm.std, fitted.std)
            replay = raw_predictions(model, data, norm, KINDS[r['kind']])
            delta = max(float(np.max(abs(p-replay[pid][perf]))) for pid, pp in raw.items() for perf, p in pp.items())
            assert delta < 1e-12
            _, _, ts = metrics(raw_predictions(model, train, norm, KINDS[r['kind']]), train, r['threshold'])
            assert abs(ts['macro_f1_tol1']-r['train_f1']) < 1e-10
            for h in latest['history']:
                histories.append(dict(run_id=run, kind=r['kind'], fold=r['fold'], seed=r['seed'], **h))
            rows.append(dict(run_id=run, kind=r['kind'], metric_error=error, replay_error=delta, train_f1=ts['macro_f1_tol1'], validation_f1=r['macro_f1_tol1'], gap=r['gap'], all_validation_performances_replayed=True))
            pd.DataFrame(rows).to_csv(OUT / 'run_audit.csv', index=False)
            if r['kind'] == 'H2':
                base = baseline[(baseline.kind == 'B') & (baseline.fold == r['fold']) & (baseline.seed == r['seed'])].iloc[0]
                braw = load_raw(ROOT / 'artifacts/relation_interval_study/metrics' / f"{base.run_id}_predictions.csv.gz", data)
                for pid, pp in raw.items():
                    for perf, p in pp.items():
                        mask = data[pid]['label_mask'].astype(bool)
                        truth = np.flatnonzero((data[pid]['labels'] > .5) & mask)
                        a = np.flatnonzero((nms_probabilities(p) >= r['threshold']) & mask)
                        b = np.flatnonzero((nms_probabilities(braw[pid][perf]) >= base.threshold) & mask)
                        # Diagnostic proximity coverage, explicitly NOT one-to-one F1.
                        ah = np.array([np.any(abs(a-t) <= 1) for t in truth])
                        bh = np.array([np.any(abs(b-t) <= 1) for t in truth])
                        disagreements.append(dict(run_id=run, piece_id=pid, performance_id=perf, truth=len(truth), both=int((ah & bh).sum()), tree_only=int((ah & ~bh).sum()), cnn_only=int((bh & ~ah).sum()), neither=int((~ah & ~bh).sum())))
            print('AUDITED', run, 'F1', r['macro_f1_tol1'], 'gap', r['gap'], flush=True)
    assert before == {str(p): sha(p) for p in (ART / 'models').glob('*/*.joblib')}
    assert all(sha(p) == h for p, h in hashes.items())
    pd.DataFrame(histories).to_csv(OUT / 'training_history.csv', index=False)
    pd.DataFrame(disagreements).to_csv(OUT / 'boundary_coverage_diagnostic.csv', index=False)
    write(OUT / 'completion_audit.json', dict(status='complete', runs=8, model_files=16, metric_recomputations=8, full_validation_model_replays=8, full_training_metric_replays=8, normalizers_train_only_rebuilt=True, source_and_model_hashes_unchanged=True, test_predictions_accessed=False, audit_seconds=time.monotonic()-started, boundary_coverage_is_not_one_to_one_f1=True))
    state.update(status='complete', pid=None)
    write(OUT / 'STATE.json', state)


if __name__ == '__main__':
    main()
