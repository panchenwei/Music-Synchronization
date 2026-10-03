"""Work-isolated boosted-tree controls on the unchanged phrase-start task."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .three_round_round2 import dataset, normalizer, split_ids
from .relation_interval_study import sha, read, write
from .local_context_study import metrics
from .phase2_models import choose_single_threshold
from .models import Normalizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/tabular_context_study'
ART = ROOT / 'artifacts/tabular_context_study'
GRID = np.arange(.1, .91, .05).round(2).tolist()
KINDS = {'H0': 0, 'H2': 2}
CAP = 1800.


def context(x, radius):
    """Offset-major concatenation; zero padding in train-normalized space."""
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or len(x) == 0 or radius not in (0, 2):
        raise ValueError('Expected nonempty [beats,features], radius 0 or 2')
    padded = np.pad(x, ((radius, radius), (0, 0)))
    return np.concatenate([padded[i:i+len(x)] for i in range(2*radius+1)], axis=1)


def train_arrays(data, norm, radius):
    xs, ys, ws, rows = [], [], [], []
    for pid, item in sorted(data.items()):
        mask = item['label_mask'].astype(bool)
        count = int(mask.sum()) * len(item['curves'])
        assert count > 0
        for curve in item['curves']:
            xs.append(context(norm.apply(curve), radius)[mask])
            ys.append(item['labels'][mask].astype(np.int64))
            ws.append(np.full(int(mask.sum()), 1./count, np.float64))
        rows.append(dict(piece_id=pid, rows=count, performances=len(item['curves']), distinct_valid_beats=int(mask.sum()), base_weight_sum=1.))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    assert np.isfinite(x).all() and set(np.unique(y)) == {0, 1}
    ratio = min(float(w[y == 0].sum() / w[y == 1].sum()), 10.)
    w *= np.where(y == 1, ratio, 1.)
    w /= w.mean()
    return x, y, w, rows, ratio


def raw_predictions(model, data, norm, radius):
    result = {}
    for pid, item in sorted(data.items()):
        x = np.concatenate([context(norm.apply(curve), radius) for curve in item['curves']])
        p = np.concatenate([model.predict_proba(x[i:i+65536])[:, 1] for i in range(0, len(x), 65536)])
        p = p.reshape(len(item['curves']), len(item['labels']))
        result[pid] = {str(perf): prob for perf, prob in zip(item['performance_ids'], p)}
    return result


def prepare():
    for p in (OUT, ART / 'models', ART / 'metrics'):
        p.mkdir(parents=True, exist_ok=True)
    assert read(ROOT / 'reports/relation_interval_study/completion_audit.json')['status'] == 'complete'
    hashes = dict(read(ROOT / 'reports/relation_interval_study/contract.json')['hashes'])
    assert all(sha(p) == h for p, h in hashes.items())
    for p in (Path(__file__), ROOT / 'tests/test_tabular_context.py', OUT / 'PROTOCOL.md', ROOT / 'artifacts/relation_interval_study/summary.csv'):
        hashes[str(p)] = sha(p)
    contract = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if (OUT / 'contract.json').exists():
        assert read(OUT / 'contract.json')['contract'] == contract
    write(OUT / 'contract.json', dict(contract=contract, hashes=hashes, sklearn_version=sklearn.__version__))
    if not (OUT / 'STATE.json').exists():
        write(OUT / 'STATE.json', dict(status='ready', completed=[], seconds=0., contract=contract, pid=None))
    return contract


def run_one(kind, fold, seed, contract):
    run = f'{kind}_seed{seed}_fold{fold}'
    dest = ART / 'models' / run
    dest.mkdir(exist_ok=True)
    result_path = ART / 'metrics' / f'{run}.json'
    if result_path.exists():
        assert read(result_path)['contract'] == contract
        print('CACHED', run, flush=True)
        return
    ids = split_ids(fold)
    assert not set(ids['train']) & set(ids['validation'])
    manifest = pd.read_csv(ROOT / 'artifacts/phase2/splits/opus_split_manifest.csv')
    part = manifest[manifest.fold == fold]
    assert not set(part[part.split == 'train'].opus) & set(part[part.split == 'validation'].opus)
    state = read(OUT / 'STATE.json')
    state.update(status='running', current_run=run, pid=os.getpid())
    write(OUT / 'STATE.json', state)
    start = time.monotonic()
    train, val = dataset(ids['train'], 'B'), dataset(ids['validation'], 'B')
    norm = normalizer(train)
    radius = KINDS[kind]
    x, y, w, counts, ratio = train_arrays(train, norm, radius)
    pd.DataFrame(counts).to_csv(OUT / f'{run}_training_counts.csv', index=False)
    model = HistGradientBoostingClassifier(learning_rate=.05, max_iter=50, max_leaf_nodes=7, max_depth=3, min_samples_leaf=100, l2_regularization=10., max_bins=127, early_stopping=False, warm_start=True, random_state=seed)
    history, best, step, prior = [], -1., 0, 0.
    latest = dest / 'latest.joblib'
    if latest.exists():
        s = joblib.load(latest)
        assert (s['contract'], s['kind'], s['seed'], s['fold']) == (contract, kind, seed, fold)
        np.testing.assert_array_equal(s['mean'], norm.mean)
        np.testing.assert_array_equal(s['std'], norm.std)
        model, history, best, step, prior = s['model'], s['history'], s['best'], s['step'], s['seconds']
    def snapshot():
        return dict(model=model, history=history, best=best, step=step, seconds=prior+time.monotonic()-start, contract=contract, kind=kind, fold=fold, seed=seed, mean=norm.mean, std=norm.std, positive_weight=ratio, training_rows=len(y))
    for step in (s for s in (50, 100, 150) if s > step):
        if state['seconds'] + prior + time.monotonic()-start > CAP:
            raise TimeoutError('1800 second round budget; preserve latest checkpoints')
        model.set_params(max_iter=step)
        model.fit(x, y, sample_weight=w)
        assert model.n_iter_ == step and not model.do_early_stopping_
        raw = raw_predictions(model, val, norm, radius)
        threshold, _ = choose_single_threshold(raw, val, GRID)
        _, _, score = metrics(raw, val, threshold)
        history.append(dict(step=step, **score))
        if score['macro_f1_tol1'] > best + 1e-9:
            best = score['macro_f1_tol1']
            joblib.dump(snapshot(), dest / 'best.joblib')
        joblib.dump(snapshot(), latest)
        print(run, step, 'F1', round(score['macro_f1_tol1'], 4), 'exact', round(score['macro_f1_tol0'], 4), 'AP', round(score['raw_ap'], 4), flush=True)
    saved = joblib.load(dest / 'best.joblib')
    model = saved['model']
    threshold = [h['threshold'] for h in saved['history'] if h['step'] == saved['step']][0]
    raw = raw_predictions(model, val, norm, radius)
    perfs, pieces, score = metrics(raw, val, threshold)
    _, _, ts = metrics(raw_predictions(model, train, norm, radius), train, threshold)
    elapsed = prior + time.monotonic()-start
    result = dict(run_id=run, kind=kind, fold=fold, seed=seed, contract=contract, best_step=saved['step'], features=x.shape[1], training_rows=len(y), positive_weight=ratio, seconds=elapsed, train_f1=ts['macro_f1_tol1'], train_ap=ts['raw_ap'], gap=ts['macro_f1_tol1']-score['macro_f1_tol1'], **score)
    perfs.to_csv(ART / 'metrics' / f'{run}_performances.csv', index=False)
    pieces.to_csv(ART / 'metrics' / f'{run}_pieces.csv', index=False)
    rows = [(pid, perf, b, float(p), int(val[pid]['labels'][b]), int(val[pid]['label_mask'][b])) for pid, pp in raw.items() for perf, arr in pp.items() for b, p in enumerate(arr)]
    pd.DataFrame(rows, columns=['piece_id', 'performance_id', 'beat', 'probability', 'label', 'valid']).to_csv(ART / 'metrics' / f'{run}_predictions.csv.gz', index=False)
    write(result_path, result)
    state['completed'].append(run)
    state['seconds'] += elapsed
    state.update(status='between_runs')
    write(OUT / 'STATE.json', state)


def report():
    df = pd.DataFrame([read(p) for p in (ART / 'metrics').glob('*_fold*.json')])
    assert len(df) == 8
    df.to_csv(ART / 'summary.csv', index=False)
    cols = ['macro_f1_tol0', 'macro_f1_tol1', 'raw_ap', 'macro_precision_tol1', 'macro_recall_tol1', 'train_f1', 'gap', 'seconds']
    means = df.groupby('kind')[cols].mean()
    means.to_csv(OUT / 'model_means.csv')
    b = pd.read_csv(ROOT / 'artifacts/relation_interval_study/summary.csv')
    b = b[b.kind == 'B'].set_index(['fold', 'seed'])
    rows = []
    for kind in KINDS:
        a = df[df.kind == kind].set_index(['fold', 'seed'])
        d = a[cols[:3]] - b[cols[:3]]
        rows.append(dict(candidate=kind, f1_delta=d.macro_f1_tol1.mean(), exact_delta=d.macro_f1_tol0.mean(), ap_delta=d.raw_ap.mean(), f1_positive=int((d.macro_f1_tol1 > 0).sum()), ap_positive=int((d.raw_ap > 0).sum()), promotion=bool(d.macro_f1_tol1.mean() >= .015 and d.macro_f1_tol0.mean() > 0 and d.raw_ap.mean() > 0 and (d.macro_f1_tol1 > 0).sum() >= 3 and (d.raw_ap > 0).sum() >= 3)))
    pd.DataFrame(rows).to_csv(OUT / 'comparisons.csv', index=False)
    assert all(sha(p) == h for p, h in read(OUT / 'contract.json')['hashes'].items())
    state = read(OUT / 'STATE.json')
    state.update(status='training_complete', pid=None)
    write(OUT / 'STATE.json', state)
    print('MEANS\n' + means.to_string(), flush=True)
    print('COMPARISONS\n' + pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'all'], default='all')
    args = parser.parse_args()
    try:
        contract = prepare()
        if args.stage == 'prepare':
            print(contract)
            return
        with threadpool_limits(limits=2):
            for fold in (0, 1):
                for seed in (42, 43):
                    for kind in KINDS:
                        run_one(kind, fold, seed, contract)
            report()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / 'failures.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.time(), traceback=traceback.format_exc()))+'\n')
        if (OUT / 'STATE.json').exists():
            state = read(OUT / 'STATE.json')
            state.update(status='failed', pid=None)
            write(OUT / 'STATE.json', state)
        raise


if __name__ == '__main__':
    main()
