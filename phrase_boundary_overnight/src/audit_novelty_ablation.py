"""Independent post-training replay; never trains or selects test predictions."""
import time
import numpy as np
import pandas as pd
import torch
from .score_novelty_ablation import OUT, ART, read, write, sha, dataset, make_model, split_ids
from .models import Normalizer
from .local_context_study import metrics, predictions


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    state = read(OUT / 'STATE.json')
    assert state['status'] in ('training_complete', 'complete')
    results = pd.read_csv(ART / 'summary.csv')
    expected = {(k, f, s) for k in ('V', 'M') for f in (0, 1) for s in (42, 43)}
    assert len(results) == 8 and set(zip(results.kind, results.fold, results.seed)) == expected
    hashes = read(OUT / 'contract.json')['hashes']
    assert all(sha(p) == h for p, h in hashes.items())
    before = {str(p): sha(p) for p in (ART / 'checkpoints').glob('*/*.pt')}
    assert len(before) == 16
    rows, history = [], []
    for r in results.to_dict('records'):
        run = r['run_id']
        ids = split_ids(int(r['fold']))
        assert not set(ids['train']) & set(ids['validation'])
        data = dataset(ids['validation'], r['kind'])
        frame = pd.read_csv(ART / 'metrics' / f'{run}_predictions.csv.gz')
        assert set(frame.piece_id) == set(ids['validation'])
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
        _, _, score = metrics(raw, data, r['threshold'])
        error = max(abs(score[c] - r[c]) for c in ('macro_f1_tol0', 'macro_f1_tol1', 'macro_f1_tol2', 'raw_ap'))
        assert error < 1e-10
        saved = torch.load(ART / 'checkpoints' / run / 'best.pt', map_location='cpu', weights_only=False)
        assert (saved['contract'], saved['kind'], saved['fold'], saved['seed'], saved['step']) == (state['contract'], r['kind'], r['fold'], r['seed'], r['best_step'])
        model = make_model('P', int(r['seed'])).eval()
        model.load_state_dict(saved['model'])
        assert sum(p.numel() for p in model.parameters()) == 3297
        norm = Normalizer(saved['mean'], saved['std'])
        pid = sorted(data)[0]
        perf = sorted(raw[pid])[len(raw[pid]) // 2]
        pi = list(data[pid]['performance_ids'].astype(str)).index(perf)
        x = torch.from_numpy(norm.apply(data[pid]['curves'][pi:pi+1]).astype(np.float32))
        with torch.no_grad():
            replayed = torch.sigmoid(model(x))[0].numpy()
        delta = float(np.max(abs(replayed - raw[pid][perf])))
        assert delta < 2e-4
        _, _, a = metrics({pid: {perf: replayed}}, {pid: data[pid]}, r['threshold'])
        _, _, b = metrics({pid: {perf: raw[pid][perf]}}, {pid: data[pid]}, r['threshold'])
        assert abs(a['macro_f1_tol1'] - b['macro_f1_tol1']) < 1e-12
        latest = torch.load(ART / 'checkpoints' / run / 'latest.pt', map_location='cpu', weights_only=False)
        assert latest['step'] == 300 and latest['contract'] == state['contract']
        for h in latest['history']:
            history.append(dict(run_id=run, kind=r['kind'], fold=r['fold'], seed=r['seed'], **h))
        train = dataset(ids['train'], r['kind'])
        _, _, ts = metrics(predictions(model, train, norm, torch.device('cpu')), train, r['threshold'])
        rows.append(dict(run_id=run, kind=r['kind'], fold=r['fold'], seed=r['seed'], metric_error=error, replay_error=delta, train_f1=ts['macro_f1_tol1'], validation_f1=r['macro_f1_tol1'], gap=ts['macro_f1_tol1']-r['macro_f1_tol1'], train_ap=ts['raw_ap'], validation_ap=r['raw_ap']))
        pd.DataFrame(rows).to_csv(OUT / 'run_audit.csv', index=False)
        print('AUDITED', run, 'F1', r['macro_f1_tol1'], 'gap', rows[-1]['gap'], flush=True)
    assert before == {str(p): sha(p) for p in (ART / 'checkpoints').glob('*/*.pt')}
    assert all(sha(p) == h for p, h in hashes.items())
    bridge = read(OUT / 'baseline_reproduction.json')
    assert bridge['passed'] and bridge['max_error'] < 1e-6
    pd.DataFrame(history).to_csv(OUT / 'training_history.csv', index=False)
    write(OUT / 'completion_audit.json', dict(status='complete', runs=8, checkpoints=16, all_metric_recomputations_pass=True, checkpoint_replays=8, train_gap_runs=8, source_and_checkpoint_hashes_unchanged=True, test_predictions_accessed=False, baseline_control_reused=True, training_runs=0, audit_seconds=time.monotonic()-started))
    state.update(status='complete', pid=None)
    write(OUT / 'STATE.json', state)


if __name__ == '__main__':
    main()
