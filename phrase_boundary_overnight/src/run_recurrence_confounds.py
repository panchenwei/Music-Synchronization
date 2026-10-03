"""Frozen availability/correspondence controls for the completed recurrence study."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import score_context_study as engine
from . import run_motif_recurrence_study as prior
from .recurrence_confounds import control_features
from .label_repaired_rebaseline import dataset as base_data
from .phrase_end_auxiliary import ROOT, read, write, sha, normalizer, split_ids
from .local_context_study import metrics, predictions
from .audit_external_stem_transfer import checked_raw

OUT = ROOT/'reports/recurrence_confound_study'
ART = ROOT/'artifacts/recurrence_confound_study'
OLD = ROOT/'artifacts/motif_recurrence_study'
COLS = prior.COLS
FACTORY = engine.make_model


def make_model(kind, seed):
    # Original engine treats literal C as a wider convolution; explicitly prevent it.
    return FACTORY('P', seed)


def dataset(ids, kind):
    if kind in ('Z', 'O'):
        return prior.dataset(ids, kind)
    assert kind in ('A', 'C')
    data = base_data(ids, 'B')
    for pid, v in data.items():
        with np.load(ART/'cache'/f'{pid}.npz', allow_pickle=False) as z:
            feat = z[kind].copy()
        assert feat.shape == (len(v['labels']), 24)
        v['pitch_profiles'] = feat
        v['curves'][..., 34:] = feat[None]
    return data


def bind():
    engine.OUT = OUT; engine.ART = ART; engine.CAP = 1200.
    engine.dataset = dataset; engine.make_model = make_model


def prepare():
    began = time.monotonic()
    for p in (OUT, ART/'cache', ART/'metrics', ART/'checkpoints'):
        p.mkdir(parents=True, exist_ok=True)
    assert read(prior.OUT/'completion_audit.json')['status'] == 'complete'
    hashes = dict(read(prior.OUT/'contract.json')['hashes'])
    assert all(sha(p) == h for p, h in hashes.items())
    # Freeze the exact reused metrics, predictions and both checkpoints too.
    for kind in ('Z', 'O'):
        for f in (0, 1):
            for s in (42, 43):
                run = f'{kind}_seed{s}_fold{f}'
                for p in [OLD/'metrics'/f'{run}.json', OLD/'metrics'/f'{run}_predictions.csv.gz',
                          OLD/'checkpoints'/run/'best.pt', OLD/'checkpoints'/run/'latest.pt']:
                    hashes[str(p)] = sha(p)
    rows = []
    for source in sorted((ROOT/'artifacts/note_relation_study/cache').glob('*.npz')):
        assert time.monotonic()-began < 300
        with np.load(source, allow_pickle=False) as z:
            events = z['note_events'].copy(); n = int(z['n_beats'])
        c = control_features(events, n)
        with np.load(OLD/'cache'/source.name, allow_pickle=False) as z:
            o = z['O'].copy()
        np.testing.assert_array_equal(c[:, 3::4], o[:, 3::4])
        a = np.zeros_like(o); a[:, 3::4] = o[:, 3::4]
        assert c.shape == (n, 24) and np.isfinite(c).all() and ((c >= 0)&(c <= 1)).all()
        dest = ART/'cache'/source.name
        if dest.exists():
            with np.load(dest, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['A'], a); np.testing.assert_array_equal(z['C'], c)
        else:
            np.savez_compressed(dest, A=a, C=c)
        hashes[str(dest)] = sha(dest)
        rows.append(dict(piece_id=source.stem, beats=n, availability_equal=True,
                         ordered_control_mean_difference=float(abs(o-c).mean())))
    assert len(rows) == 43
    pd.DataFrame(rows).to_csv(OUT/'feature_audit.csv', index=False)
    checks = []
    for f in (0, 1):
        ids = split_ids(f)
        assert not set(ids['train'])&set(ids['validation'])
        manifest = pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv')
        part = manifest[manifest.fold == f]
        assert not set(part[part.split == 'train'].opus)&set(part[part.split == 'validation'].opus)
        old = base_data(ids['train'], 'B'); oldnorm = normalizer(old)
        for kind in ('A', 'C'):
            data = dataset(ids['train'], kind); norm = normalizer(data)
            np.testing.assert_array_equal(norm.mean[:34], oldnorm.mean[:34])
            np.testing.assert_array_equal(norm.std[:34], oldnorm.std[:34])
            for pid, v in data.items():
                for field in ('labels', 'label_mask'):
                    np.testing.assert_array_equal(v[field], old[pid][field])
                np.testing.assert_array_equal(v['curves'][..., :34], old[pid]['curves'][..., :34])
            checks.append(dict(fold=f, kind=kind, train=len(data), base_inputs_labels_masks_equal=True))
    pd.DataFrame(checks).to_csv(OUT/'input_audit.csv', index=False)
    for s in (42, 43):
        ref = FACTORY('O', s)
        for k in ('A', 'C'):
            m = make_model(k, s)
            assert not m.frontend.wide and sum(p.numel() for p in m.parameters()) == 3297
            for name, value in ref.state_dict().items():
                torch.testing.assert_close(value, m.state_dict()[name], rtol=0, atol=0)
    for p in (Path(__file__), ROOT/'src/recurrence_confounds.py',
              ROOT/'tests/test_recurrence_confounds.py', OUT/'PROTOCOL.md'):
        hashes[str(p)] = sha(p)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():
        assert read(OUT/'contract.json')['contract'] == digest
    write(OUT/'contract.json', dict(contract=digest, hashes=hashes))
    done = [read(p) for p in (ART/'metrics').glob('*_fold*.json')]
    assert all(r['contract'] == digest for r in done)
    write(OUT/'STATE.json', dict(status='ready', contract=digest, pid=None,
          completed=[r['run_id'] for r in done], seconds=sum(r['seconds'] for r in done),
          prepare_seconds=time.monotonic()-began))
    return digest


def audit():
    began = time.monotonic(); contract = read(OUT/'contract.json')
    assert all(sha(p) == h for p, h in contract['hashes'].items())
    rows = []; checks = []; before = {}
    for kind in ('Z', 'O', 'A', 'C'):
        source = OLD if kind in ('Z', 'O') else ART
        expected = read(prior.OUT/'contract.json')['contract'] if source == OLD else contract['contract']
        for fold in (0, 1):
            for seed in (42, 43):
                assert time.monotonic()-began < 900
                run = f'{kind}_seed{seed}_fold{fold}'; r = read(source/'metrics'/f'{run}.json')
                ids = split_ids(fold); train = dataset(ids['train'], kind)
                val = dataset(ids['validation'], kind); norm = normalizer(train)
                cp = source/'checkpoints'/run
                for p in (cp/'best.pt', cp/'latest.pt'): before[str(p)] = sha(p)
                best = torch.load(cp/'best.pt', map_location='cpu', weights_only=False)
                last = torch.load(cp/'latest.pt', map_location='cpu', weights_only=False)
                assert (best['contract'], best['kind'], best['fold'], best['seed']) == (expected, kind, fold, seed)
                assert last['contract'] == r['contract'] == expected and last['step'] == 300
                assert best['step'] == r['best_step'] == max(last['history'], key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(best['mean'], norm.mean); np.testing.assert_array_equal(best['std'], norm.std)
                raw = checked_raw(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'), val)
                score = metrics(raw, val, r['threshold'])[2]
                assert max(abs(score[k]-r[k]) for k in COLS) < 1e-10
                m = make_model(kind, seed).cuda(); m.load_state_dict(best['model'])
                assert not m.frontend.wide and r['params'] == 3297
                replay = predictions(m, val, norm, torch.device('cuda'))
                error = max(float(abs(replay[p][k]-raw[p][k]).max()) for p in raw for k in raw[p])
                assert error < 2e-4
                assert max(abs(metrics(replay, val, r['threshold'])[2][k]-score[k]) for k in COLS) < 1e-10
                checks.append(dict(run_id=run, replay_error=error, params=r['params'], narrow_frontend=True,
                                   reused=source == OLD))
                rows.append(r); pd.DataFrame(checks).to_csv(OUT/'run_audit.csv', index=False)
                print('AUDITED', run, flush=True)
    assert len(before) == 32 and all(sha(p) == h for p, h in before.items())
    assert all(sha(p) == h for p, h in contract['hashes'].items())
    df = pd.DataFrame(rows); df.to_csv(ART/'summary.csv', index=False)
    means = df.groupby('kind')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean()
    means.to_csv(OUT/'means.csv'); comparisons = []
    for candidate, reference in (('O','C'), ('O','A'), ('C','Z'), ('A','Z')):
        a = df[df.kind == candidate].set_index(['fold','seed'])
        b = df[df.kind == reference].set_index(['fold','seed']); d = a[COLS]-b[COLS]
        d.to_csv(OUT/f'delta_{candidate}_{reference}.csv')
        comparisons.append(dict(candidate=candidate, reference=reference, **d.mean().to_dict(),
          f1_positive=int((d.macro_f1_tol1 > 0).sum()), ap_positive=int((d.raw_ap > 0).sum()),
          passed=bool(d.macro_f1_tol1.mean() >= .015 and d.macro_f1_tol0.mean() > 0 and d.raw_ap.mean() > 0
                      and (d.macro_f1_tol1 > 0).sum() >= 3 and (d.raw_ap > 0).sum() >= 3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv', index=False)
    write(OUT/'checkpoint_hashes.json', before)
    write(OUT/'completion_audit.json', dict(status='complete', new_training_runs=8, reused_runs=8,
          full_gpu_replays=16, checkpoints=32, hashes_unchanged=True, norms_rebuilt=True,
          test_predictions_accessed=False, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
          new_training_validation_seconds=float(df[df.kind.isin(['A','C'])].seconds.sum()),
          audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json', dict(status='complete', contract=contract['contract'], pid=None))
    print(means.to_string(), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('prepare','all','audit'), default='all')
    args = p.parse_args(); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    try:
        bind()
        if args.stage == 'audit': audit(); return
        with threadpool_limits(2): digest = prepare()
        if args.stage == 'prepare': print(digest, flush=True); return
        for f in (0,1):
            for s in (42,43):
                for k in ('A','C'): engine.train_one(k, f, s, digest)
        write(OUT/'STATE.json', {**read(OUT/'STATE.json'), 'status':'auditing', 'pid':os.getpid()})
        audit()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT/'failures.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.time(), traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json', dict(status='failed', pid=None)); raise


if __name__ == '__main__': main()
