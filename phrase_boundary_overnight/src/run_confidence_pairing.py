"""Evaluate frozen endpoint decoding on existing development predictions only."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .confidence_pair_decoder import decode
from .phrase_end_auxiliary import ROOT, read, write, sha, dataset, end_data, split_ids
from .audit_external_stem_transfer import checked_raw
from .phase2_models import evaluate_single_performance, nms_probabilities

OUT = ROOT/'reports/confidence_pairing'
ART = ROOT/'artifacts/confidence_pairing'
COLS = ['macro_precision_tol1', 'macro_recall_tol1', 'macro_f1_tol1', 'macro_f1_tol0']


def main():
    began = time.monotonic(); ART.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), ROOT/'src/confidence_pair_decoder.py', OUT/'PROTOCOL.md', ROOT/'tests/test_confidence_pair_decoder.py', ROOT/'src/evaluation.py', ROOT/'src/phase2_models.py']
    for folder in ('phrase_end_auxiliary', 'independent_phrase_end'):
        assert read(ROOT/f'reports/{folder}/completion_audit.json')['status'] == 'complete'
        inherited = read(ROOT/f'reports/{folder}/contract.json')['hashes']
        assert all(sha(p) == h for p, h in inherited.items())
        files.extend(Path(p) for p in inherited)
    for fold in (0, 1):
        for seed in (42, 43):
            for folder, kind in [('phrase_end_auxiliary', 'C'), ('independent_phrase_end', 'D')]:
                for suffix in ('.json', '_predictions.csv.gz'):
                    files.append(ROOT/f'artifacts/{folder}/metrics/{kind}_seed{seed}_fold{fold}{suffix}')
    hashes = {str(p):sha(p) for p in files}
    contract = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists(): assert read(OUT/'contract.json')['contract'] == contract
    write(OUT/'contract.json', dict(contract=contract, hashes=hashes))
    write(OUT/'STATE.json', dict(status='running', contract=contract))
    rows = []; audits = []
    for fold in (0, 1):
        ids = split_ids(fold); assert not set(ids['train']) & set(ids['validation'])
        val = dataset(ids['validation']); ev = end_data(val)
        for seed in (42, 43):
            assert time.monotonic()-began < 600
            run = f'pair_seed{seed}_fold{fold}'
            cp = ROOT/f'artifacts/phrase_end_auxiliary/metrics/C_seed{seed}_fold{fold}'
            dp = ROOT/f'artifacts/independent_phrase_end/metrics/D_seed{seed}_fold{fold}'
            c, d = read(str(cp)+'.json'), read(str(dp)+'.json')
            cf, df = pd.read_csv(str(cp)+'_predictions.csv.gz'), pd.read_csv(str(dp)+'_predictions.csv.gz')
            cr, dr = checked_raw(cf, val), checked_raw(df, ev)
            assert cr.keys() == dr.keys()
            paired = {}; spans_rows = []; output = []
            for pid, perfs in cr.items():
                assert perfs.keys() == dr[pid].keys(); paired[pid] = {}
                for perfid, start in perfs.items():
                    end = dr[pid][perfid]
                    score, spans, objective = decode(start, end, c['threshold'], d['threshold'])
                    assert set(np.flatnonzero(score)) <= set(np.flatnonzero(nms_probabilities(start) >= c['threshold']))
                    paired[pid][perfid] = score
                    spans_rows.extend(dict(piece_id=pid, performance_id=perfid, start=s, end=e, start_confidence=float(start[s]), end_confidence=float(end[e])) for s, e in spans)
                    output.extend((pid, perfid, b, float(p), int(val[pid]['labels'][b]), int(val[pid]['label_mask'][b]), float(start[b]), float(end[b])) for b, p in enumerate(score))
            bp, _, bs = evaluate_single_performance(cr, val, c['threshold'])
            pp, pieces, ps = evaluate_single_performance(paired, val, c['threshold'])
            assert max(abs(bs[k]-c[k]) for k in COLS) < 1e-10
            frame = pd.DataFrame(output, columns=['piece_id','performance_id','beat','probability','label','valid','start_probability','end_probability'])
            dest = ART/f'{run}_predictions.csv.gz'; frame.to_csv(dest, index=False)
            pd.DataFrame(spans_rows).to_csv(ART/f'{run}_spans.csv', index=False)
            pp.to_csv(ART/f'{run}_performances.csv', index=False); pieces.to_csv(ART/f'{run}_pieces.csv', index=False)
            replay = checked_raw(pd.read_csv(dest), val)
            err = max(abs(evaluate_single_performance(replay, val, c['threshold'])[2][k]-ps[k]) for k in COLS)
            assert err < 1e-10
            # Count differences include rematching under the unchanged greedy evaluator.
            counts = {f'net_{key}_change':int(pp[key].sum()-bp[key].sum()) for key in ('tp_tol1','fp_tol1','fn_tol1','predicted_boundaries')}
            rows.append(dict(run_id=run, fold=fold, seed=seed, start_threshold=c['threshold'], end_threshold=d['threshold'], **{'baseline_'+k:bs[k] for k in COLS}, **{'paired_'+k:ps[k] for k in COLS}, **{'delta_'+k:ps[k]-bs[k] for k in COLS}, raw_start_ap_unchanged=c['raw_ap'], **counts))
            audits.append(dict(run_id=run, replay_metric_error=err, performances=len(pp), pieces=len(pieces), spans=len(spans_rows), saved_predictions_hash=sha(dest)))
            pd.DataFrame(rows).to_csv(OUT/'comparison.csv', index=False)
            print(run, 'start F1', bs['macro_f1_tol1'], '->', ps['macro_f1_tol1'], flush=True)
    table = pd.DataFrame(rows); assert len(table) == 4
    summary = {k:float(table[k].mean()) for k in table if k.startswith(('baseline_', 'paired_', 'delta_', 'raw_'))}
    summary.update(f1_positive=int((table.delta_macro_f1_tol1 > 0).sum()), promotion=bool(table.delta_macro_f1_tol1.mean() >= .015 and table.delta_macro_f1_tol0.mean() > 0 and (table.delta_macro_f1_tol1 > 0).sum() >= 3))
    write(OUT/'summary.json', summary); pd.DataFrame(audits).to_csv(OUT/'run_audit.csv', index=False)
    assert all(sha(p) == h for p,h in hashes.items())
    write(OUT/'completion_audit.json', dict(status='complete', runs=4, saved_metric_replays=4, source_input_hashes_unchanged=True, thresholds_retuned=False, decoder_receives_gold=False, test_predictions_accessed=False, training_steps=0, seconds=time.monotonic()-began))
    write(OUT/'STATE.json', dict(status='complete', contract=contract)); print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
