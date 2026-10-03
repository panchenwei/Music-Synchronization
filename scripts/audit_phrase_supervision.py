"""Trace actual training sampler mask weights without training or reading holdouts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'phrase_boundary_overnight'
sys.path[:0] = [str(RESEARCH / 'reports/mentor_continuation_20260916'), str(RESEARCH)]
from score_only_export_training import data_u, normalizer, ContextSampler


def main():
    frame, data = data_u()
    results = []
    for fold in (0, 1):
        part = frame[frame.fold == fold]
        train_ids = set(part[part.split == 'train'].piece_id)
        validation_ids = set(part[part.split == 'validation'].piece_id)
        assert train_ids.isdisjoint(validation_ids)
        train = {p: data[p] for p in sorted(train_ids)}
        norm = normalizer(train)
        for seed in (42, 43, 44, 45):
            sampler = ContextSampler(train, norm, seed, False)
            counts = {p: [0, 0., 0., 0.] for p in train}
            for _ in range(600):
                _, labels, mask, _ = sampler.batch()
                sizes = mask.sum(dim=1).numpy()
                total = float(sizes.sum())
                assert total > 0
                for i, (pid, *_rest) in enumerate(sampler.last_selection):
                    row = counts[pid]
                    row[0] += 1
                    row[1] += float(sizes[i])
                    row[2] += float(sizes[i]) / total
                    row[3] += float((labels[i] * mask[i]).sum())
            assert sum(r[0] for r in counts.values()) == 4800
            assert abs(sum(r[2] for r in counts.values()) - 600) < 1e-4
            ratios = []
            for pid, (n, supervised, weight, positives) in counts.items():
                sample_share = n / 4800
                loss_share = weight / 600
                ratios.append(loss_share / sample_share)
                results.append(dict(fold=fold, seed=seed, piece=pid, samples=n,
                                    supervised=supervised, positives=positives,
                                    sample_share=sample_share, loss_share=loss_share,
                                    relative_weight=loss_share / sample_share))
            print(json.dumps(dict(fold=fold, seed=seed, min_relative_weight=min(ratios),
                                  max_relative_weight=max(ratios)), ensure_ascii=False), flush=True)
    out = RESEARCH / 'reports/supervision_exposure_20261004'
    out.mkdir(exist_ok=True)
    (out / 'results.json').write_text(json.dumps(dict(status='reproduced',
        optimizer_steps=0, sampled_batches=4800, holdouts_read=False,
        interpretation='Mask-denominator exposure, not gradient magnitude or causal evidence.',
        rows=results), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
