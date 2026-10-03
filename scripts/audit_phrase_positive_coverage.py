"""Exact expected window supervision versus whole-work supervision (no fitting)."""
import json
import numpy as np
from audit_phrase_supervision import data_u, RESEARCH
from src.phase6_models import window_starts


def main():
    frame, data = data_u()
    rows = []
    for pid, item in sorted(data.items()):
        mask = np.asarray(item['label_mask'])
        y = np.asarray(item['labels'])
        assert mask.shape == y.shape and np.isfinite(mask).all()
        starts = window_starts(len(y), 64, 32)
        exposure = np.zeros(len(y))
        for start in starts:
            exposure[start:start+64] += 1 / len(starts)
        assert (exposure > 0).all()
        scored = float(mask.sum())
        positives = float((y * mask).sum())
        whole_density = positives / scored
        window_density = float((y * mask * exposure).sum() / (mask * exposure).sum())
        sampled = json.loads((RESEARCH / 'reports/supervision_exposure_20261004/results.json').read_text(encoding='utf-8'))
        observed = [r['positives'] / r['supervised'] for r in sampled['rows'] if r['piece'] == pid]
        rows.append(dict(piece=pid, positions=len(y), scored=scored, positive_labels=positives,
                         windows=len(starts), whole_density=whole_density,
                         expected_sampled_density=window_density,
                         density_ratio=window_density / whole_density,
                         observed_density_min=min(observed) if observed else None,
                         observed_density_max=max(observed) if observed else None,
                         folds_in_train=sorted(frame[(frame.piece_id == pid) & (frame.split == 'train')].fold.unique().tolist())))
    out = RESEARCH / 'reports/supervision_exposure_20261004/positive_coverage.json'
    value = dict(status='reproduced', works=len(rows), holdouts_read=False,
                 zero_exposure_positions=0, optimizer_steps=0, rows=rows)
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(works=len(rows), min_density_ratio=min(r['density_ratio'] for r in rows),
                         max_density_ratio=max(r['density_ratio'] for r in rows),
                         lowest=sorted(rows, key=lambda r: r['density_ratio'])[:3]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
