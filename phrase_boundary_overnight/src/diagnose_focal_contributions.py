"""Read-only loss/logit-gradient comparison on TRAIN batches of audited B models."""
import numpy as np
import pandas as pd
import torch
from .score_novelty_study import ROOT, dataset, make_model, split_ids, read, write
from .phase7_models import CurvePieceBalancedSampler
from .models import Normalizer
from .phase6_models import positive_weight
from .boundary_focal import WeightedFocal


def main():
    torch.set_num_threads(2)
    out = ROOT / 'reports/focal_boundary_study'
    records = []
    for fold in (0, 1):
        train = dataset(split_ids(fold)['train'], 'B')
        for seed in (42, 43):
            run = f'B_seed{seed}_fold{fold}'
            saved = torch.load(ROOT/'artifacts/score_novelty_study/checkpoints'/run/'best.pt', map_location='cpu', weights_only=False)
            model = make_model('P', seed).eval()
            model.load_state_dict(saved['model'])
            norm = Normalizer(saved['mean'], saved['std'])
            sampler = CurvePieceBalancedSampler(train, norm, 64, 32, 20260910)
            weight = positive_weight(train, 10)
            for batch in range(12):
                x, y, mask, valid = sampler.batch()
                with torch.no_grad():
                    frozen_logits = model(x, padding_mask=~valid.bool())
                pt = torch.where(y > .5, frozen_logits.sigmoid(), 1-frozen_logits.sigmoid())
                for gamma in (0, 1, 2):
                    logits = frozen_logits.detach().clone().requires_grad_(True)
                    element_loss = WeightedFocal(torch.tensor(weight), gamma)(logits, y)
                    masked = element_loss * mask
                    grad = torch.autograd.grad(masked.sum()/mask.sum().clamp_min(1), logits)[0]
                    for cls in (0, 1):
                        for easy in (False, True):
                            select = (mask > 0) & ((y > .5) == bool(cls)) & ((pt >= .9) == easy)
                            records.append(dict(run=run, fold=fold, seed=seed, batch=batch, gamma=gamma,
                                target=cls, easy_correct=easy, count=int(select.sum()),
                                loss_sum=float(element_loss.detach()[select].sum()),
                                absolute_logit_gradient_sum=float(grad[select].abs().sum())))
    frame = pd.DataFrame(records)
    frame.to_csv(out/'baseline_loss_contributions.csv', index=False)
    total = frame.groupby(['gamma','target','easy_correct'])[['count','loss_sum','absolute_logit_gradient_sum']].sum().reset_index()
    for col in ('loss_sum','absolute_logit_gradient_sum'):
        total[col+'_fraction'] = total[col]/total.groupby('gamma')[col].transform('sum')
    total.to_csv(out/'baseline_loss_contribution_totals.csv', index=False)
    write(out/'baseline_loss_contributions.json', dict(status='complete', models=4, batches_per_model=12,
        training_only=True, optimizer_steps=0, compared_on_same_frozen_logits=True,
        easy_definition='probability assigned to ground truth >=0.9',
        limitations='Post-hoc B best-checkpoint train batches, not early training or new focal models; absolute logit gradients are not net parameter gradients; no causal proof or independent validation.'))
    print(total.to_string(index=False))


if __name__ == '__main__':
    main()
