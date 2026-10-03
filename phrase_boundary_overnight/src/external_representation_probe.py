"""Frozen feature-use diagnostic; no classifier fitting or event-score selection."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import external_representation_study as study
from . import external_representation_features as features
from .score_context_study import ROOT,read,write,sha,CurvePieceBalancedSampler,positive_weight
from .three_round_round2 import split_ids

OUT=ROOT/'reports/external_representation_probe'


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);rows=[];hashes={}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    hashes.update(read(study.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    for fold in (0,1):
        ids=split_ids(fold)
        for seed in (42,43):
            features.ACTIVE_SEED=seed
            for kind in ('E','S','Z'):
                name=f'{kind}_seed{seed}_fold{fold}';path=study.ART/'checkpoints'/name/'best.pt';hashes[str(path)]=sha(path)
                cp=torch.load(path,map_location='cpu',weights_only=False);model=features.make_model(kind,seed).cuda().eval();model.load_state_dict(cp['model'])
                train=features.dataset(ids['train'],kind);val=features.dataset(ids['validation'],kind);norm=features.normalizer(train)
                np.testing.assert_array_equal(norm.mean,cp['mean']);np.testing.assert_array_equal(norm.std,cp['std'])
                posweight=torch.tensor(positive_weight(train,10),device='cuda');w=model.input_projection.weight.detach()
                if kind=='Z':assert torch.count_nonzero(w[:,58:]).item()==0
                for split,data in [('train',train),('validation',val)]:
                    sampler=CurvePieceBalancedSampler(data,norm,64,32,20260913)
                    for batch in range(4):
                        assert time.monotonic()-started<600
                        x,y,known,valid=(t.cuda() for t in sampler.batch());other=x.clone();other[...,58:]=0
                        with torch.no_grad():
                            original=model(x,~valid.bool());removed=model(other,~valid.bool());den=known.sum().clamp_min(1)
                            before=torch.nn.functional.binary_cross_entropy_with_logits(original,y,reduction='none',pos_weight=posweight)
                            after=torch.nn.functional.binary_cross_entropy_with_logits(removed,y,reduction='none',pos_weight=posweight)
                            delta=float(((removed-original).abs()*known).sum()/den)
                            if kind=='Z':assert delta==0
                            rows.append(dict(kind=kind,seed=seed,fold=fold,split=split,batch=batch,
                                extra_weight_rms=float(w[:,58:].square().mean().sqrt()),original_weight_rms=float(w[:,:58].square().mean().sqrt()),
                                mean_abs_logit_delta=delta,mean_abs_probability_delta=float(((removed.sigmoid()-original.sigmoid()).abs()*known).sum()/den),
                                bce_original=float((before*known).sum()/den),bce_removed=float((after*known).sum()/den),
                                bce_removed_minus_original=float(((after-before)*known).sum()/den),known_queries=int(known.sum())))
                print('PROBED',name,flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'windows.csv',index=False)
    means=table.groupby(['kind','split'])[['extra_weight_rms','mean_abs_logit_delta','mean_abs_probability_delta','bce_removed_minus_original']].mean();means.to_csv(OUT/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',models_probed=12,batches_per_model_per_split=4,
        training_runs=0,weights_unchanged=True,source_hashes_unchanged=True,seconds=time.monotonic()-started,
        caution='Fixed-window mean-feature ablation, not new event F1 or a causal musical interpretation.'))
    print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
