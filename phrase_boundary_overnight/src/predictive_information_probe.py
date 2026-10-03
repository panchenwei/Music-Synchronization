"""Fixed-window post-result information removal, no retraining or test use."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from . import predictive_information_study as study
from .predictive_information_model import make_model
from .score_context_study import ROOT,read,write,sha,CurvePieceBalancedSampler,positive_weight
from .three_round_round2 import split_ids

OUT=ROOT/'reports/predictive_information_probe'
PITCH=[58,59,60,65,66,67];RHYTHM=[61,62,63,68,69,70]


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);rows=[];hashes=dict(read(study.OUT/'contract.json')['hashes'])
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    for p in (Path(__file__),OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for f in (0,1):
        ids=split_ids(f)
        for kind in ('G','D','Z'):
            train=study.dataset(ids['train'],kind);val=study.dataset(ids['validation'],kind);norm=study.normalizer(train)
            weight=torch.tensor(positive_weight(train,10),device='cuda')
            for seed in (42,43):
                cp=study.ART/'checkpoints'/f'{kind}_seed{seed}_fold{f}'/'best.pt';hashes[str(cp)]=sha(cp)
                saved=torch.load(cp,map_location='cpu',weights_only=False);model=make_model(kind,seed).cuda().eval();model.load_state_dict(saved['model'])
                np.testing.assert_array_equal(norm.mean,saved['mean']);np.testing.assert_array_equal(norm.std,saved['std'])
                for split,data in [('train',train),('validation',val)]:
                    sampler=CurvePieceBalancedSampler(data,norm,64,32,20260914)
                    for batch in range(4):
                        assert time.monotonic()-began<600
                        x,y,m,v=(z.cuda() for z in sampler.batch());use=m>.5;truth=y[use].cpu().numpy();reference=None;baseline=None
                        for intervention,columns in [('full',[]),('no_pitch',PITCH),('no_rhythm',RHYTHM),('no_information',PITCH+RHYTHM)]:
                            changed=x.clone();changed[:,:,columns]=0
                            with torch.no_grad():logit=model(changed,padding_mask=~v.bool());prob=logit.sigmoid()
                            bce=float((torch.nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=weight,reduction='none')*m).sum()/m.sum().clamp_min(1))
                            ap=float(average_precision_score(truth,prob[use].cpu().numpy()))
                            if reference is None:reference=prob;baseline=(bce,ap)
                            difference=float((prob-reference).abs()[use].mean())
                            if kind=='Z':assert difference==0
                            rows.append(dict(fold=f,seed=seed,kind=kind,split=split,batch=batch,intervention=intervention,bce=bce,ap=ap,
                                bce_delta=bce-baseline[0],ap_delta=ap-baseline[1],probability_change=difference))
                print('PROBED',f,seed,kind,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'metrics.csv',index=False)
    means=df.groupby(['kind','split','intervention'])[['bce','ap','bce_delta','ap_delta','probability_change']].mean();means.to_csv(OUT/'means.csv')
    subset=means.loc[('G','validation')]
    gate=bool(subset.loc['no_pitch','bce_delta']<-.01 and subset.loc['no_rhythm','bce_delta']>.01)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',models=12,training_runs=0,test_used=False,hashes_unchanged=True,
        rhythm_only_followup_gate=gate,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True);print('Rhythm-only follow-up gate',gate,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
