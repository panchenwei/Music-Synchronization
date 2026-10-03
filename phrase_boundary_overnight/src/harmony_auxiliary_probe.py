"""Frozen best-model auxiliary learning diagnostic, fixed windows, no fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import harmony_auxiliary_study as study
from .harmony_auxiliary import HarmonySampler,make_model
from .score_context_study import ROOT,read,write,sha,normalizer
from .three_round_round2 import split_ids

OUT=ROOT/'reports/harmony_auxiliary_probe'


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);rows=[];hashes=dict(read(study.OUT/'contract.json')['hashes'])
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    for p in (Path(__file__),OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for f in (0,1):
        ids=split_ids(f);train=study.dataset(ids['train'],'G');val=study.dataset(ids['validation'],'G');norm=normalizer(train)
        freq=sum((np.bincount(v['harmony_labels'][v['harmony_mask']>.5],minlength=7) for v in train.values()),start=np.zeros(7,dtype=int));majority=int(freq.argmax())
        for seed in (42,43):
            for kind in ('G','D','Z'):
                cp=study.ART/'checkpoints'/f'{kind}_seed{seed}_fold{f}'/'best.pt';hashes[str(cp)]=sha(cp)
                state=torch.load(cp,map_location='cpu',weights_only=False);model=make_model(kind,seed).cuda().eval();model.load_state_dict(state['model'])
                np.testing.assert_array_equal(norm.mean,state['mean']);np.testing.assert_array_equal(norm.std,state['std'])
                for split,data in [('train',train),('validation',val)]:
                    sampler=HarmonySampler(data,norm,20260914,'G');cm=np.zeros((7,7),np.int64);losses=[];major=[]
                    for batch in range(4):
                        assert time.monotonic()-began<600
                        x,y,m,v,hy,hm=(z.cuda() for z in sampler.batch())
                        with torch.no_grad():
                            _,a=model(x,padding_mask=~v.bool(),both=True);use=hm>.5;truth=hy[use];pred=a.argmax(-1)[use]
                            cm+=np.bincount((truth*7+pred).cpu().numpy(),minlength=49).reshape(7,7)
                            losses.append(float(torch.nn.functional.cross_entropy(a[use],truth)));major.append(float((truth==majority).float().mean()))
                    precision=np.diag(cm)/np.maximum(cm.sum(0),1);recall=np.diag(cm)/np.maximum(cm.sum(1),1);f1=2*precision*recall/np.maximum(precision+recall,1e-12)
                    rows.append(dict(fold=f,seed=seed,kind=kind,split=split,accuracy=float(np.trace(cm)/cm.sum()),macro_f1=float(f1.mean()),
                        ce=float(np.mean(losses)),majority_accuracy=float(np.mean(major)),confusion=json_matrix(cm)))
                print('PROBED',kind,f,seed,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'metrics.csv',index=False);means=df.groupby(['kind','split'])[['accuracy','macro_f1','ce','majority_accuracy']].mean();means.to_csv(OUT/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',models=12,training_runs=0,hashes_unchanged=True,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


def json_matrix(a):
    import json
    return json.dumps(a.tolist())


if __name__=='__main__':
    with threadpool_limits(2):main()
