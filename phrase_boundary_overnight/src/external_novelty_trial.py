"""One interpretable audio-change channel added to the unchanged small CNN."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import external_mert_trial as template
from .external_structure_probe import shuffled
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_novelty_trial';ART=ROOT/'artifacts/external_novelty_trial'
SOURCE=ROOT/'artifacts/external_structure_probe';base=template.base


def replace_features(x,novelty):
    x=x.copy();assert novelty.shape==(len(x),)
    x[:,28:120]=0;x[:,28]=novelty;return x


def load_fold(data,fold,kind):
    with np.load(SOURCE/f'M_fold{fold}_predictions.npz',allow_pickle=False) as z:
        result={}
        for k,v in data.items():
            value=z[k].copy()
            if kind=='F':value=shuffled(value[:,None],v['features'][:,-1]>.5,k)[:,0]
            result[k]={**v,'features':replace_features(v['features'],value)}
    return result


def prepare():
    for p in (OUT,ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/external_structure_probe/completion_audit.json')['status']=='complete'
    comparisons=read(ROOT/'reports/external_structure_probe/comparisons.json')
    assert all(r['passed'] for r in comparisons if r['candidate']=='M')
    hashes=dict(read(ROOT/'reports/external_structure_probe/contract.json')['hashes']);hashes.update(read(ROOT/'reports/external_structure_probe/artifact_hashes.json'))
    for p in (Path(__file__),ROOT/'src/external_mert_trial.py',ROOT/'tests/test_external_novelty_trial.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    data=base.load_data();sp=read(template.OLD_OUT/'splits.json');checks=[]
    for f in range(3):
        old=base.subset(data,sp[str(f)]['train']);oldnorm=base.normalizer(old)
        for kind in ('S','E','F'):
            condition=load_fold(data,f,kind);train=base.subset(condition,sp[str(f)]['train']);norm=base.normalizer(train)
            for k in old:
                np.testing.assert_array_equal(train[k]['labels'],old[k]['labels']);np.testing.assert_array_equal(train[k]['label_mask'],old[k]['label_mask'])
                np.testing.assert_array_equal(train[k]['features'][:,:28],old[k]['features'][:,:28]);np.testing.assert_array_equal(train[k]['features'][:,-1],old[k]['features'][:,-1])
                if kind=='S':np.testing.assert_array_equal(template.mask_modalities(norm.apply(train[k]['features']),'S'),template.mask_modalities(oldnorm.apply(old[k]['features']),'S'))
            checks.append(dict(fold=f,kind=kind,old_score_labels_mask_coverage_equal=True,S_normalized_inputs_equal=True))
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    return data,sp,digest


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);data,sp,digest=prepare()
    template.OUT=OUT;template.ART=ART;template.load_fold=load_fold;base.OUT=OUT;base.ART=ART;base.mask_modalities=template.mask_modalities
    began=time.monotonic()
    for f in range(3):
        for seed in (42,43):
            for kind in ('S','E','F'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                assert time.monotonic()-began<5400;base.train_one(load_fold(data,f,kind),sp,kind,f,seed,digest)
    template.audit(data,sp,digest)


if __name__=='__main__':
    with threadpool_limits(2):main()
