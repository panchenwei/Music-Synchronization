"""Single output-relation change on the existing C3 training recipe."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import score_context_study as engine
from . import start_guided_attention_study as audit_engine
from .learned_boundary_contrast import make_model
from .score_context_study import ROOT,read,write,sha,normalizer

OUT=ROOT/'reports/learned_boundary_contrast';ART=ROOT/'artifacts/learned_boundary_contrast';dataset=audit_engine.dataset


def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(audit_engine.base.OUT/'contract.json')['hashes'])
    files=[Path(__file__),ROOT/'src/learned_boundary_contrast.py',ROOT/'src/start_guided_attention_study.py',ROOT/'src/score_context_study.py',ROOT/'tests/test_learned_boundary_contrast.py',OUT/'PROTOCOL.md',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/interstart_decoder.py',ROOT/'src/recurrence_message_probe.py']
    files+=list((ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'))
    for f in (0,1):
        for seed in (42,43):
            for suffix in ('.json','_predictions.csv.gz'):files.append(audit_engine.base.ART/'metrics'/f'C3_seed{seed}_fold{f}{suffix}')
    for seed in (42,43):
        state=make_model('G',seed).state_dict()
        for kind in ('D','Z'):
            for k,v in make_model(kind,seed).state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);digest=prepare()
    engine.OUT=OUT;engine.ART=ART;engine.CAP=4800.;engine.dataset=dataset;engine.normalizer=normalizer;engine.make_model=make_model
    for f in (0,1):
        for seed in (42,43):
            for kind in ('G','D','Z'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent'];engine.train_one(kind,f,seed,digest)
    audit_engine.OUT=OUT;audit_engine.ART=ART;audit_engine.make_model=make_model;audit_engine.audit()


if __name__=='__main__':
    with threadpool_limits(2):main()
