"""Quantify failed replay without changing frozen tolerance or saved predictions."""
import time
import numpy as np
import pandas as pd
import torch
from .run_preaggregate_note_study import OUT,ART,OLD_ART,OLD_OUT,dataset,COLS
from .preaggregate_note_graph import PreAggregateBoundary
from .run_note_relation_study import predictions
from .phrase_end_auxiliary import read,write,sha,normalizer,split_ids
from .local_context_study import metrics
from .phase2_models import nms_probabilities
from .audit_external_stem_transfer import checked_raw


def main():
    torch.set_num_threads(2);began=time.monotonic();dest=OUT/'replay_diagnosis';dest.mkdir(exist_ok=True)
    hashes=read(OUT/'contract.json')['hashes'];assert all(sha(p)==h for p,h in hashes.items())
    cps=list((ART/'checkpoints').glob('*/*.pt'));before={str(p):sha(p) for p in cps};rows=[];changes=[]
    for rp in sorted((ART/'metrics').glob('N_seed*_fold*.json')):
        r=read(rp);ids=split_ids(r['fold']);val=dataset(ids['validation']);norm=normalizer(dataset(ids['train']))
        c=torch.load(ART/'checkpoints'/r['run_id']/'best.pt',map_location='cpu',weights_only=False)
        model=PreAggregateBoundary(r['seed']).cuda();model.load_state_dict(c['model']);baseline=checked_raw(pd.read_csv(ART/'metrics'/f"{r['run_id']}_predictions.csv.gz"),val)
        original_flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        previous=None
        for mode in ('native_1','native_2','tf32_off'):
            assert time.monotonic()-began<600
            if mode=='tf32_off':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            raw=predictions(model,val,norm);score=metrics(raw,val,r['threshold'])[2]
            prob_delta=max(float(np.max(abs(raw[p][k]-baseline[p][k]))) for p in raw for k in raw[p])
            repeat_delta=None if previous is None else max(float(np.max(abs(raw[p][k]-previous[p][k]))) for p in raw for k in raw[p])
            saved_score=metrics(baseline,val,r['threshold'])[2]
            assert max(abs(saved_score[k]-r[k]) for k in COLS)<1e-10
            decision_changes=0
            for p,pp in raw.items():
                valid=val[p]['label_mask'].astype(bool)
                for k,a in pp.items():
                    old=(nms_probabilities(baseline[p][k])>=r['threshold'])&valid;new=(nms_probabilities(a)>=r['threshold'])&valid
                    for b in np.flatnonzero(old!=new):
                        decision_changes+=1;changes.append(dict(run_id=r['run_id'],mode=mode,piece=p,performance=k,beat=int(b),saved=float(baseline[p][k][b]),replay=float(a[b]),label=int(val[p]['labels'][b])))
            row=dict(run_id=r['run_id'],mode=mode,probability_max_delta=prob_delta,repeat_max_delta=repeat_delta,decision_changes=decision_changes,**{k:score[k] for k in COLS},**{k+'_delta':score[k]-r[k] for k in COLS})
            rows.append(row);pd.DataFrame(rows).to_csv(dest/'scores.csv',index=False);print(row,flush=True);previous=raw
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=original_flags
    assert before=={str(p):sha(p) for p in cps} and all(sha(p)==h for p,h in hashes.items())
    pd.DataFrame(changes).to_csv(dest/'changed_decisions.csv',index=False)
    write(dest/'audit.json',dict(status='diagnosis_complete',cases=12,seconds=time.monotonic()-began,source_checkpoint_hashes_unchanged=True,torch_version=torch.__version__,original_matmul_tf32=original_flags[0],original_cudnn_tf32=original_flags[1],does_not_override_failed_audit=True))


if __name__=='__main__':main()
