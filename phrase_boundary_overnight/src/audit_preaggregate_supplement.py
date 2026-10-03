"""Versioned deterministic evidence, does not waive the original failed gate."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
import numpy as np
import pandas as pd
import torch
from .run_preaggregate_note_study import OUT,ART,OLD_ART,OLD_OUT,dataset,COLS
from .preaggregate_note_graph import PreAggregateBoundary
from .note_relation_graph import NoteRelationBoundary
from .run_note_relation_study import predictions
from .phrase_end_auxiliary import read,write,sha,normalizer,split_ids
from .local_context_study import metrics
from .phase2_models import nms_probabilities
from .audit_external_stem_transfer import checked_raw


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);began=time.monotonic();dest=OUT/'deterministic_supplement';dest.mkdir(exist_ok=True)
    contract=read(OUT/'contract.json');hashes=contract['hashes'];assert all(sha(p)==h for p,h in hashes.items())
    cps=list((ART/'checkpoints').glob('*/*.pt'))+list((OLD_ART/'checkpoints').glob('G*/*.pt'));assert len(cps)==16
    before={str(p):sha(p) for p in cps};rows=[];results=[]
    for label,root in (('N',ART),('L',OLD_ART)):
        files=sorted((root/'metrics').glob(('N' if label=='N' else 'G')+'_seed*_fold*.json'));assert len(files)==4
        for rp in files:
            assert time.monotonic()-began<600
            r=read(rp);ids=split_ids(r['fold']);train=dataset(ids['train']);val=dataset(ids['validation']);norm=normalizer(train)
            best=torch.load(root/'checkpoints'/r['run_id']/'best.pt',map_location='cpu',weights_only=False);last=torch.load(root/'checkpoints'/r['run_id']/'latest.pt',map_location='cpu',weights_only=False)
            expected=contract['contract'] if label=='N' else read(OLD_OUT/'contract.json')['contract']
            assert r['contract']==best['contract']==last['contract']==expected and last['step']==300
            assert r['best_step']==best['step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
            np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
            raw=checked_raw(pd.read_csv(root/'metrics'/f"{r['run_id']}_predictions.csv.gz"),val);saved=metrics(raw,val,r['threshold'])[2]
            assert max(abs(saved[k]-r[k]) for k in COLS)<1e-10
            m=(PreAggregateBoundary(r['seed']) if label=='N' else NoteRelationBoundary(True,r['seed'])).cuda();m.load_state_dict(best['model'])
            first=predictions(m,val,norm);second=predictions(m,val,norm);repeat=max(float(abs(first[p][k]-second[p][k]).max()) for p in first for k in first[p]);assert repeat==0
            score=metrics(first,val,r['threshold'])[2];delta=max(float(abs(first[p][k]-raw[p][k]).max()) for p in first for k in first[p])
            changes=sum(int((((nms_probabilities(first[p][k])>=r['threshold'])!=(nms_probabilities(raw[p][k])>=r['threshold']))&val[p]['label_mask'].astype(bool)).sum()) for p in first for k in first[p])
            ts=metrics(predictions(m,train,norm),train,r['threshold'])[2]
            rows.append(dict(mode=label,run_id=r['run_id'],repeat_delta=repeat,saved_probability_delta=delta,changed_decisions=changes,original_probability_gate_pass=delta<2e-4,original_metric_gate_pass=max(abs(score[k]-saved[k]) for k in COLS)<1e-10,train_f1=ts['macro_f1_tol1'],**{k:score[k] for k in COLS},**{k+'_delta':score[k]-saved[k] for k in COLS}))
            results.append({**{k:v for k,v in r.items() if k!='history'},'mode':label})
            pd.DataFrame(rows).to_csv(dest/'replays.csv',index=False)
            frames=[(p,k,b,float(prob)) for p,pp in first.items() for k,a in pp.items() for b,prob in enumerate(a)]
            pd.DataFrame(frames,columns=['piece_id','performance_id','beat','probability']).to_csv(dest/f"{label}_{r['run_id']}_predictions.csv.gz",index=False)
            print('CHECKED',label,r['run_id'],rows[-1]['original_metric_gate_pass'],'changes',changes,flush=True)
    assert before=={str(p):sha(p) for p in cps} and all(sha(p)==h for p,h in hashes.items())
    df=pd.DataFrame(results);df.to_csv(dest/'saved_result_summary.csv',index=False);df.groupby('mode')[COLS].mean().to_csv(dest/'saved_means.csv')
    d=df[df['mode']=='N'].set_index(['fold','seed'])[COLS]-df[df['mode']=='L'].set_index(['fold','seed'])[COLS]
    write(dest/'saved_comparison.json',dict(mean_delta=d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=False,reason='Original audit failed; stored F1 does not meet improvement gate either.'))
    write(dest/'audit.json',dict(status='supplement_complete_with_caveat',runs=8,total_checkpoints=16,deterministic_full_replays=16,norms_rebuilt=True,saved_prediction_metric_recomputations_pass=True,source_and_checkpoint_hashes_unchanged=True,original_audit_still_failed=True,original_gates_all_pass=all(r['original_probability_gate_pass'] and r['original_metric_gate_pass'] for r in rows),changed_decisions=sum(r['changed_decisions'] for r in rows),seconds=time.monotonic()-began,test_predictions_accessed=False))
    write(dest/'checkpoint_hashes.json',before)


if __name__=='__main__':main()
