"""Read-only trained-checkpoint interventions; not new validation-selected models."""
import time
import numpy as np
import pandas as pd
import torch
from .run_note_relation_study import OUT,ART,dataset,predictions,NoteRelationBoundary,COLS
from .phrase_end_auxiliary import ROOT,read,write,sha,normalizer,split_ids
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw
from .note_relation_graph import RelationLayer


def edge_association_probe():
    # Mechanistic counterexample only: edge reassignment need not be a legal score.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(192);layer=RelationLayer().eval();h=torch.randn(3,24)
        edge=torch.tensor([[1,2],[0,0]])
        features=torch.tensor([[1.,-.2,.1,0.],[-2.,.4,-.1,0.]])
        a=layer(h,{'edge_index':edge,'edge_features':features},True)
        b=layer(h,{'edge_index':edge,'edge_features':features.flip(0)},True)
        error=float((a-b).abs().max().detach())
        assert error<1e-6
        return dict(max_error=error,explanation='Linear neighbor plus linear edge followed by mean loses edge-to-neighbor association at this layer. Counterfactual reassignment, not a valid-score invariance or proof of F1 cause.')


def main():
    torch.set_num_threads(2);began=time.monotonic()
    assert read(OUT/'completion_audit.json')['status']=='complete'
    contract=read(OUT/'contract.json');assert all(sha(p)==v for p,v in contract['hashes'].items())
    cps=list((ART/'checkpoints').glob('*/*.pt'));before={str(p):sha(p) for p in cps}
    dest=OUT/'inference_interventions';dest.mkdir(exist_ok=True);rows=[]
    for rp in sorted((ART/'metrics').glob('G_seed*_fold*.json')):
        r=read(rp);ids=split_ids(r['fold']);val=dataset(ids['validation']);norm=normalizer(dataset(ids['train']))
        cp=torch.load(ART/'checkpoints'/r['run_id']/'best.pt',map_location='cpu',weights_only=False)
        model=NoteRelationBoundary(True,r['seed']).cuda();model.load_state_dict(cp['model'])
        original={p:v['graph']['edge_features'].copy() for p,v in val.items()}
        baseline=checked_raw(pd.read_csv(ART/'metrics'/f"{r['run_id']}_predictions.csv.gz"),val)
        for mode in ('native','zero_edge_features','self_neighbors'):
            assert time.monotonic()-began<480
            model.use_relations=mode!='self_neighbors'
            for p,v in val.items():v['graph']['edge_features']=np.zeros_like(original[p]) if mode=='zero_edge_features' else original[p]
            raw=predictions(model,val,norm);score=metrics(raw,val,r['threshold'])[2]
            difference=np.concatenate([abs(raw[p][k]-baseline[p][k]) for p in raw for k in raw[p]])
            if mode=='native':
                assert float(difference.max())<2e-4
                assert max(abs(score[k]-r[k]) for k in COLS)<1e-10
            row=dict(run_id=r['run_id'],mode=mode,mean_probability_change=float(difference.mean()),max_probability_change=float(difference.max()),**score)
            rows.append(row);pd.DataFrame(rows).to_csv(dest/'scores.csv',index=False)
            records=[(p,k,b,float(prob)) for p,pp in raw.items() for k,a in pp.items() for b,prob in enumerate(a)]
            pd.DataFrame(records,columns=['piece_id','performance_id','beat','probability']).to_csv(dest/f"{r['run_id']}_{mode}.csv.gz",index=False)
            print(r['run_id'],mode,round(score['macro_f1_tol1'],6),flush=True)
    assert len(rows)==12 and before=={str(p):sha(p) for p in cps}
    assert all(sha(p)==v for p,v in contract['hashes'].items())
    df=pd.DataFrame(rows);df.groupby('mode')[COLS+['mean_probability_change']].mean().to_csv(dest/'means.csv')
    write(dest/'audit.json',dict(status='complete',cases=12,native_replays=4,thresholds_frozen=True,checkpoints_unchanged=True,source_hashes_unchanged=True,seconds=time.monotonic()-began,edge_association_probe=edge_association_probe(),limitations='Post-training input interventions are distribution shifts, not retrained ablations. They measure dependence, not causal usefulness for generalization. No test split accessed.'))
    write(dest/'checkpoint_hashes.json',before)


if __name__=='__main__':main()
