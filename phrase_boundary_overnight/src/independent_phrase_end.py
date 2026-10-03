"""Independent end network; retains C start networks for a later frozen pairing study."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import ensemble_seed_replication as engine
from .phrase_end_auxiliary import ROOT,read,write,sha,dataset as dual_data,end_data,DualBoundary,split_ids,normalizer,predictions,metrics,COLS
from .models import Normalizer
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/independent_phrase_end';ART=ROOT/'artifacts/independent_phrase_end'


def dataset(ids,kind='D'):
    assert kind=='D'
    return end_data(dual_data(ids))


def build_model(kind,seed):
    assert kind=='D'
    dual=DualBoundary(seed)
    # Same shared weights/end output/GPU RNG as E at initialization; no start loss.
    core=dual.core;core.output=dual.end_head
    return core


def prepare():
    assert read(ROOT/'reports/phrase_end_auxiliary/completion_audit.json')['status']=='complete'
    assert not read(ROOT/'reports/phrase_end_auxiliary/comparison.json')['promotion']
    for p in (OUT,ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(ROOT/'reports/phrase_end_auxiliary/contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_independent_phrase_end.py'):hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==contract for r in done)
    write(OUT/'STATE.json',dict(status='ready',pid=None,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),contract=contract,target='structural_end_auxiliary_to_start_goal'))
    return contract


def report_and_audit():
    began=time.monotonic();df=pd.DataFrame([{k:v for k,v in read(p).items() if k!='history'} for p in (ART/'metrics').glob('*_fold*.json')]);assert len(df)==4
    assert set(zip(df.fold,df.seed))=={(f,s) for f in (0,1) for s in (42,43)};df.to_csv(ART/'summary.csv',index=False)
    hashes=read(OUT/'contract.json')['hashes'];assert all(sha(p)==h for p,h in hashes.items());state=read(OUT/'STATE.json')
    cps=list((ART/'checkpoints').glob('*/*.pt'));assert len(cps)==8;before={str(p):sha(p) for p in cps};rows=[]
    from .phrase_end_auxiliary import EndView
    for r in df.to_dict('records'):
        ids=split_ids(r['fold']);assert not set(ids['train'])&set(ids['validation']);train=dataset(ids['train']);val=dataset(ids['validation']);norm=normalizer(train)
        path=ART/'checkpoints'/r['run_id'];best=torch.load(path/'best.pt',map_location='cpu',weights_only=False);last=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
        assert last['step']==300 and best['contract']==last['contract']==state['contract'];assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
        np.testing.assert_array_equal(best['mean'],norm.mean);np.testing.assert_array_equal(best['std'],norm.std)
        frame=pd.read_csv(ART/'metrics'/f"{r['run_id']}_predictions.csv.gz");raw=checked_raw(frame,val);score=metrics(raw,val,r['threshold'])[2]
        error=max(abs(score[c]-r[c]) for c in COLS);assert error<1e-10
        model=build_model('D',r['seed']);assert sum(p.numel() for p in model.parameters())==3297;model.load_state_dict(best['model']);model.to('cuda').eval()
        replay=predictions(model,val,norm,torch.device('cuda'));delta=max(float(np.max(abs(replay[p][k]-raw[p][k]))) for p in raw for k in raw[p]);assert delta<2e-4
        assert max(abs(metrics(replay,val,r['threshold'])[2][c]-score[c]) for c in COLS)<1e-10
        ts=metrics(predictions(model,train,norm,torch.device('cuda')),train,r['threshold'])[2]
        # Common final-step comparison removes differences in checkpoint-selection target.
        model.load_state_dict(last['model']);iraw=predictions(model,val,norm,torch.device('cuda'))
        joint=DualBoundary(r['seed']);js=torch.load(ROOT/'artifacts/phrase_end_auxiliary/checkpoints'/f"E_seed{r['seed']}_fold{r['fold']}"/'latest.pt',map_location='cpu',weights_only=False);assert js['step']==300
        joint.load_state_dict(js['model']);joint.to('cuda').eval();jraw=predictions(EndView(joint),val,norm,torch.device('cuda'))
        from .phase2_models import choose_single_threshold
        from .phrase_end_auxiliary import GRID
        it,_=choose_single_threshold(iraw,val,GRID);jt,_=choose_single_threshold(jraw,val,GRID)
        si=metrics(iraw,val,it)[2];sj=metrics(jraw,val,jt)[2]
        rows.append(dict(run_id=r['run_id'],metric_error=error,gpu_replay_error=delta,train_end_f1=ts['macro_f1_tol1'],validation_end_f1=r['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1'],**{'independent_final_'+c:si[c] for c in COLS},**{'shared_final_'+c:sj[c] for c in COLS}))
        pd.DataFrame(rows).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED independent',r['run_id'],flush=True)
    assert before=={str(p):sha(p) for p in cps} and all(sha(p)==h for p,h in hashes.items())
    write(OUT/'checkpoint_hashes.json',before);df[list(COLS)].mean().to_csv(OUT/'end_means.csv')
    write(OUT/'completion_audit.json',dict(status='complete',runs=4,checkpoints=8,metric_recomputations=True,full_gpu_replays=4,train_norms_recomputed=True,common_final_step_comparisons=4,source_and_checkpoint_hashes_unchanged=True,test_predictions_accessed=False,audit_training_updates=0,seconds=time.monotonic()-began))
    state.update(status='complete',pid=None);write(OUT/'STATE.json',state);print('END ONLY\n'+df[list(COLS)].mean().to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','prepare','audit'],default='all');a=p.parse_args();torch.set_num_threads(2)
    try:
        if a.stage=='audit':report_and_audit();return
        contract=prepare()
        if a.stage=='prepare':print(contract);return
        engine.OUT=OUT;engine.ART=ART;engine.dataset=dataset;engine.build_model=build_model;engine.predictions=predictions
        for fold in (0,1):
            for seed in (42,43):engine.train_one('D',fold,seed,contract)
        report_and_audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise


if __name__=='__main__':main()
