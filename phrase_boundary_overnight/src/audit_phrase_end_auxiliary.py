"""Recompute both tasks, main checkpoint selection, norms and full GPU replay."""
import time
import numpy as np
import pandas as pd
import torch
from .phrase_end_auxiliary import OUT,ART,read,write,sha,dataset,end_data,DualBoundary,EndView,split_ids,normalizer,Normalizer,predictions,metrics,COLS,original_dataset
from .audit_external_stem_transfer import checked_raw


def main():
    began=time.monotonic();torch.set_num_threads(2);state=read(OUT/'STATE.json');assert state['status'] in ('training_complete','complete')
    contract=read(OUT/'contract.json');assert contract['contract']==state['contract'] and all(sha(p)==h for p,h in contract['hashes'].items())
    df=pd.read_csv(ART/'summary.csv');assert len(df)==8 and set(zip(df.kind,df.fold,df.seed))=={(k,f,s) for k in ('C','E') for f in (0,1) for s in (42,43)}
    cps=list((ART/'checkpoints').glob('*/*.pt'));assert len(cps)==16;before={str(p):sha(p) for p in cps};rows=[]
    for r in df.to_dict('records'):
        run=r['run_id'];ids=split_ids(r['fold']);assert not set(ids['train'])&set(ids['validation'])
        val=dataset(ids['validation']);train=dataset(ids['train']);old=original_dataset(ids['train'],'B')
        for pid,item in train.items():
            for field in ('curves','pitch_profiles','labels','label_mask'):np.testing.assert_array_equal(item[field],old[pid][field])
        norm=normalizer(train);s=torch.load(ART/'checkpoints'/run/'best.pt',map_location='cpu',weights_only=False);last=torch.load(ART/'checkpoints'/run/'latest.pt',map_location='cpu',weights_only=False)
        assert (s['contract'],s['kind'],s['fold'],s['seed'],s['step'])==(state['contract'],r['kind'],r['fold'],r['seed'],r['best_step'])
        assert last['step']==300 and last['contract']==state['contract']
        assert max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']==s['step']
        np.testing.assert_array_equal(s['mean'],norm.mean);np.testing.assert_array_equal(s['std'],norm.std)
        model=DualBoundary(r['seed']);assert sum(p.numel() for p in model.parameters())==3330
        model.load_state_dict(s['model']);model.to('cuda').eval();frame=pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz');record=dict(run_id=run)
        for task,items,view,prefix in [('start',val,model,''),('end',end_data(val),EndView(model),'end_')]:
            if task=='start':f=frame[['piece_id','performance_id','beat','probability','label','valid']]
            else:f=frame[['piece_id','performance_id','beat','end_probability','end_label','end_valid']].rename(columns={'end_probability':'probability','end_label':'label','end_valid':'valid'})
            raw=checked_raw(f,items);score=metrics(raw,items,r[prefix+'threshold'])[2]
            error=max(abs(score[c]-r[prefix+c]) for c in COLS);assert error<1e-10
            replay=predictions(view,items,norm,torch.device('cuda'))
            delta=max(float(np.max(abs(replay[p][k]-raw[p][k]))) for p in raw for k in raw[p]);assert delta<2e-4
            rs=metrics(replay,items,r[prefix+'threshold'])[2];assert max(abs(rs[c]-score[c]) for c in COLS)<1e-10
            record[task+'_metric_error']=error;record[task+'_gpu_replay_error']=delta
        ts=metrics(predictions(model,train,norm,torch.device('cuda')),train,r['threshold'])[2]
        record.update(train_f1=ts['macro_f1_tol1'],validation_f1=r['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1'],train_ap=ts['raw_ap'])
        rows.append(record);pd.DataFrame(rows).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    assert before=={str(p):sha(p) for p in cps} and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',before);write(OUT/'completion_audit.json',dict(status='complete',runs=8,checkpoints=16,start_and_end_full_gpu_replays=16,all_metrics_recomputed=True,main_checkpoint_selection_verified=True,train_inputs_and_main_labels_unchanged=True,train_normalizers_recomputed=True,source_and_checkpoint_hashes_unchanged=True,test_predictions_accessed=False,training_runs=0,seconds=time.monotonic()-began))
    state.update(status='complete',pid=None);write(OUT/'STATE.json',state)


if __name__=='__main__':main()
