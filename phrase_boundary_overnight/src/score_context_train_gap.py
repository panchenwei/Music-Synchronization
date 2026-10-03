"""Inference-only train-gap audit for already completed runs; no selection changes."""
import argparse,time
import numpy as np
import pandas as pd
import torch
from .score_context_study import OUT,ART,read,dataset,make_model,split_ids
from .models import Normalizer
from .local_context_study import predictions,metrics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--first-cell',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    output=OUT/'train_gap_diagnostics.csv';old=pd.read_csv(output) if output.exists() else pd.DataFrame();done=set(old.run_id) if len(old) else set();rows=old.to_dict('records')
    for p in sorted((ART/'metrics').glob('*_fold*.json')):
        r=read(p);run=r['run_id']
        if run in done or (args.first_cell and (r['seed']!=42 or r['fold']!=0)):continue
        start=time.monotonic();s=torch.load(ART/'checkpoints'/run/'best.pt',map_location='cpu',weights_only=False);model=make_model(r['kind'],r['seed']).eval();model.load_state_dict(s['model']);norm=Normalizer(s['mean'],s['std'])
        data=dataset(split_ids(r['fold'])['train'],r['kind']);raw=predictions(model,data,norm,torch.device('cpu'));_,_,score=metrics(raw,data,r['threshold'])
        row=dict(run_id=run,kind=r['kind'],fold=r['fold'],seed=r['seed'],train_f1=score['macro_f1_tol1'],validation_f1=r['macro_f1_tol1'],gap=score['macro_f1_tol1']-r['macro_f1_tol1'],train_ap=score['raw_ap'],validation_ap=r['raw_ap'],diagnostic_seconds=time.monotonic()-start)
        rows.append(row);pd.DataFrame(rows).to_csv(output,index=False);print(row,flush=True)

if __name__=='__main__':main()
