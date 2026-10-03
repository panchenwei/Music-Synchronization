"""Diagnostic only: measure learned transposition stability, never select a shift."""
import argparse,time
import numpy as np
import pandas as pd
import torch
from .pitch_invariance_study import ROOT,OUT,ART,read,sha,score_dataset,normalization,make_model,split_ids,rotate_windows
from .models import Normalizer
from .phase2_models import nms_probabilities

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--first-cell',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);started=time.monotonic();path=OUT/'rotation_stability.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else [];done={r['run_id'] for r in rows}
    for p in sorted((ART/'metrics').glob('*_fold*.json')):
        r=read(p);run=r['run_id']
        if run in done or (args.first_cell and (r['fold']!=0 or r['seed']!=42)):continue
        cp=ART/'checkpoints'/run/'best.pt';before=sha(cp);s=torch.load(cp,map_location='cpu',weights_only=False)
        m=make_model('P',r['seed']).eval();m.load_state_dict(s['model']);norm=Normalizer(s['mean'],s['std'])
        data=score_dataset(split_ids(r['fold'])['validation'],'P');piece_rows=[]
        for pid,item in sorted(data.items()):
            perfs=list(item['performance_ids'].astype(str));perf=sorted(perfs)[len(perfs)//2];index=perfs.index(perf)
            raw=np.repeat(item['curves'][index:index+1],12,axis=0)
            rotated=rotate_windows(torch.from_numpy(raw),np.arange(12)).numpy()
            x=torch.from_numpy(norm.apply(rotated).astype(np.float32))
            with torch.no_grad():prob=torch.sigmoid(m(x)).numpy()
            valid=item['label_mask'].astype(bool);truth=(item['labels']>0)&valid
            predicted=np.stack([(nms_probabilities(p)>=r['threshold'])&valid for p in prob])
            piece_rows.append(dict(piece_id=pid,performance_id=perf,
                probability_sd_mean=float(prob[:,valid].std(0).mean()),
                probability_abs_delta_vs_original=float(abs(prob[1:,valid]-prob[0,valid]).mean()),
                decoded_beat_disagreement=float((predicted[1:,valid]!=predicted[0,valid]).mean()),
                positive_label_probability_sd=float(prob[:,truth].std(0).mean()) if truth.any() else float('nan')))
        assert sha(cp)==before
        df=pd.DataFrame(piece_rows);df.to_csv(OUT/f'{run}_rotation_pieces.csv',index=False)
        row=dict(run_id=run,kind=r['kind'],fold=r['fold'],seed=r['seed'],works=len(df),shifts=12,
                 **df.select_dtypes(include='number').mean().to_dict())
        rows.append(row);pd.DataFrame(rows).to_csv(path,index=False);print(row,flush=True)
    print('INFERENCE_ONLY_SECONDS',time.monotonic()-started,flush=True)

if __name__=='__main__':main()
