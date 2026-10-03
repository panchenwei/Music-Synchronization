"""Inference perturbations: distinguish stability from ignoring pitch inputs."""
import argparse,time
import numpy as np
import pandas as pd
import torch
from .pitch_invariance_study import OUT,ART,read,sha,score_dataset,make_model,split_ids
from .models import Normalizer
from sklearn.metrics import average_precision_score

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--first-cell',action='store_true');args=parser.parse_args();torch.set_num_threads(2);started=time.monotonic()
    path=OUT/'pitch_usage.csv';rows=pd.read_csv(path).to_dict('records') if path.exists() else [];done={r['run_id'] for r in rows}
    for p in sorted((ART/'metrics').glob('*_fold*.json')):
        r=read(p);run=r['run_id']
        if run in done or (args.first_cell and (r['fold']!=0 or r['seed']!=42)):continue
        cp=ART/'checkpoints'/run/'best.pt';before=sha(cp);s=torch.load(cp,map_location='cpu',weights_only=False)
        m=make_model('P',r['seed']).eval();m.load_state_dict(s['model']);norm=Normalizer(s['mean'],s['std']);data=score_dataset(split_ids(r['fold'])['validation'],'P');pr=[]
        for pid,item in sorted(data.items()):
            perfs=list(item['performance_ids'].astype(str));perf=sorted(perfs)[len(perfs)//2];i=perfs.index(perf)
            x=norm.apply(item['curves'][i]).astype(np.float32);valid=item['label_mask'].astype(bool);y=item['labels'][valid]
            versions=np.repeat(x[None],3,axis=0);versions[1,:,34:]=0
            # Same deterministic time permutation for all run types and both blocks.
            order=np.random.default_rng(12026).permutation(len(x));versions[2,:,34:]=x[order,34:]
            with torch.no_grad():prob=torch.sigmoid(m(torch.from_numpy(versions))).numpy()
            ap=[average_precision_score(y,v[valid]) for v in prob]
            pr.append(dict(piece_id=pid,performance_id=perf,original_ap=ap[0],mean_ap=ap[1],shuffle_ap=ap[2],
                mean_probability_abs_delta=float(abs(prob[1,valid]-prob[0,valid]).mean()),shuffle_probability_abs_delta=float(abs(prob[2,valid]-prob[0,valid]).mean())))
        assert sha(cp)==before
        df=pd.DataFrame(pr);df.to_csv(OUT/f'{run}_usage_pieces.csv',index=False)
        weight=m.input_projection.weight.detach().numpy();row=dict(run_id=run,kind=r['kind'],fold=r['fold'],seed=r['seed'],works=len(df),
            pitch_weight_norm=float(np.linalg.norm(weight[:,34:])),old_weight_norm=float(np.linalg.norm(weight[:,:34])),**df.select_dtypes(include='number').mean().to_dict())
        rows.append(row);pd.DataFrame(rows).to_csv(path,index=False);print(row,flush=True)
    print('INFERENCE_ONLY_SECONDS',time.monotonic()-started,flush=True)

if __name__=='__main__':main()
