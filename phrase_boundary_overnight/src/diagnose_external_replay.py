"""Investigate a failed cross-device tolerance without changing results or models."""
import numpy as np
import pandas as pd
import torch
from .external_stem_transfer import OUT,ART,read,write,sha,external_dataset,Normalizer,RollBoundary,roll_predictions,metrics
from .audit_external_stem_transfer import checked_raw


def main():
    torch.set_num_threads(2);data=external_dataset('validation');norm=Normalizer(np.zeros(58,np.float32),np.ones(58,np.float32));rows=[];hashes={}
    settings=dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,matmul_precision=torch.get_float32_matmul_precision(),torch_version=torch.__version__)
    for seed in (42,43):
        dest=ART/'external'/f'seed{seed}';p=dest/'best.pt';hashes[str(p)]=sha(p);hashes[str(dest/'validation_predictions.csv.gz')]=sha(dest/'validation_predictions.csv.gz')
        saved=torch.load(p,map_location='cpu',weights_only=False);r=read(dest/'result.json');frame=pd.read_csv(dest/'validation_predictions.csv.gz');raw=checked_raw(frame,data);base_score=metrics(raw,data,r['threshold'])[2]
        replays={}
        for mode,device,tf32 in (('cpu','cpu',True),('gpu_original','cuda',True),('gpu_ieee','cuda',False)):
            torch.backends.cudnn.allow_tf32=tf32;model=RollBoundary('L',seed);model.load_state_dict(saved['model']);model.to(device).eval()
            replay=roll_predictions(model,data,norm,torch.device(device));replays[mode]=replay;score=metrics(replay,data,r['threshold'])[2]
            for pid in data:
                error=float(np.max(abs(replay[pid]['score']-raw[pid]['score'])))
                rows.append(dict(seed=seed,mode=mode,piece_id=pid,max_probability_error=error,threshold_crossings=int(((replay[pid]['score']>=r['threshold'])!=(raw[pid]['score']>=r['threshold'])).sum()),f1_delta_full_validation=score['macro_f1_tol1']-base_score['macro_f1_tol1'],exact_delta_full_validation=score['macro_f1_tol0']-base_score['macro_f1_tol0'],ap_delta_full_validation=score['raw_ap']-base_score['raw_ap']))
        delta=max(float(np.max(abs(replays['cpu'][p]['score']-replays['gpu_ieee'][p]['score']))) for p in data)
        print('seed',seed,'CPU versus IEEE GPU',delta,flush=True)
    torch.backends.cudnn.allow_tf32=settings['cudnn_tf32']
    assert all(sha(p)==h for p,h in hashes.items());df=pd.DataFrame(rows);df.to_csv(OUT/'numerical_replay_diagnostic.csv',index=False);write(OUT/'numerical_replay_environment.json',settings);write(OUT/'numerical_replay_hashes.json',hashes)
    print(df.groupby(['seed','mode'])[['max_probability_error','threshold_crossings','f1_delta_full_validation','exact_delta_full_validation','ap_delta_full_validation']].max().to_string(),flush=True)


if __name__=='__main__':main()
