"""Input-only pitch-profile shift diagnostics; no labels or model selection."""
import numpy as np
import pandas as pd
from .score_context_study import OUT,ART,split_ids
from .phase3_models import fit_train_normalizer

def cosine(a,b):
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))

def main():
    rows=[];scale=[]
    for fold in (0,1):
        ids=split_ids(fold);train={p:np.load(ART/'cache'/f'{p}.npy') for p in ids['train']}
        norm=fit_train_normalizer(list(train.values()))
        refs=[x[:,:12].mean(0) for x in train.values()]
        for p in ids['validation']:
            x=np.load(ART/'cache'/f'{p}.npy');v=x[:,:12].mean(0);z=norm.apply(x)
            raw=max(cosine(v,r) for r in refs);shift=max(cosine(np.roll(v,k),r) for r in refs for k in range(12))
            rows.append(dict(fold=fold,piece_id=p,raw_nearest_train_cosine=raw,best_rotation_nearest_cosine=shift,
                             rotation_improvement=shift-raw,z_abs_gt3_fraction=float((abs(z)>3).mean()),z_abs_max=float(abs(z).max())))
        scale.extend(dict(fold=fold,column=i,mean=float(norm.mean[i]),std=float(norm.std[i])) for i in range(24))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'pitch_profile_shift.csv',index=False);pd.DataFrame(scale).to_csv(OUT/'pitch_normalizer.csv',index=False)
    print(frame.groupby('fold')[['raw_nearest_train_cosine','best_rotation_nearest_cosine','rotation_improvement','z_abs_gt3_fraction','z_abs_max']].mean().to_string())

if __name__=='__main__':main()
