"""Fold-isolated PCA, before downstream classification; no label fitting."""
import hashlib,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import external_audio_trial as base
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_mert_trial';ART=ROOT/'artifacts/external_mert_trial'


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    source=ROOT/'reports/external_mert_features';assert read(source/'completion_audit.json')['status']=='complete'
    frame=pd.read_csv(source/'feature_manifest.csv');data=base.load_data();sp=read(base.OUT/'splits.json');embeddings={};availability={}
    hashes=dict(read(source/'source_hashes.json'));hashes[str(base.OUT/'splits.json')]=sha(base.OUT/'splits.json')
    for row in frame.itertuples():
        assert sha(row.path)==row.sha256;key=Path(row.path).stem
        with np.load(row.path,allow_pickle=False) as z:embeddings[key]=z['embeddings'].copy();availability[key]=z['availability']>.5
        assert key in data and len(embeddings[key])==len(data[key]['labels'])
    rows=[]
    for f in range(3):
        dest=ART/'cache'/f'fold{f}';dest.mkdir(parents=True,exist_ok=True)
        train=base.subset(data,sp[str(f)]['train']);fit=np.concatenate([embeddings[k][availability[k]] for k in sorted(train)])
        scaler=StandardScaler();scaled=scaler.fit_transform(fit);pca=PCA(n_components=32,svd_solver='randomized',random_state=20260914).fit(scaled)
        ppath=dest/'pca.joblib'
        if ppath.exists():
            old=joblib.load(ppath);np.testing.assert_allclose(pca.components_,old['pca'].components_,atol=1e-6)
            assert old['fit_records']==sorted(train);scaler=old['scaler'];pca=old['pca']
        else:joblib.dump(dict(scaler=scaler,pca=pca,fit_records=sorted(train),fit_groups=sp[str(f)]['train']),ppath)
        hashes[str(ppath)]=sha(ppath)
        for key in sorted(data):
            good=availability[key];true=np.zeros((len(good),32),np.float32);true[good]=pca.transform(scaler.transform(embeddings[key][good])).astype(np.float32)
            shuffled=true.copy();salt=int(hashlib.sha256(f'{key}:MERT-permutation:20260914'.encode()).hexdigest()[:16],16)
            shuffled[good]=np.random.default_rng(salt).permutation(true[good]);assert np.isfinite(true).all()
            path=dest/f'{key}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_allclose(z['E'],true,atol=1e-5);np.testing.assert_allclose(z['F'],shuffled,atol=1e-5)
            else:np.savez_compressed(path,E=true,F=shuffled)
            hashes[str(path)]=sha(path)
        rows.append(dict(fold=f,train_records=len(train),fit_audio_frames=len(fit),components=32,explained_variance=float(pca.explained_variance_ratio_.sum()),labels_used=False))
        assert time.monotonic()-began<600;print('PCA',rows[-1],flush=True)
    for p in (Path(__file__),ROOT/'src/external_audio_trial.py',OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes);pd.DataFrame(rows).to_csv(OUT/'pca_audit.csv',index=False)
    write(OUT/'preparation_audit.json',dict(status='prepared',folds=3,feature_files=213,pca_train_only=True,labels_used=False,hashes_unchanged=True,seconds=time.monotonic()-began))


if __name__=='__main__':
    with threadpool_limits(2):main()
