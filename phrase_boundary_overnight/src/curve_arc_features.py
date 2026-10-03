"""Two-sided linear trend and residual features; no boundary labels are read."""
import numpy as np

SCALES=(4,8,16)

def side_trends(values, observed, scale):
    x=np.asarray(values,dtype=np.float64)
    ok=np.asarray(observed,dtype=bool)&np.isfinite(x)
    assert x.ndim==1 and ok.shape==x.shape and scale>=3
    n=len(x); out=np.zeros((n,4),np.float32); quality=np.zeros(n,np.float32)
    if n<2*scale:return out,quality
    windows=np.lib.stride_tricks.sliding_window_view(np.where(ok,x,0.),scale)
    valid=np.lib.stride_tricks.sliding_window_view(ok,scale).all(-1)
    t=np.linspace(-1.,1.,scale)
    slope=windows@t/(t@t)
    residual=windows-windows.mean(-1,keepdims=True)-slope[:,None]*t
    rmse=np.sqrt(np.mean(residual**2,axis=-1))
    b=np.arange(scale,n-scale+1); good=valid[b-scale]&valid[b]
    out[b,0]=slope[b-scale];out[b,1]=slope[b]
    out[b,2]=rmse[b-scale];out[b,3]=rmse[b]
    out[b[~good]]=0;quality[b[good]]=1
    return out,quality

def transform(curves):
    curves=np.asarray(curves,np.float32)
    assert curves.ndim==3 and curves.shape[-1]>=9
    p,n=curves.shape[:2];features=np.zeros((p,n,2,3,5),np.float32)
    for j,row in enumerate(curves):
        tempo_ok=np.zeros(n,bool);tempo_ok[:-1]=(row[:-1,7]>.5)&(row[1:,7]>.5)
        for c,index,valid in ((0,0,tempo_ok),(1,3,row[:,8]>.5)):
            for si,s in enumerate(SCALES):
                features[j,:,c,si,:4],features[j,:,c,si,4]=side_trends(row[:,index],valid,s)
    assert np.isfinite(features).all()
    return features
