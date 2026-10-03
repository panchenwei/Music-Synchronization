"""Signed, mask-aware Haar-like mean contrasts; scale is not acoustic pitch."""
import numpy as np

SCALES=(1,2,4,8,16)


def contrast(values,observed,scale):
    values=np.asarray(values,dtype=np.float64);observed=np.asarray(observed,dtype=bool)
    assert values.ndim==1 and observed.shape==values.shape and scale>=1
    n=len(values);out=np.zeros(n,np.float32);quality=np.zeros(n,np.float32)
    ok=observed&np.isfinite(values);v=np.where(ok,values,0)
    sums=np.r_[0.,np.cumsum(v)];counts=np.r_[0,np.cumsum(ok)]
    # No partial geometric windows or wraparound at the score endpoints.
    for b in range(scale,n-scale+1):
        nl=counts[b]-counts[b-scale];nr=counts[b+scale]-counts[b]
        if min(nl,nr)<max(1,(scale+1)//2):continue
        out[b]=(sums[b+scale]-sums[b])/nr-(sums[b]-sums[b-scale])/nl
        quality[b]=min(nl,nr)/scale
    return out,quality


def transform(curves):
    curves=np.asarray(curves,np.float32)
    assert curves.ndim==3 and curves.shape[-1]>=9
    p,n=curves.shape[:2];result=np.zeros((p,n,4,len(SCALES)),np.float32)
    for j,row in enumerate(curves):
        time=row[:,7]>.5;tempo_ok=np.zeros(n,bool)
        tempo_ok[:-1]=time[:-1]&time[1:]
        # Last log-tempo is an estimate, not an observed outgoing interval.
        for channel,index,valid in ((0,0,tempo_ok),(1,3,row[:,8]>.5)):
            for si,scale in enumerate(SCALES):
                result[j,:,channel,si],result[j,:,channel+2,si]=contrast(row[:,index],valid,scale)
    assert np.isfinite(result).all()
    return result
