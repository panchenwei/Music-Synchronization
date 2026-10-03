"""Label-blind score/RMS/CQT observables on the audited external beat grid."""
import hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import resample_poly
from math import gcd
from threadpoolctl import threadpool_limits
from .score_context_study import ROOT,read,write,sha,pitch_profiles
from .external_audio_candidates import ASAP
from .external_audio_tie_audit import midi_intervals

OUT=ROOT/'reports/external_audio_features';ART=ROOT/'artifacts/external_audio_features'
SR=22050;HOP=512;BINS=88


def score_features(intervals,n):
    events=[(a,b-a,int(p)) for a,b,p in intervals if b>a]
    profile=pitch_profiles(events,n);extra=np.zeros((n,4),np.float32)
    for q in range(n):
        active=[(a,d,p) for a,d,p in events if a<q+1 and a+d>q]
        attacks=[(a,d,p) for a,d,p in active if a>=q-1e-5]
        extra[q]=[np.log1p(len(attacks)),np.log1p(len(active)),np.mean([p for a,d,p in active])/127 if active else 0,
                  np.log1p(np.mean([d for a,d,p in attacks])) if attacks else 0]
    return np.c_[profile,extra].astype(np.float32)


def aggregate_frames(frame_times,features,seconds):
    """Average in non-overlapping local half-beat cells; no label-mask argument."""
    n=len(seconds);output=np.zeros((n,features.shape[0]),np.float32);available=np.zeros(n,np.float32)
    good=np.isfinite(seconds)
    for q in range(1,n-1):
        if not good[q-1:q+2].all():continue
        lo=(seconds[q-1]+seconds[q])*.5;hi=(seconds[q]+seconds[q+1])*.5
        if not lo<seconds[q]<hi:continue
        a,b=np.searchsorted(frame_times,[lo,hi],side='left')
        if b>a and lo>=frame_times[0] and hi<=frame_times[-1]:output[q]=features[:,a:b].mean(1);available[q]=1
    return output,available


def audio_features(path,seconds):
    info=sf.info(path);valid=np.asarray(seconds)[np.isfinite(seconds)]
    lo=max(0.,float(valid.min())-2);hi=min(info.duration,float(valid.max())+2)
    first=max(0,int(np.floor(lo*info.samplerate)));last=min(info.frames,int(np.ceil(hi*info.samplerate)))
    y,fs=sf.read(path,start=first,stop=last,always_2d=True,dtype='float32');y=y.mean(1)
    d=gcd(fs,SR);y=resample_poly(y,SR//d,fs//d).astype(np.float32)
    c=np.abs(librosa.cqt(y=y,sr=SR,hop_length=HOP,fmin=27.5,n_bins=BINS,bins_per_octave=12)).astype(np.float32)
    rms=librosa.feature.rms(y=y,frame_length=2048,hop_length=HOP)[0]
    count=min(c.shape[1],len(rms));c=c[:,:count];rms=rms[:count]
    power=c*c;logspec=np.log(np.maximum(power,1e-10));logrms=np.log(np.maximum(rms,1e-7))
    unit=c/np.maximum(c.sum(0,keepdims=True),1e-8)
    flux=np.r_[0,np.maximum(np.diff(unit,axis=1),0).sum(0)]
    slope=np.r_[0,np.diff(logrms)];centroid=(unit*np.linspace(0,1,BINS)[:,None]).sum(0)
    frames=np.vstack([logrms,slope,flux,centroid,logspec]);times=first/fs+np.arange(count)*HOP/SR
    return aggregate_frames(times,frames,seconds)


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    groups=ROOT/'reports/external_audio_group_audit/prospective_group_manifest.csv'
    grids=ROOT/'reports/external_audio_masks_v2/grid_manifest.csv'
    candidates=ROOT/'reports/external_audio_candidates/audio_candidates.csv'
    table=pd.read_csv(groups);manifest=pd.read_csv(grids).set_index('performance');candidate=pd.read_csv(candidates).set_index('asap_performance')
    hashes={str(p):sha(p) for p in (Path(__file__),groups,grids,candidates,OUT/'PROTOCOL.md',ROOT/'tests/test_external_audio_features.py',ROOT/'src/external_audio_tie_audit.py',ROOT/'src/score_context_study.py')}
    audiohash=pd.read_csv(ROOT/'reports/external_audio_group_audit/audio_source_hashes.csv')
    for r in audiohash.itertuples():assert sha(r.audio_path)==r.sha256;hashes[r.audio_path]=r.sha256
    hashes.update(read(ROOT/'reports/external_audio_masks_v2/source_hashes.json'))
    assert all(sha(p)==h for p,h in hashes.items())
    rows=[]
    for r in table.sort_values('performance').itertuples():
        assert time.monotonic()-started<2400,'Feature extraction cap; cached completed records can be resumed'
        name=hashlib.sha256(r.performance.encode()).hexdigest()[:16];dest=ART/f'{name}.npz'
        src=manifest.loc[r.performance];assert sha(src.npz)==src.sha256;hashes[src.npz]=src.sha256
        score=ASAP/candidate.loc[r.performance,'asap_folder']/'midi_score.mid';hashes[str(score)]=sha(score)
        with np.load(src.npz,allow_pickle=False) as z:y=z['labels'].copy();mask=z['label_mask'].copy();seconds=z['audio_seconds'].copy()
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                x=z['features'].copy();np.testing.assert_array_equal(z['labels'],y);np.testing.assert_array_equal(z['label_mask'],mask)
                assert str(z['source_grid_hash'])==src.sha256 and str(z['extractor_hash'])==sha(Path(__file__))
        else:
            s=score_features(midi_intervals(score),len(y));a,available=audio_features(src.audio_path,seconds)
            x=np.c_[s,a,available].astype(np.float32)
            np.savez_compressed(dest,features=x,labels=y,label_mask=mask,source_grid_hash=src.sha256,extractor_hash=sha(Path(__file__)))
        assert x.shape==(len(y),121) and np.isfinite(x).all()
        with np.load(dest,allow_pickle=False) as z:np.testing.assert_array_equal(z['features'],x)
        rows.append(dict(performance=r.performance,piece_id=r.piece_id,group=r.independence_group,audio_sha256=r.audio_sha256,path=str(dest),sha256=sha(dest),
            beats=len(y),known=int(mask.sum()),positive=int(y.sum()),audio_known=int((x[:,-1]*mask).sum())))
        pd.DataFrame(rows).to_csv(OUT/'feature_manifest.csv',index=False);print('EXTRACTED',len(rows),r.performance,flush=True)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    result=dict(status='complete',records=len(rows),pieces=int(table.piece_id.nunique()),groups=int(table.independence_group.nunique()),
        features=121,known=sum(r['known'] for r in rows),audio_known=sum(r['audio_known'] for r in rows),
        training_runs=0,hashes_unchanged=True,seconds=time.monotonic()-started,
        limitation='Uses provided ASAP reference beat alignment, not estimated old-pipeline alignment. No CQT or RMS inference accuracy yet.')
    write(OUT/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
