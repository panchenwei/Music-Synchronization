"""Frozen MERT embeddings on reference-aligned beat cells; label-blind extraction."""
import time,hashlib
from pathlib import Path
from math import gcd
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch
from threadpoolctl import threadpool_limits
from .mert_local import load_model,normalized_waveform,frame_centres
from .score_context_study import ROOT,read,write,sha
from .external_audio_features import aggregate_frames

OUT=ROOT/'reports/external_mert_features';ART=ROOT/'artifacts/external_mert_features';SR=24000


def chunk_schedule(length):
    for start in range(0,length,3*SR):
        if start and start+SR>=length:break
        stop=min(start+5*SR,length)
        if stop-start<400:continue
        lo=0 if start==0 else start+SR;hi=min(start+4*SR,length)
        yield start,stop,lo,hi


def extract(model,path,seconds):
    info=sf.info(path);valid=seconds[np.isfinite(seconds)];first=max(0,int(np.floor((float(valid.min())-2)*info.samplerate)))
    last=min(info.frames,int(np.ceil((float(valid.max())+2)*info.samplerate)))
    wave,fs=sf.read(path,start=first,stop=last,always_2d=True,dtype='float32');wave=wave.mean(1);d=gcd(fs,SR)
    wave=resample_poly(wave,SR//d,fs//d).astype(np.float32);times=[];frames=[];chunks=0
    for start,stop,lo,hi in chunk_schedule(len(wave)):
        x=torch.from_numpy(normalized_waveform(wave[start:stop]))[None].cuda()
        with torch.inference_mode():hidden=model(x).last_hidden_state[0].float().cpu().numpy()
        centres=frame_centres(stop-start)+start/SR;assert hidden.shape==(len(centres),768) and np.isfinite(hidden).all()
        keep=(centres>=lo/SR)&(centres<hi/SR);times.append(centres[keep]+first/fs);frames.append(hidden[keep]);chunks+=1
    t=np.concatenate(times);z=np.concatenate(frames);assert (np.diff(t)>0).all()
    result,available=aggregate_frames(t,z.T,seconds)
    return result,available,chunks


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/mert_vendor/smoke.json')['status']=='passed'
    model=load_model();torch.set_num_threads(2);source=ROOT/'reports/external_audio_features/feature_manifest.csv'
    data=pd.read_csv(source);grids=ROOT/'reports/external_audio_masks_v2/grid_manifest.csv';grid=pd.read_csv(grids).set_index('performance')
    hashes={str(p):sha(p) for p in (Path(__file__),ROOT/'src/mert_local.py',ROOT/'src/external_audio_features.py',source,grids,OUT/'PROTOCOL.md',ROOT/'tests/test_mert_features.py')}
    vendor=read(ROOT/'reports/mert_vendor/weights_manifest.json')
    for entry in vendor['files']:hashes[str(ROOT/'external_runtime/mert_reviewed'/entry['file'])]=entry['sha256']
    hashes.update(read(ROOT/'reports/external_audio_features/source_hashes.json'));rows=[]
    for r in data.sort_values('performance').itertuples():
        assert time.monotonic()-began<2400,'40-minute embedding cap; resume completed caches, do not increase cap blindly'
        record=grid.loc[r.performance];assert sha(record.npz)==record.sha256
        with np.load(record.npz,allow_pickle=False) as z:seconds=z['audio_seconds'].copy();mask=z['label_mask'].copy()
        dest=ART/(hashlib.sha256(r.performance.encode()).hexdigest()[:16]+'.npz');start=time.monotonic()
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                embeddings=z['embeddings'].copy();available=z['availability'].copy();chunks=int(z['chunks'])
                assert str(z['extractor_hash'])==sha(Path(__file__)) and str(z['source_grid_hash'])==record.sha256
        else:
            embeddings,available,chunks=extract(model,record.audio_path,seconds)
            assert embeddings.shape==(r.beats,768) and np.isfinite(embeddings).all()
            np.savez_compressed(dest,embeddings=embeddings,availability=available,chunks=chunks,source_grid_hash=record.sha256,extractor_hash=sha(Path(__file__)))
        assert int((mask*available).sum())==int(mask.sum()),'Known audio coverage changed; do not silently drop labels'
        hashes[str(dest)]=sha(dest);rows.append(dict(performance=r.performance,piece_id=r.piece_id,group=r.group,path=str(dest),sha256=sha(dest),
            beats=r.beats,known=int(mask.sum()),covered=int((mask*available).sum()),chunks=chunks,seconds=time.monotonic()-start))
        pd.DataFrame(rows).to_csv(OUT/'feature_manifest.csv',index=False);print('MERT',len(rows),r.performance,'seconds',round(rows[-1]['seconds'],2),flush=True)
    assert len(rows)==71 and all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',records=71,features=768,trainable_encoder_parameters=0,known=sum(r['known'] for r in rows),
        covered=sum(r['covered'] for r in rows),chunks=sum(r['chunks'] for r in rows),hashes_unchanged=True,seconds=time.monotonic()-began,
        pretraining_overlap_unknown=True,reference_alignment=True,downstream_training_runs=0))


if __name__=='__main__':
    with threadpool_limits(2):main()
