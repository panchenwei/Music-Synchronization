"""Offline, reviewed, weights-only MERT loader with isolated dependencies."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HOME']=str(ROOT/'external_runtime/mert_hf_cache')
sys.path.insert(0,str(ROOT/'external_runtime/mert_dependencies'));sys.path.insert(0,str(ROOT/'external_runtime'))
import numpy as np
import torch
from .score_context_study import read,sha,write


def load_model():
    folder=ROOT/'external_runtime/mert_reviewed';manifest=read(ROOT/'reports/mert_vendor/weights_manifest.json')
    assert manifest['revision']=='12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
    assert all(sha(folder/r['file'])==r['sha256'] for r in manifest['files'])
    from mert_reviewed.configuration_MERT import MERTConfig
    from mert_reviewed.modeling_MERT import MERTModel
    config=MERTConfig(**read(folder/'config.json'));model=MERTModel(config)
    weights=torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True)
    result=model.load_state_dict(weights,strict=True);assert not result.missing_keys and not result.unexpected_keys
    return model.requires_grad_(False).eval().cuda()


def normalized_waveform(y):
    y=np.asarray(y,np.float32);assert y.ndim==1 and len(y)>=400 and np.isfinite(y).all()
    return (y-y.mean())/np.sqrt(y.var()+1e-7)


def frame_centres(length,start_seconds=0.):
    # Seven valid convolutions: receptive field400, output stride320 samples.
    n=(length-400)//320+1
    return start_seconds+(np.arange(max(n,0))*320+199.5)/24000


def main():
    import time
    start=time.monotonic();torch.set_num_threads(2);model=load_model();fs=24000
    y=np.sin(2*np.pi*440*np.arange(fs*5)/fs).astype(np.float32);x=torch.from_numpy(normalized_waveform(y))[None].cuda()
    with torch.inference_mode():
        a=model(x).last_hidden_state;b=model(x).last_hidden_state;c=model(torch.zeros_like(x)).last_hidden_state
    assert a.shape==(1,len(frame_centres(len(y))),768) and torch.isfinite(a).all() and torch.isfinite(c).all()
    assert torch.equal(a,b) and not torch.allclose(a,c)
    from transformers import __version__
    result=dict(status='passed',params=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        transformer_version=__version__,shape=list(a.shape),finite_silence=True,repeat_equal=True,nonconstant_output=True,seconds=time.monotonic()-start,
        caveat='Sine/silence compatibility check, not musical task performance.')
    write(ROOT/'reports/mert_vendor/smoke.json',result);print(result,flush=True)


if __name__=='__main__':main()
