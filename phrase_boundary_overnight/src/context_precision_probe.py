"""Read-only numerical localization after the initial context audit stopped."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from .context_inference_audit import base,normalizer,split_ids,window_logits,OUT,write

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    train=base.dataset(split_ids(0)['train'],'C3');val=base.dataset(split_ids(0)['validation'],'C3');norm=normalizer(train)
    cp=torch.load(base.ART/'checkpoints/C3_seed42_fold0/best.pt',map_location='cpu',weights_only=False)
    rows=[]
    for device,strict in [('cuda',False),('cuda',True),('cpu',True)]:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=not strict
        m=base.make_model('C3',42).to(device).eval();m.load_state_dict(cp['model'])
        with torch.no_grad():
            for pid,item in val.items():
                x=torch.from_numpy(norm.apply(item['curves'][:1]).astype(np.float32)).to(device)
                a=torch.sigmoid(m(x));b=torch.sigmoid(window_logits(m,x));diff=abs(a-b).cpu().numpy()
                rows.append(dict(device=device,strict=strict,piece_id=pid,max_error=float(diff.max()),mean_error=float(diff.mean()),worst_beat=int(diff.argmax())))
    write(OUT/'precision_probe.json',rows)
    for device,strict in [('cuda',False),('cuda',True),('cpu',True)]:
        r=[x for x in rows if x['device']==device and x['strict']==strict]
        print(device,strict,max(x['max_error'] for x in r),flush=True)

if __name__=='__main__':main()
