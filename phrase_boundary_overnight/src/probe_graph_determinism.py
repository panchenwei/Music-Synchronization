"""Training-work-only localization of non-repeatable graph inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
import numpy as np
import torch
from .run_preaggregate_note_study import OUT,ART,dataset
from .preaggregate_note_graph import PreAggregateBoundary
from .note_relation_graph import tensors
from .phrase_end_auxiliary import write,sha,normalizer,split_ids


def main():
    torch.set_num_threads(2);began=time.monotonic();pid=split_ids(0)['train'][0];data=dataset([pid]);v=data[pid];norm=normalizer(data)
    path=ART/'checkpoints/N_seed42_fold0/best.pt';before=sha(path);c=torch.load(path,map_location='cpu',weights_only=False)
    m=PreAggregateBoundary(42).cuda().eval();m.load_state_dict(c['model']);g=tensors(v['graph'],'cuda');x=torch.from_numpy(norm.apply(v['curves'][:8]).astype(np.float32)).cuda();rows=[]
    with torch.no_grad():
        frozen=m.encode_beats(g)[None]
        for deterministic in (False,True):
            torch.use_deterministic_algorithms(deterministic);observations={k:[] for k in ('nodes','beats','output','fixed_graph_output')}
            for _ in range(10):
                observations['nodes'].append(m.encode_nodes(g).cpu().numpy())
                observations['beats'].append(m.encode_beats(g).cpu().numpy())
                observations['output'].append(m(x,g).cpu().numpy())
                observations['fixed_graph_output'].append(m.forward_embedded(x,frozen).cpu().numpy())
            rows.append(dict(deterministic=deterministic,**{k:max(float(abs(a-values[0]).max()) for a in values) for k,values in observations.items()}))
    assert sha(path)==before
    write(OUT/'replay_diagnosis/determinism_probe.json',dict(status='complete',training_piece=pid,repeats=10,seconds=time.monotonic()-began,results=rows,checkpoint_unchanged=True,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],validation_accessed=False))
    print(rows,flush=True)


if __name__=='__main__':main()
