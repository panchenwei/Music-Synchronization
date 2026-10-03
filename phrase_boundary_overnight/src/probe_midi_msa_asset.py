"""Compatibility-only smoke test; no target data, metrics or external code exec."""
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
import torch
from torch import nn


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'external_data/midi_msa_runtime'))
    import torchvision
    path = root / 'external_data/midi_msa_probe/pretrained_no_overtones_no_drums.pt'
    data = path.read_bytes()
    git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert git_blob == 'bbcb3a1132aade42a6ccb2e0cf4502e18a9ba5cd'
    state = torch.load(path, map_location='cpu', weights_only=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    network = torchvision.models.mobilenet_v3_small(weights=None)
    network.classifier[-1] = nn.Sequential(nn.Linear(network.classifier[-1].in_features, 1))
    model = nn.Sequential(OrderedDict([('backbone', network)])).eval()
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 128, 512))
    assert output.shape == (2, 1) and torch.isfinite(output).all()
    report = dict(status='compatibility_smoke_passed_not_reproduced', torch=torch.__version__, torchvision=torchvision.__version__, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), git_blob_sha1=git_blob, source_commit='09e3fb488a79c28ad957132c11321fa5ec5e8154', source_model_definition='notebooks/evaluation.ipynb BoundaryClassifier', strict_state_load=True, input_shape=[2,3,128,512], output_shape=list(output.shape), parameters=sum(p.numel() for p in model.parameters()), blank_input_logits=output.flatten().tolist(), target_data_accessed=False, target_f1=None, license_status='No explicit repository license found; redistribution/commercial use not cleared', pretraining_target_overlap='not audited; cannot claim independent confirmation from this asset', input_adaptation_status='not implemented', existing_training_environment_modified=False)
    dest = root / 'reports/external_midi_msa/asset_probe.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
