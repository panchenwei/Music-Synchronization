import torch
from src.external_audio_trial import make_model


def test_tiny_overfit_and_checkpoint(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(812);x=torch.randn(2,16,121);y=(x[...,0]>0).float()
    model=make_model(42);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    model.eval();before=float(torch.nn.functional.binary_cross_entropy_with_logits(model(x),y).detach())
    for _ in range(120):
        model.train();opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y);assert torch.isfinite(loss);loss.backward();opt.step()
    model.eval();prediction=model(x).detach();after=float(torch.nn.functional.binary_cross_entropy_with_logits(prediction,y));assert after<before*.2
    path=tmp_path/'tiny.pt';torch.save(model.state_dict(),path);other=make_model(42).eval();other.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(prediction,other(x),atol=0,rtol=0)
