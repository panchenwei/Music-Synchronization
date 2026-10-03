import torch
from src.context_inference_audit import segments,offset_logits,window_logits
from src.pure_transformer_reference import make_model
from src.recurrence_depth_models import make_model as cnn

def test_segments_cover_with_context():
    for n in (1,33,63,64,65,97,140,999):
        cover=torch.zeros(n)
        for lo,hi,a,b in segments(n):
            assert 0<=lo<=a<b<=hi<=n and hi-lo<=64
            if lo>0:assert a-lo>=16
            if hi<n:assert hi-b>=16
            cover[a:b]+=1
        assert (cover==1).all()

def test_offset_forward_equals_original_at_zero():
    m=make_model('T',42).eval();x=torch.randn(2,64,58)
    with torch.no_grad():torch.testing.assert_close(m(x),offset_logits(m,x,0),atol=1e-6,rtol=1e-6)
    assert not torch.allclose(offset_logits(m,x,0),offset_logits(m,x,64))

def test_cnn_stitch_does_not_change_predictions():
    m=cnn('C3',42).eval();x=torch.randn(2,171,58)
    # Nonzero deeper residual paths ensure the full receptive field is exercised.
    for layer in m.frontend.layers:
        torch.nn.init.normal_(layer.pointwise.weight,std=.03)
    with torch.no_grad():torch.testing.assert_close(m(x),window_logits(m,x),atol=2e-6,rtol=2e-6)

def test_short_transformer_identical():
    m=make_model('T',42).eval();x=torch.randn(1,43,58)
    with torch.no_grad():torch.testing.assert_close(m(x),window_logits(m,x),atol=1e-6,rtol=1e-6)
