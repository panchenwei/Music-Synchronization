import numpy as np
from src.external_mert_features import chunk_schedule,SR
from src.mert_local import frame_centres,normalized_waveform


def test_context_tiling_without_duplicate_core():
    for duration in (1.1,4.5,6.2,9.,11.7):
        n=int(duration*SR);chunks=list(chunk_schedule(n));assert chunks[0][2]==0 and chunks[-1][3]==n
        assert all(a[3]==b[2] for a,b in zip(chunks,chunks[1:]))
        times=[]
        for a,b,lo,hi in chunks:
            t=frame_centres(b-a)+a/SR;times.extend(t[(t>=lo/SR)&(t<hi/SR)])
        assert np.all(np.diff(times)>0) and np.max(np.diff(times))<.03


def test_wave_normalization_and_conv_timing():
    assert len(frame_centres(120000))==374
    assert frame_centres(400)[0]==199.5/24000
    z=normalized_waveform(np.zeros(24000));assert np.isfinite(z).all() and np.max(abs(z))==0
    y=np.sin(np.arange(24000)*.01).astype(np.float32);a=normalized_waveform(y);b=normalized_waveform(y*2)
    np.testing.assert_allclose(a,b,atol=1e-6)
