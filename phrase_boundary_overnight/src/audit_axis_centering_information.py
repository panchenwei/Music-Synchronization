"""Reuse the frozen information probe with isolated centered-model outputs."""
import traceback
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from . import audit_axis_music_information as probe
from . import run_axis_centering_study as centered
from .axis_content_centering import ContentBoundary
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/axis_centering_information';ART=ROOT/'artifacts/axis_centering_information'


def bind():
    # The frozen probe requests data with historical name S. All centered modes
    # use the identical score/curve inputs, so only the factory must vary C/N.
    probe.study=SimpleNamespace(OUT=centered.OUT,ART=centered.ART,KINDS=centered.KINDS,COLS=centered.COLS,
                               dataset=lambda ids,kind:centered.dataset(ids,'C'))
    probe.AxisBoundary=ContentBoundary;probe.OUT=OUT;probe.ART=ART


def main():
    extra={str(p):sha(p) for p in (Path(__file__),ROOT/'tests/test_axis_centering_information.py')}
    assert read(centered.OUT/'completion_audit.json')['status']=='complete'
    bind();probe.main()
    df=pd.read_csv(ART/'summary.csv')
    z=df[df.variant=='zero_embedding'].set_index('run_id')
    s=df[df.variant=='silent_roll'].set_index('run_id')
    cols=centered.COLS+['mean_abs_probability_change']
    assert abs(z[cols]-s[cols]).to_numpy().max()<1e-12
    assert all(sha(p)==h for p,h in extra.items());write(OUT/'adapter_hashes.json',extra)
    write(OUT/'completion_audit.json',{**read(OUT/'completion_audit.json'),'centered_silent_equals_zero_embedding':True,'adapter_hashes_unchanged':True})


if __name__=='__main__':
    try:main()
    except Exception:
        error=dict(status='failed',pid=None,traceback=traceback.format_exc())
        write(OUT/'STATE.json',error);write(OUT/'completion_audit.json',error);raise
