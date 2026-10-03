import copy
import numpy as np
from src.external_interval_probe import fit_prior


def example(piece,audio,starts):
    y=np.zeros(32);y[starts]=1
    return dict(piece_id=piece,audio_hash=audio,labels=y,label_mask=np.ones(32))


def test_work_balance_duplicate_source_and_unknown_gaps():
    data={'a':example('one','audio1',[2,6,10]),'b':example('two','audio2',[2,10,18])};p=fit_prior(data);assert p['works']==2 and p['mean']==6
    data['duplicate']=copy.deepcopy(data['a']);q=fit_prior(data);np.testing.assert_array_equal(p['histogram'],q['histogram']);assert p['mean']==q['mean']
    data['b']['label_mask'][7:9]=0;r=fit_prior(data);assert r['intervals']==3 and r['mean']==6
