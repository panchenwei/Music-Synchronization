import numpy as np
from src.recurrence_modality_controls import control_item, VALUE_COLUMNS
from src.score_context_study import normalizer


def item():
    rng=np.random.default_rng(3)
    score=rng.random((20,16)).astype('float32');p=rng.random((20,24)).astype('float32')
    curves=rng.random((3,20,58)).astype('float32');curves[...,9:25]=score;curves[...,34:]=p
    return dict(curves=curves,selected_score=score,pitch_profiles=p,round2_extra=curves[...,25:34].copy(),
                labels=np.zeros(20,'float32'),label_mask=np.ones(20,'float32'))


def test_only_declared_values_change_and_source_is_not_mutated():
    v=item();before=v['curves'].copy()
    for k in ('O','D','S'):
        w=control_item(v,k);keep=[i for i in range(58) if i not in VALUE_COLUMNS[k]]
        np.testing.assert_array_equal(w['curves'][...,keep],before[...,keep])
        assert not w['curves'][...,VALUE_COLUMNS[k]].any()
        np.testing.assert_array_equal(w['curves'][...,[7,8,31,33]],before[...,[7,8,31,33]])
        np.testing.assert_array_equal(w['curves'][...,25:34],w['round2_extra'])
    np.testing.assert_array_equal(v['curves'],before)


def test_removed_values_are_also_zero_after_train_normalization():
    v=item();old=normalizer({'x':v})
    for k in ('D','S'):
        w=control_item(v,k);norm=normalizer({'x':w});removed=VALUE_COLUMNS[k]
        keep=[i for i in range(58) if i not in removed]
        np.testing.assert_array_equal(norm.mean[keep],old.mean[keep]);np.testing.assert_array_equal(norm.std[keep],old.std[keep])
        assert not norm.apply(w['curves'])[...,removed].any()
        np.testing.assert_array_equal(w['labels'],v['labels']);np.testing.assert_array_equal(w['label_mask'],v['label_mask'])
