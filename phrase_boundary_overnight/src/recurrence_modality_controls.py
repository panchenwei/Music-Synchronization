"""Remove performance VALUES while preserving all quality/availability flags."""
import numpy as np

VALUE_COLUMNS = {'O': [], 'D': list(range(25,31)), 'S': list(range(7))+list(range(25,31))}


def control_item(item, kind):
    assert kind in VALUE_COLUMNS and item['curves'].shape[-1] == 58
    v = dict(item)
    v['curves'] = item['curves'].copy()
    v['round2_extra'] = item['round2_extra'].copy()
    v['curves'][..., VALUE_COLUMNS[kind]] = 0
    if kind in ('D','S'): v['round2_extra'][..., :6] = 0
    # Raw energy may remain as provenance, but forward uses curves and normalizer
    # uses round2_extra, both explicitly corrected above. No recomputation from it.
    return v
