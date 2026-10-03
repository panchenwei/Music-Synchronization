"""Label-free experimental score-coverage postprocessor for offline C3 output.

This is a research candidate, not a calibrated probability estimator. It accepts
already aligned quarter-note probabilities; it does not perform score alignment
or run the neural network. Preserve the historical C3 baseline alongside it.
"""
import numpy as np
from .recurrence_message_probe import build_graph,mix_probabilities
from .recurrence_mean_control import build_mean_graph
from .interstart_decoder import decode

class ScoreCoverageDecision:
    """Prepare once per score, then apply separately to each performance.

    note_events: tie-merged rows beginning onset_quarterbeat, duration, MIDI pitch,
    notated staff. Units and unfolding must match the upstream probability axis.
    threshold and prior must come from the permitted training/development flow;
    no labels, gold phrase lengths, or another performance's output are accepted.
    """
    def __init__(self,note_events,n_beats):
        events=np.asarray(note_events)
        if events.ndim!=2 or events.shape[1]<4 or not np.isfinite(events[:,:4]).all():
            raise ValueError('Expected finite tie-merged note-event rows with >=4 columns')
        if not isinstance(n_beats,(int,np.integer)) or n_beats<1:
            raise ValueError('n_beats must be a positive integer')
        self.n_beats=int(n_beats)
        self.recurrence_graph=build_graph(events,self.n_beats)
        self.mean_graph=build_mean_graph(self.recurrence_graph,'not_used_for_M','M')

    def predict(self,probabilities,threshold,train_prior,*,strength=1.0):
        if not np.isfinite(threshold) or not 0<threshold<1:
            raise ValueError('threshold must be between zero and one')
        if not np.isfinite(strength) or strength<0:
            raise ValueError('strength must be finite and nonnegative')
        p=np.asarray(probabilities,dtype=np.float64)
        if p.shape!=(self.n_beats,):raise ValueError('Probability and score timelines must match')
        adjusted=mix_probabilities(p,self.mean_graph)
        return {'adjusted_probabilities':adjusted,'boundaries':decode(adjusted,float(threshold),train_prior,float(strength)),
                'coverage':self.mean_graph.sum(1)>0,'coordinate':'quarter-note score position',
                'status':'development candidate; not independently validated'}
