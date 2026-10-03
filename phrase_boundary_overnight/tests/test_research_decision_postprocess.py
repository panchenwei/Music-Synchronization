import numpy as np
import pytest
from src.research_decision_postprocess import ScoreCoverageDecision
from src.recurrence_message_probe import build_graph,mix_probabilities
from src.recurrence_mean_control import build_mean_graph
from src.interstart_decoder import decode

def example():
    events=np.array([(i,.8,60+i%7,s) for i in range(72) for s in (1,2)],float)
    prior={'mean':18.,'histogram':(np.ones(256)/256).tolist()}
    return events,prior

def test_interface_equals_frozen_mechanism():
    events,prior=example();processor=ScoreCoverageDecision(events,72);p=np.linspace(.01,.99,72)
    result=processor.predict(p,.55,prior)
    expected=mix_probabilities(p,build_mean_graph(build_graph(events,72),'ignored','M'))
    np.testing.assert_array_equal(result['adjusted_probabilities'],expected)
    np.testing.assert_array_equal(result['boundaries'],decode(expected,.55,prior,1.))

def test_bad_coordinates_are_rejected():
    events,prior=example();processor=ScoreCoverageDecision(events,72)
    with pytest.raises(ValueError):processor.predict(np.zeros(71),.55,prior)
    with pytest.raises(ValueError):processor.predict(np.zeros(72),1.,prior)
    with pytest.raises(ValueError):ScoreCoverageDecision(np.ones((3,3)),72)
