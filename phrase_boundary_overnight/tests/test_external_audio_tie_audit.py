import mido
import numpy as np
from src.external_audio_tie_audit import xml_sound_events,midi_intervals,continuations_sustain


def test_sound_tie_stop_is_not_a_new_attack(tmp_path):
    path=tmp_path/'t.xml'
    path.write_text('''<score-partwise><part><measure><attributes><divisions>1</divisions></attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><tie type="start"/></note></measure>
    <measure><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><tie type="stop"/></note>
    <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><notations><tied type="stop"/></notations></note>
    </measure></part></score-partwise>''',encoding='utf-8')
    views=xml_sound_events(path)
    assert views[0]['attacks']==((0.,60,2.),)
    assert views[1]['attacks']==((1.,62,1.),) and views[1]['continuations']==((0.,60,1.),)
    assert continuations_sustain(views[1]['continuations'],np.array([[0.,3.,60.]]),2.)
    assert not continuations_sustain(views[1]['continuations'],np.array([[0.,2.5,60.]]),2.)
    assert not continuations_sustain(views[1]['continuations'],np.array([[2.,3.,60.]]),2.)


def test_midi_zero_velocity_off_duration(tmp_path):
    path=tmp_path/'t.mid';m=mido.MidiFile(ticks_per_beat=480);t=mido.MidiTrack();m.tracks.append(t)
    t.extend([mido.Message('note_on',note=60,velocity=80,time=0),mido.Message('note_on',note=60,velocity=0,time=1440)])
    m.save(path);np.testing.assert_array_equal(midi_intervals(path),[[0.,3.,60.]])
