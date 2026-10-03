import mido
import numpy as np
from src.external_audio_projection import midi_timing,seconds_to_quarters,attacks_match
from src.xml_dcml_identity_audit import xml_measures,compare


def test_midi_tempo_inverse_and_attack_rounding(tmp_path):
    m=mido.MidiFile(ticks_per_beat=480);track=mido.MidiTrack();m.tracks.append(track)
    track.extend([mido.MetaMessage('set_tempo',tempo=500000,time=0),mido.Message('note_on',note=60,velocity=80,time=480),
        mido.MetaMessage('set_tempo',tempo=1000000,time=480),mido.Message('note_on',note=64,velocity=80,time=480)])
    p=tmp_path/'tempo.mid';m.save(p);tempo,notes=midi_timing(p)
    np.testing.assert_allclose(seconds_to_quarters([0,.5,1,2,3],tempo),[0,1,2,3,4],atol=1e-10)
    np.testing.assert_allclose(notes,[[1,60],[3,64]])
    signature=((0.,60,1.),(1.,64,1.))
    assert attacks_match(signature,np.array([[1.000001,60],[2.000001,64],[3.,72]]),1.,3.)
    assert not attacks_match(signature,np.array([[1.,60],[2.,65]]),1.,3.)


def test_xml_backup_chords_and_unique_measure_gate(tmp_path):
    p=tmp_path/'score.xml'
    p.write_text('''<score-partwise><part id="P1"><measure number="1"><attributes><divisions>4</divisions></attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
    <note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration></note>
    <note><rest/><duration>4</duration></note><backup><duration>8</duration></backup>
    <note><pitch><step>C</step><octave>3</octave></pitch><duration>8</duration></note>
    </measure></part></score-partwise>''',encoding='utf-8')
    x=xml_measures(p);assert x[0]['signature']==((0.,48,2.),(0.,60,1.),(0.,64,1.)) and x[0]['extent']==2.
    d=[dict(mc=1,number=1,signature=x[0]['signature'],extent=2.)]
    assert compare(x,d)[0]['exact_events']
    assert not compare(x+x,d)[0]['exact_events']
