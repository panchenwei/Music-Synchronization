"""Resolve only unambiguous notated ties; retain unresolved attacks conservatively."""
import numpy as np
import pandas as pd
from .data import choose_measure_path,to_float
from .slice_energy_features import voiced_events
from .score_piano_roll import piano_roll


def tied_rows(piece,n):
    notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t')
    assert 'tied' in notes
    path,_,_=choose_measure_path(measures,n);assert path
    folded={int(r.mc):to_float(r.quarterbeats) for r in measures.itertuples()};events=[];ties=[]
    for occurrence in path:
        mc=int(occurrence['mc'])
        for r in notes[notes.mc==mc].itertuples():
            o=occurrence['start_qb']+to_float(r.quarterbeats)-folded[mc];d=max(to_float(r.duration_qb),0)
            if np.isfinite(o) and np.isfinite(d):
                events.append((o,d,int(r.midi),int(r.staff),int(r.voice)))
                assert pd.isna(r.tied) or r.tied in (-1,0,1)
                ties.append(None if pd.isna(r.tied) else int(r.tied))
    assert events==voiced_events(piece,n)
    return events,ties


def attack_flags(events,ties):
    if len(events)!=len(ties):raise ValueError('flags length')
    attack=np.array([e[1]>0 for e in events],dtype=bool)
    by_key={}
    for i,e in enumerate(events):by_key.setdefault(tuple(e[2:5]),[]).append(i)
    counts=dict(connected=0,orphan=0,ambiguous=0)
    for i,(e,tie) in enumerate(zip(events,ties)):
        if tie not in (None,-1,0,1):raise ValueError('invalid tie')
        if e[1]<=0 or tie not in (-1,0):continue
        # Same pitch AND notated staff/voice, contiguous times and outgoing tie.
        candidates=[j for j in by_key[tuple(e[2:5])] if j!=i and ties[j] in (0,1) and events[j][1]>0 and events[j][0]<e[0] and abs(events[j][0]+events[j][1]-e[0])<1e-8]
        if len(candidates)==1:attack[i]=False;counts['connected']+=1
        elif len(candidates)==0:counts['orphan']+=1
        else:counts['ambiguous']+=1
    return attack,counts


def tied_piano_roll(events,ties,n):
    result=piano_roll(events,n)
    attack,counts=attack_flags(events,ties)
    result[:,1]=0
    for e,yes in zip(events,attack):
        o,d,p,*_=e
        if yes and 0<=o<n:
            tick=min(int(np.floor(o*4+1e-9)),n*4-1)
            result[tick//4,1,int(p),tick%4]=1
    return result,counts
