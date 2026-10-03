"""Reproducible, prediction-blind annotated event excerpts from development works."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .score_context_study import ROOT,sha,write
from .slice_energy_study import DCML
from .data import discover_dcml_pieces
from .score_local_coordinates import positioned_rows
from .three_round_round2 import split_ids

CASES=[('A','chopin_op24_no1',18,28,'End then start'),
       ('B','chopin_op24_no1',66,77,'End, cadence, then start'),
       ('C','chopin_op24_no2',258,270,'Interlocking end and start')]


def main():
    out=ROOT/'reports/phrase_semantics_review/cases';out.mkdir(parents=True,exist_ok=True)
    pieces=discover_dcml_pieces(DCML);dev=set(split_ids(0)['validation']+split_ids(1)['validation'])
    hashes={};summaries=[];all_notes=[];all_annotations=[]
    fig,axes=plt.subplots(3,1,figsize=(13,10),constrained_layout=True)
    for ax,(cid,pid,a,b,title) in zip(axes,CASES):
        assert pid in dev;piece=pieces[pid];cache=ROOT/'artifacts/coordinate_repair_preview'/f'{pid}.npz'
        with np.load(cache,allow_pickle=False) as z:n=len(z['labels'])
        notes=pd.read_csv(piece.notes_path,sep='\t');measures=pd.read_csv(piece.measures_path,sep='\t');h=pd.read_csv(piece.harmony_path,sep='\t')
        nr,_,_=positioned_rows(notes,measures,n);hr,_,_=positioned_rows(h,measures,n);note_rows=[];ann=[]
        for q,i,o,r in nr:
            d=float(r.duration_qb)
            if q<=b and q+d>=a:
                row=dict(case=cid,piece_id=pid,quarterbeat=q,duration_qb=d,midi=int(r.midi),staff=int(r.staff),voice=int(r.voice),tied=None if pd.isna(r.tied) else int(r.tied),name=str(r['name']),octave=int(r.octave),mc=int(o['mc']),mn=str(r.mn),visit=int(o['visit']),source_row=int(i))
                note_rows.append(row);color='#2876b9' if int(r.staff)==1 else '#b07130'
                ax.plot([max(q,a),min(q+d,b)],[r.midi]*2,color=color,linewidth=2.8,alpha=.8)
                if a<=q<=b:ax.scatter([q],[r.midi],s=12,c=color)
        for q,i,o,r in hr:
            if not a<=q<=b:continue
            marker='' if pd.isna(r.get('phraseend')) else str(r.phraseend)
            cadence='' if pd.isna(r.get('cadence')) else str(r.cadence)
            if not marker and not cadence:continue
            row=dict(case=cid,piece_id=pid,quarterbeat=q,marker=marker,cadence=cadence,label=str(r.get('label','')),mc=int(o['mc']),mn=str(r.get('mn','')),visit=int(o['visit']),source_row=int(i));ann.append(row)
            text=('END + START' if '}' in marker and '{' in marker else 'START' if '{' in marker else 'END' if '}' in marker else '')+(' / '+cadence if cadence else '')
            ax.axvline(q,color='#c02673' if '{' in marker else '#555555',linestyle='--',linewidth=1.1)
            ax.text(q,.98,text,transform=ax.get_xaxis_transform(),rotation=90,va='top',ha='right',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
        # Reconstruct every bar line from traversal, including bars without a note attack.
        from .data import choose_measure_path
        path,_,_=choose_measure_path(measures,n)
        for o in path:
            q=o['start_qb']
            if a<=q<=b:ax.axvline(q,color='#bbbbbb',linewidth=.6,zorder=0)
        ax.set(xlim=(a,b),ylabel='MIDI pitch',xlabel='Unfolded quarterbeat (zero-based)',title=f'{cid}. {pid}: {title}')
        ax.set_xticks(np.arange(a,b+1));ax.spines[['top','right']].set_visible(False)
        all_notes.extend(note_rows);all_annotations.extend(ann)
        summaries.append(dict(case=cid,piece_id=pid,range_qb=[a,b],annotations=ann,noteheads=len(note_rows)))
        for p in (piece.notes_path,piece.harmony_path,piece.measures_path,cache):hashes[str(p)]=sha(p)
    fig.savefig(out/'source_event_excerpts.png',dpi=160);fig.savefig(out/'source_event_excerpts.svg');plt.close(fig)
    pd.DataFrame(all_notes).to_csv(out/'note_contexts.csv',index=False);pd.DataFrame(all_annotations).to_csv(out/'annotation_contexts.csv',index=False)
    assert all(sha(p)==v for p,v in hashes.items())
    write(out/'manifest.json',dict(status='source_context_extracted',cases=summaries,source_hashes=hashes,script_sha256=sha(__file__),selection='fixed source-topology examples, no model scores',development_only=True,model_predictions_read=False,audio_listened=False,manual_expert_adjudication=False,source_labels_modified=False,note='Piano-roll view of source noteheads with explicit ties in CSV; dots are not necessarily new attacks. Not a rendered original staff score or validated melody extraction.'))
    print(pd.DataFrame(all_annotations).to_string(index=False))


if __name__=='__main__':main()
