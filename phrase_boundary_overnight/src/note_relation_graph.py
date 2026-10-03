"""Label-free note graph: retain note identity before beat aggregation."""
from collections import Counter
import numpy as np
import torch
from torch import nn
from .resolve_tie_audit import predecessors
from .score_context_study import make_model


def merged_events(events,ties):
    links={}
    for i,t in enumerate(ties):
        if t in (0,-1):
            _,ids=predecessors(events,ties,i)
            if len(ids)==1:links[i]=ids[0]
    count=Counter(links.values());links={i:j for i,j in links.items() if count[j]==1}
    groups={}
    for i,e in enumerate(events):
        if e[1]<=0:continue
        root=i
        while root in links:root=links[root]
        groups.setdefault(root,[]).append(i)
    rows=[]
    for root,ids in groups.items():
        q,d,p,s,v=events[root];end=max(events[i][0]+events[i][1] for i in ids)
        rows.append((q,end-q,p,s,v,float(ties[root] in (0,-1))))
    return sorted(rows),len(links)


def build_graph(events,ties,beat_number):
    n=len(beat_number);nodes,merged=merged_events(events,ties)
    nodes=[e for e in nodes if e[0]<n and e[0]+e[1]>0]
    if not nodes:raise ValueError('No positive-duration score events')
    a=np.asarray(nodes,float);q=a[:,0];end=q+a[:,1];pitch=a[:,2];N=len(a)
    x=np.zeros((N,21),np.float32)
    x[:,0]=(pitch-60)/24;x[:,1]=np.log1p(a[:,1])/4
    x[:,2]=np.sin(2*np.pi*q);x[:,3]=np.cos(2*np.pi*q)
    meter=max(float(np.nanmax(beat_number))+1,1)
    bi=np.clip(np.floor(q).astype(int),0,n-1);phase=(beat_number[bi]+q-np.floor(q))/meter
    x[:,4]=np.sin(2*np.pi*phase);x[:,5]=np.cos(2*np.pi*phase)
    x[:,6]=(a[:,3]==a[:,3].min());x[:,7]=(a[:,3]==a[:,3].max());x[:,8]=a[:,5]
    x[np.arange(N),9+pitch.astype(int)%12]=1
    edges=set();groups={}
    for i,t in enumerate(q):groups.setdefault(round(float(t),8),[]).append(i)
    times=sorted(groups)
    def connect(left,right):
        for i in left:
            for j in right:
                if i!=j:edges.add((i,j));edges.add((j,i))
    for ix,t in enumerate(times):
        group=groups[t];connect(group,group)
        # Adjacent onset groups bridge genuine gaps; unlike cadence paper's rests.
        if ix+1<len(times):connect(group,groups[times[ix+1]])
    for i in range(N):
        # Strictly later onsets while this note is held, plus exact offset contacts.
        targets=np.flatnonzero((q>q[i]+1e-7)&(q<=end[i]+1e-7))
        connect([i],targets)
    ei=np.asarray(sorted(edges),dtype=np.int64).T if edges else np.empty((2,0),np.int64)
    src,dst=ei
    dp=(pitch[src]-pitch[dst])/12;dt=q[src]-q[dst];dd=np.log1p(a[src,1])-np.log1p(a[dst,1])
    ef=np.column_stack([np.clip(dp,-4,4),np.sign(dt)*np.log1p(abs(dt))/4,dd/4,(abs(dt)<1e-7).astype(float)]).astype(np.float32)
    pool=[]
    for i,(t,d,*_) in enumerate(nodes):
        if 0<=t<n:pool.append((int(np.floor(t)),i,0,1.))
        for b in range(max(0,int(np.floor(t))),min(n,int(np.ceil(t+d)))):
            w=max(0.,min(b+1,t+d)-max(b,t))
            if w>0:pool.append((b,i,1,w))
    pool=np.asarray(pool,float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite node inputs')
    return dict(node_features=x,edge_index=ei,edge_features=ef,pool_beat=pool[:,0].astype(np.int64),pool_node=pool[:,1].astype(np.int64),pool_kind=pool[:,2].astype(np.int64),pool_weight=pool[:,3].astype(np.float32),n_beats=np.array(n,np.int64),note_events=a,merged_ties=np.array(merged,np.int64))


def tensors(graph,device):
    return {k:torch.as_tensor(v,device=device) for k,v in graph.items() if k not in ('note_events','merged_ties')}


class RelationLayer(nn.Module):
    def __init__(self):
        super().__init__();self.own=nn.Linear(24,24);self.neighbor=nn.Linear(24,24);self.relative=nn.Linear(4,24,bias=False);self.norm=nn.LayerNorm(24)
    def forward(self,h,g,use_relations):
        if use_relations:
            src,dst=g['edge_index'];msg=self.neighbor(h[src])+self.relative(g['edge_features'])
            agg=torch.zeros_like(h).index_add(0,dst,msg)
            degree=torch.zeros((len(h),1),device=h.device,dtype=h.dtype).index_add(0,dst,torch.ones((len(dst),1),device=h.device,dtype=h.dtype))
            agg=agg/degree.clamp_min(1)
        else:
            # Same layers/weights; neighbor is replaced by self and relative inputs disabled.
            agg=self.neighbor(h)+self.relative(torch.zeros((len(h),4),device=h.device,dtype=h.dtype))
        return self.norm(h+torch.nn.functional.gelu(self.own(h)+agg))


class NoteRelationBoundary(nn.Module):
    def __init__(self,use_relations,seed):
        super().__init__();self.core=make_model('P',seed);self.use_relations=use_relations
        with torch.random.fork_rng():
            torch.manual_seed(seed+7000)
            self.node_input=nn.Linear(21,24);self.layers=nn.ModuleList([RelationLayer(),RelationLayer()])
            self.beat_output=nn.Sequential(nn.Linear(48,24),nn.LayerNorm(24))
    def encode_nodes(self,g):
        h=torch.nn.functional.gelu(self.node_input(g['node_features']))
        for layer in self.layers:h=layer(h,g,self.use_relations)
        return h
    def encode_beats(self,g):
        h=self.encode_nodes(g);n=int(g['n_beats']);indices=g['pool_beat']*2+g['pool_kind'];weight=g['pool_weight'][:,None]
        sums=torch.zeros((n*2,24),device=h.device,dtype=h.dtype).index_add(0,indices,h[g['pool_node']]*weight)
        den=torch.zeros((n*2,1),device=h.device,dtype=h.dtype).index_add(0,indices,weight)
        return self.beat_output((sums/den.clamp_min(1e-8)).reshape(n,48))
    def forward_embedded(self,x,features):
        return self.core(torch.cat([x[...,:34],features.expand(x.shape[0],-1,-1)],-1))
    def forward(self,x,g):return self.forward_embedded(x,self.encode_beats(g)[None])
