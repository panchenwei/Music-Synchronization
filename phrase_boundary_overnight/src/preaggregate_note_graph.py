"""One change to frozen graph model: nonlinear pair messages before averaging."""
import torch
from torch.nn import functional as F
from .note_relation_graph import NoteRelationBoundary


def preaggregate(layer,h,g):
    src,dst=g['edge_index']
    msg=F.gelu(layer.neighbor(h[src])+layer.relative(g['edge_features']))
    agg=torch.zeros_like(h).index_add(0,dst,msg)
    den=torch.zeros((len(h),1),device=h.device,dtype=h.dtype).index_add(0,dst,torch.ones((len(dst),1),device=h.device,dtype=h.dtype))
    return layer.norm(h+F.gelu(layer.own(h)+agg/den.clamp_min(1)))


class PreAggregateBoundary(NoteRelationBoundary):
    def __init__(self,seed):
        super().__init__(True,seed)
    def encode_nodes(self,g):
        h=F.gelu(self.node_input(g['node_features']))
        for layer in self.layers:h=preaggregate(layer,h,g)
        return h
