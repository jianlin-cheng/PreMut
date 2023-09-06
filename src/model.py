import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, TransformerConv, aggr
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU

from dataset import *

from torch.nn import Linear, Parameter

from torch_geometric.nn import MessagePassing,GATConv, GATv2Conv, GraphUNet
from torch_geometric.nn import global_mean_pool, MeanAggregation
from egnn_clean import EGNN




class egnn_ablation_atom_types_only(torch.nn.Module):
    def __init__(self,in_channels: List,hidden_channels,num_classes,num_hidden_layers = NUM_HIDDEN_LAYERS) -> None:
        super().__init__()
        self.model = EGNN(in_node_nf=in_channels[1],hidden_nf=hidden_channels,out_node_nf=3,in_edge_nf=1,attention=True,n_layers=num_hidden_layers)

    def edge_index_to_edges(self,edge_index):
        edges = []
        edges.append(edge_index[0])
        edges.append(edge_index[1])
        return edges
    def forward(self,node_feats: tuple,edge_index, edge_attr,batch):
        # input = torch.cat((node_feats[1],node_feats[2]),dim=1)
        input = node_feats[1]
        edges = self.edge_index_to_edges(edge_index=edge_index)
        h,x = self.model(h=input,x=node_feats[0],edges=edges,edge_attr=edge_attr)
        return h




















