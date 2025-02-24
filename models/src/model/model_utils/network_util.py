#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:46:24 2020

@author: sc
"""

import torch
from torch_geometric.nn.conv import MessagePassing
from src.model.model_utils.networks_base import mySequential
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

def MLP(channels: list, do_bn=False, on_last=False, drop_out=None):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    offset = 0 if on_last else 1
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-offset):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
            
            if drop_out is not None:
                layers.append(torch.nn.Dropout(drop_out))
    return mySequential(*layers)


def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
   layers = []
   for i in range(len(dim_list) - 1):
     dim_in, dim_out = dim_list[i], dim_list[i + 1]
     layers.append(torch.nn.Linear(dim_in, dim_out))
     final_layer = (i == len(dim_list) - 2)
     if not final_layer or on_last:
       if do_bn:
         layers.append(torch.nn.BatchNorm1d(dim_out))
       if activation == 'relu':
         layers.append(torch.nn.ReLU())
       elif activation == 'leakyrelu':
         layers.append(torch.nn.LeakyReLU())
     if dropout > 0:
       layers.append(torch.nn.Dropout(p=dropout))
   return torch.nn.Sequential(*layers)


class Gen_Index(MessagePassing):
    """ A sequence of scene graph convolution layers  """
    def __init__(self,aggr,  flow="target_to_source"):#node_dim
        super().__init__(aggr=aggr, flow=flow) #, node_dim=node_dim

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] =[None,None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype ==torch.long
            assert edge_index.dim()==2
            assert edge_index.size(0)==2
            if size is not None:
                the_size[0]=size[0]
                the_size[1]=size[1]
            return the_size
        elif isinstance(edge_index, SparseTensor):
            if self.flow =='target_to_source':
                raise ValueError("whatever, I dont care.")
            the_size[0]= edge_index.sparse_size(1)
            the_size[1]= edge_index.sparse_size(0)
            return the_size

        
    def forward(self, x, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(edges_indices,size, 'edge_index', {"x":x},)
        x_i, x_j = coll_dict['x_i'], coll_dict['x_j'] #self.inspector.distribute('message', coll_dict)
        #msg_kwargs = self.__distribute__(coll_dict, 'message' )
        x_i, x_j = self.message(x_i, x_j)
        return x_i, x_j
    def message(self, x_i, x_j):
        return x_i,x_j

class Aggre_Index(MessagePassing):
    def __init__(self,aggr='add', node_dim=-2,flow="source_to_target"):
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] =[None,None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype ==torch.long
            assert edge_index.dim()==2
            assert edge_index.size(0)==2
            if size is not None:
                the_size[0]=size[0]
                the_size[1]=size[1]
            return the_size
        elif isinstance(edge_index, SparseTensor):
            if self.flow =='target_to_source':
                raise ValueError("whatever, I dont care.")
            the_size[0]= edge_index.sparse_size(1)
            the_size[1]= edge_index.sparse_size(0)
            return the_size

    def forward(self, x, edge_index, dim_size,dim):
        size = self.__check_input__(edge_index, None)
        coll_dict = self.__collect__(edge_index, size,'edge_index',{})
        coll_dict['dim_size'] = dim_size
        x = self.aggregate(x, edge_index[0], dim_size =dim_size,dim=dim)
        return x

    def aggregate(self, inputs, index, ptr=None, dim_size=None,dim=None):

        if ptr is not None:
            for _ in range(dim):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=dim, dim_size=dim_size,
                           reduce=self.aggr)


if __name__ == '__main__':
    flow = 'source_to_target'
    # flow = 'target_to_source'
    g = Gen_Index(flow = flow)
    
    edge_index = torch.LongTensor([[0,1,2],
                                  [2,1,0]])
    x = torch.zeros([3,5])
    x[0,:] = 0
    x[1,:] = 1
    x[2,:] = 2
    x_i,x_j = g(x,edge_index)
    print('x_i',x_i)
    print('x_j',x_j)
    
    tmp = torch.zeros_like(x_i)
    tmp = torch.zeros([5,2])
    edge_index = torch.LongTensor([[0,1,2,1,0],
                                  [2,1,1,1,1]])
    for i in range(5):
        tmp[i] = -i
    aggr = Aggre_Index(flow=flow,aggr='max')
    xx = aggr(tmp, edge_index,dim_size=x.shape[0])
    print(x)
    print(xx)