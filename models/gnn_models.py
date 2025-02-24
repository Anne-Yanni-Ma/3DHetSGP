from utils import MLP
import torch
import torch.nn as nn


class GraphEncoder(nn.Module):
    def __init__(self, ndim=512, nlayer=5):
    #def __init__(self, ndim=512, nlayer=5):
        super(GraphEncoder, self).__init__()
        self.nlayer = nlayer
        self.ndim = ndim
        self.sgconv = nn.ModuleList([SceneGraphConv(ndim=self.ndim) for i in range(self.nlayer)])
        self.LN = nn.LayerNorm(self.ndim)

    def forward(self, G):
        s_idx, o_idx = G.edge_index[0, :].contiguous(), G.edge_index[1, :].contiguous()
        for i in range(self.nlayer):
            G = self.sgconv[i](G, s_idx, o_idx)
        G.h_outputs = torch.cat(G.h_outputs, dim=0)
        G.h_edge_outputs = torch.cat(G.h_edge_outputs, dim=0)
        G.h = self.LN(G.h_outputs.sum(dim=0))
        G.h_edge = self.LN(G.h_edge_outputs.sum(dim=0))
        return G.h, G.h_edge


class SceneGraphConv(nn.Module):
    def __init__(self, ndim=512):
        super(SceneGraphConv, self).__init__()
        self.ndim = ndim
        self.phis = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.phio = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.phip = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.LN = nn.LayerNorm(512)################ 512-->256
        self.node_GRU = nn.GRUCell(self.ndim, self.ndim)
        self.edge_GRU = nn.GRUCell(self.ndim, self.ndim)

    def forward(self, G, s_idx, o_idx):
        insnum = G.h.shape[0]
        Hs, Ho, Hp = G.h[s_idx], G.h[o_idx], G.h_edge
        Mn, Mp = self.message(Hs, Ho, Hp, s_idx, o_idx, insnum)

        G.h = self.node_GRU(Mn, G.h)
        G.h_edge = self.edge_GRU(Mp, G.h_edge)

        G.h_outputs.append(G.h.view(1, -1, self.ndim))
        G.h_edge_outputs.append(G.h_edge.view(1, -1, self.ndim))
        return G

    def message(self, Hs, Ho, Hp, s_idx, o_idx, insnum):
        Ms = self.LN(self.phio(Ho) + self.phip(Hp))
        Mo = self.LN(self.phis(Hs) + self.phip(Hp))
        Mp = self.LN(self.phis(Hs) + self.phio(Ho))
        Mn = self.average_pooling(Ms, Mo, s_idx, o_idx, insnum)
        return Mn, Mp

    def average_pooling(self, Ms, Mo, s_idx, o_idx, insnum):
        Mpooling = torch.zeros(insnum, self.ndim).cuda()
        Mpooling = Mpooling.scatter_add(0, s_idx.view(-1, 1).expand_as(Ms), Ms)
        Mpooling = Mpooling.scatter_add(0, o_idx.view(-1, 1).expand_as(Mo), Mo)
        obj_counts = torch.zeros(insnum).cuda()
        #ones = torch.ones(1000).cuda()
        ones = torch.ones(self.ndim).cuda()
        obj_counts = obj_counts.scatter_add(0, s_idx, ones)  #RuntimeError: invalid argument 3: Index tensor must not have larger size than input tensor, but got index [1722] input [512]
        obj_counts = obj_counts.scatter_add(0, o_idx, ones)
        obj_counts = obj_counts.clamp(min=1)
        Mpooling = Mpooling / obj_counts.view(-1, 1)
        return Mpooling


class GraphEncoderKnowledgeFusion(nn.Module):
    def __init__(self, ndim=512, nlayer=5):
        super(GraphEncoderKnowledgeFusion, self).__init__()
        self.nlayer = nlayer
        self.ndim = ndim
        self.sgconv = nn.ModuleList([SceneGraphConv(ndim=self.ndim) for i in range(self.nlayer)])
        self.node_transfomer = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.edge_transfomer = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.knode_transfomer = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.kedge_transfomer = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.LN = nn.LayerNorm(self.ndim)

    def forward(self, G, Nknowledge, Eknowledge):
        s_idx, o_idx = G.edge_index[0, :].contiguous(), G.edge_index[1, :].contiguous()
        G = self.knowledge_fusion(G, Nknowledge, Eknowledge)
        for i in range(self.nlayer):
            G = self.sgconv[i](G, s_idx, o_idx)
        G.h_outputs = torch.cat(G.h_outputs, dim=0)
        G.h_edge_outputs = torch.cat(G.h_edge_outputs, dim=0)
        G.h = self.LN(G.h_outputs.sum(dim=0))
        G.h_edge = self.LN(G.h_edge_outputs.sum(dim=0))
        return G.h, G.h_edge


    def knowledge_fusion(self, G, Nknowledge, Eknowledge):
        G.h = self.LN(G.h + Nknowledge)
        G.h = self.LN(G.h + self.node_transfomer(G.h))
        G.h_edge = self.LN(G.h_edge + Eknowledge)
        G.h_edge = self.LN(G.h_edge + self.edge_transfomer(G.h_edge))
        return G