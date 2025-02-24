import torch
import torch.nn as nn
import torch.nn.functional as F

import pathmagic  # noqa
from models.pred_classification.pointnet import PointNetEncoderwoBN
from models.pred_classification.utils import FocalLoss, MLP
from pointnet2_ops.pointnet2_utils import FurthestPointSampling
from models.graph import SceneGraph

support_label = [1,14,15,16,17,18,19,20,21,22,23,24,25,26]  # len: 14
proximity_label = [2,3,4,5,6,7] #
comparative_label = [8,9,10,11,12,13] # 6个
cp_label = [2,3,4,5,6,7,8,9,10,11,12,13]
useBCELoss = False
useFocalLoss = True

def fps_sampling(xyz, npoints):
    # xyz: B, N, 3
    fps = FurthestPointSampling()
    idx = fps.apply(xyz, npoints).long()      # B, N
    return idx.long()

def Gate_threshold(W, threshold):
    Gate_W = W.clone()
    for i in range(Gate_W.shape[0]):
        prob = Gate_W[i]
        if prob > threshold:
            prob = 1
        else:
            prob = 0
        Gate_W[i] = prob
    return Gate_W


def Gate(W, alp, bel):
    Gate_W = W.clone()
    for i in range(Gate_W.shape[0]):
        prob = Gate_W[i]
        if prob <= bel:
            prob = 0
        elif prob >= 1 / alp + bel:
            prob = 1
        else:
            prob =  alp * prob - alp * bel  # alp*prob - alp*bel

        Gate_W[i] = prob
    return Gate_W

def gen_type_weight(type_output4):
    none_weight, support_weight, p_weight, c_weight = type_output4[:,0],type_output4[:,1],type_output4[:,2],type_output4[:,3]
    #nps, npp, npc = support_weight.cpu().numpy(), p_weight.cpu().numpy(),c_weight.cpu().numpy()
    link_w = torch.ones(none_weight.shape[0]).cuda()
    none_w = Gate_threshold(none_weight, 0.8)
    link_w = link_w-none_w
    support_w = Gate_threshold(support_weight,0.15)
    p_w = Gate_threshold(p_weight,0.1)
    c_w = Gate_threshold(c_weight,0.1)

    Gated_types_weight = [link_w, support_w, p_w, c_w]
    return Gated_types_weight

def gen_type_weight_soft(type_output4):
    none_weight, support_weight, p_weight, c_weight = type_output4[:,0],type_output4[:,1],type_output4[:,2],type_output4[:,3]
    #nps, npp, npc = support_weight.cpu().numpy(), p_weight.cpu().numpy(),c_weight.cpu().numpy()
    link_weight = torch.ones(none_weight.shape[0]).cuda()
    link_weight = link_weight-none_weight
    link_w = Gate(link_weight, 2.2, 0.025)
    support_w = Gate(support_weight,2.2,0.025)
    p_w = Gate(p_weight,2.2,0.025)
    c_w = Gate(c_weight,2.2,0.025)

    Gated_types_weight = [link_w, support_w, p_w, c_w]
    return Gated_types_weight

class get_model_pred(nn.Module):
    def __init__(self):
        super(get_model_pred, self).__init__()
        self.pred_pointnet_lv1 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=512)
        self.pred_pointnet_lv2 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=512)
        self.pred_pointnet_lv3 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=512)

        self.pred_mlp = MLP(mlp=[512*3, 1024, 512, 256])
        self.pred_mlp2 = MLP(mlp=[269, 256, 256])
        self.pred_classifer = MLP(mlp=[256, 128, 27])
        self.edge_classifer = MLP(mlp=[256, 128, 2])
        self.type_classifer = MLP(mlp=[512, 128, 4])
        self.pred_mlp_edge = MLP(mlp=[512,512, 512])
        self.gnn = GraphEncoder(ndim=512, nlayer=3)

        self.relu = nn.ReLU()


    def forward(self, obj_codes, pred_codes):
        insnum = obj_codes.shape[0] # Nn
        edge_features = pred_codes
        edge_codes = self.pred_mlp_edge(edge_features) #　512

        edge_embed = edge_codes
        # Classification
        type_pred = self.type_classifer(edge_embed)  #4
        type_output = F.softmax(type_pred, dim=1)
        none_w, support_w, p_w, c_w = type_output[:,0],type_output[:,1],type_output[:,2],type_output[:,3]
        types_w = [ 1-none_w,  support_w ,p_w, c_w]

        multiW = gen_type_weight_soft(type_output)

        return  types_w, multiW, type_output, edge_embed

    def prepare_edges(self, insnum):
        edge_index = torch.zeros(2, insnum*insnum-insnum).long().cuda()
        idx = 0
        for i in range(insnum):
            for j in range(insnum):
                if i != j:
                    edge_index[0, idx] = i
                    edge_index[1, idx] = j
                    idx += 1
        return edge_index




class TypeLoss(nn.Module):
    def __init__(self,gamma, pred_w):
        super(TypeLoss, self).__init__()
        self.a = 0.5
        self.b = 0.5
        self.gamma = gamma
        self.focal_loss_type = FocalLoss(class_num=4, alpha=pred_w,  gamma=gamma, size_average=True)

    def forward(self, type_output, obj_gt, rel_gt):
        type_gt = self.prepare_typegt(obj_gt, rel_gt)
        type_loss =self.focal_loss_type(type_output, type_gt)
        edge_loss = type_loss
        return edge_loss

    def prepare_typegt(self, obj_gt, rel_gt):
        insnum = obj_gt.shape[0]
        type_gt = torch.zeros((insnum * insnum - insnum, 4)).long().cuda()
        # none, support, proximity, comparative

        for i in range(rel_gt.shape[0]):
            idx_i = rel_gt[i, 0]
            idx_j = rel_gt[i, 1]
            predgt =  rel_gt[i, 2]
            if (predgt in support_label):
                type_id = 1
            elif (predgt in proximity_label):
                type_id = 2
            elif(predgt in comparative_label):
                type_id = 3
            else: type_id = 0
            if idx_i < idx_j:
                type_gt[int(idx_i * (insnum - 1) + idx_j - 1), type_id] = 1
            elif idx_i > idx_j:
                type_gt[int(idx_i * (insnum - 1) + idx_j), type_id] = 1

        for i in range(insnum * insnum - insnum):
            if torch.sum(type_gt[i, :]) == 0:
                type_gt[i, 0] = 1

        return type_gt



class GraphEncoder(nn.Module):
    def __init__(self, ndim, nlayer=5):
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
