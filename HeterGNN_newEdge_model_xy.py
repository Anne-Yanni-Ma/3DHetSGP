import torch
import torch.nn as nn
import torch.nn.functional as F
import pathmagic  # noqa
import numpy as np
from models.utils import FocalLoss, MLP
from models.graph import SceneGraph
from models.src.model.model_utils.network_PointNet import PointNetfeat
from models.src.utils import op_utils

support_label = [1,14,15,16,17,18,19,20,21,22,23,24,25,26]  # len: 14
suppport_map = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
proximity_label = [2,3,4,5,6,7] # 不除去inside  6个
proximity_map = [1,2,3,4,5,6]
comparative_label = [8,9,10,11,12,13] # 6个
comparative_map = [1,2,3,4,5,6]
cp_label = [2,3,4,5,6,7,8,9,10,11,12,13] #12个
NOWEIGHT = False
def re_edge(edge_f,map):
    redge_f = edge_f.clone()
    for i in range(edge_f.shape[0]):
        redge_f[i] = edge_f[int(map[i])]
    return redge_f

def gen_gt_typeslink(pc_mat_node, gt_rel):
    # Input: gt_rel(N'e, 3)
    # Outuput: link_w  support_w, cp_w, comp_w(Ne,1)
    # 用gt_rel生成(Ne, 1)的边类型标记　(若存在则标1
    insnum = pc_mat_node.shape[0]
    link_w = torch.zeros(insnum*insnum -insnum).cuda() # (Ne,)
    support_w = torch.zeros(insnum*insnum -insnum).cuda() # (Ne,)
    p_w = torch.zeros(insnum*insnum -insnum).cuda() # (Ne,)
    comp_w = torch.zeros(insnum*insnum -insnum).cuda() # (Ne,)

    for i in range(gt_rel.shape[0]):
        idx_i = gt_rel[i, 0]
        idx_j = gt_rel[i, 1]
        pred_gt = gt_rel[i, 2]

        if idx_i < idx_j:
            link_w[int(idx_i * (insnum - 1) + idx_j - 1)] = 1
        elif idx_i > idx_j:
            link_w[int(idx_i * (insnum - 1) + idx_j)] = 1
        if pred_gt in support_label:
            if idx_i < idx_j:
                support_w[int(idx_i * (insnum - 1) + idx_j - 1)] = 1
            elif idx_i > idx_j:
                support_w[int(idx_i * (insnum - 1) + idx_j)] = 1
        if pred_gt in proximity_label:
            if idx_i < idx_j:
                p_w[int(idx_i * (insnum - 1) + idx_j - 1)] = 1
            elif idx_i > idx_j:
                p_w[int(idx_i * (insnum - 1) + idx_j)] = 1
        if pred_gt in comparative_label:
            if idx_i < idx_j:
                comp_w[int(idx_i * (insnum - 1) + idx_j - 1)] = 1
            elif idx_i > idx_j:
                comp_w[int(idx_i * (insnum - 1) + idx_j)] = 1

    types_w = link_w, support_w, p_w, comp_w
    return types_w


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.assimilation = 2
        self.gen_HEmb = gen_heter_Emb()
        #self.gen_HEmb_no = gen_heter_Emb_no()
        #self.gen_HEmb_p = gen_heter_Emb_p()
        #self.gen_HEmb_c = gen_heter_Emb_c()

        if NOWEIGHT==False:
            self.gnn_l = TypeGraphEncoder(ndim=512, nlayer=1)
            self.gnn_s = TypeGraphEncoder(ndim=512, nlayer=10)
            self.gnn_p = TypeGraphEncoder(ndim=512, nlayer=10)
            #self.gnn_p = GraphEncoder(ndim=512, nlayer=5)
            self.gnn_comp = TypeGraphEncoder(ndim=512, nlayer=10)

        if NOWEIGHT == True:
            self.gnn_l = GraphEncoder(ndim=512, nlayer=1)
            self.gnn_s = GraphEncoder(ndim=512, nlayer=5)
            self.gnn_p = GraphEncoder(ndim=512, nlayer=5)
            self.gnn_comp = GraphEncoder(ndim=512, nlayer=5)

        self.s_mlp = MLP(mlp=[256, 512])
        self.p_mlp = MLP(mlp=[256, 512])
        self.c_mlp = MLP(mlp=[256, 512])
        #self.pred_mlp_l = MLP(mlp=[512*3,512,512])

        self.node_mlp = MLP(mlp=[512, 512, 512])
        self.edge_mlp = MLP(mlp=[512, 512, 512])
        self.node_classifer = MLP(mlp=[512, 256, 256, 160])
        #self.edge_classifer = MLP(mlp=[512, 256, 128, 27])

        self.edge_classifer_s = MLP(mlp=[512,256, 128, 15])
        self.edge_classifer_p = MLP(mlp=[512,256, 128, 7])  # proximity
        self.edge_classifer_comp = MLP(mlp=[512,256, 128, 7])

        self.conv1d = torch.nn.Conv1d(256, 256, 1)
        self.fc_s = nn.Linear(512, 256)
        self.fc_p = nn.Linear(512, 256)
        self.fc_c = nn.Linear(512, 256)
        self.rel_encoder = PointNetfeat(global_feat=True,batch_norm=True,point_size=11,input_transform=False,
                                        feature_transform=False, out_size=512)
        #self.IMP = IMP(dim_node=512, dim_edge=512, dim_atten=512)  # 256

    def Gate(self, W, alp, bel):
        np_W = W.cpu().numpy()
        for i in range(W.shape[0]):
            prob = W[i]
            if prob<=bel:
                prob=0
            elif prob>=1/alp+bel:
                prob = 1
            else:
                prob = alp*prob - alp*bel
            np_W[i]=prob
        return np_W

    def forward(self, obj_codes, pred_codes, types_w, pc_geom_info,edge_descriptor): #obj_codes(Nn,C_konw) pred_codes (Ne, C)
        insnum = obj_codes.shape[0]
        link_w, support_w, p_w, comp_w  = types_w

        multi_codes = self.gen_HEmb(insnum, pred_codes,pc_geom_info)
        link_codes,support_codes,proximity_codes,comp_codes = multi_codes

        insnum = obj_codes.shape[0] # Nn
        edge_index = self.prepare_edges(insnum) # (2, Nn*(Nn-1))
        remap = []
        edge_indices = edge_index


        obj_codes_s = obj_codes
        obj_codes_p = obj_codes
        obj_codes_c = obj_codes
        g_l = SceneGraph(x=obj_codes, edge_index=edge_index, edge_attr=link_codes, type="link", edge_weight=link_w , remap=remap)  #edge_weight(Ne,) link图
        g_support = SceneGraph(x=obj_codes_s, edge_index=edge_index, edge_attr=support_codes,type="support", edge_weight=support_w,remap=remap)  #edge_weight(Ne,)
        g_p = SceneGraph(x=obj_codes_p, edge_index=edge_index, edge_attr=proximity_codes, type="proximity", edge_weight=p_w,remap=remap)  #edge_weight(Ne,)
        g_comp = SceneGraph(x=obj_codes_c, edge_index=edge_index, edge_attr=comp_codes, type="comparative",edge_weight=comp_w, remap=remap)  #edge_weight(Ne,)

        # link的obj间传递有效信息
        self.type = "link"
        node_embed, edge_embed_l = self.gnn_l(g_l)
        node_output = self.node_classifer(node_embed) # ([7, 160])  # ([42, 27])
        node_output = F.softmax(node_output, dim=1)

        self.type = "support"
        node_embed_s, edge_embed_s = self.gnn_s(g_support)
        self.type = "proximity"
        node_embed_p, edge_embed_p = self.gnn_p(g_p)
        self.type = "comparative"
        node_embed_c, edge_embed_c = self.gnn_comp(g_comp)


        logit_s = self.edge_classifer_s(edge_embed_s)  # ([7, 160])  # ([42, 27])
        edge_output_s = F.softmax(logit_s, dim=1) #(Ne,7)

        logit_p = self.edge_classifer_p(edge_embed_p)  # (Ne,512) (Ne,7)
        edge_output_p = torch.sigmoid(logit_p)
        #edge_output_p = F.softmax(logit_p, dim=1) #(Ne,7)

        logit_c = self.edge_classifer_comp(edge_embed_c)  # (Ne,512) (Ne,7)
        edge_output_c = F.softmax(logit_c, dim=1) #(Ne,7)

        multi_edge_output = [edge_output_s, edge_output_p, edge_output_c]
        multi_logits = [logit_s, logit_p, logit_c]
        return node_output, multi_logits,multi_edge_output  # ([7, 160])  # ([42, 27])



    def Gate_only_true(self, W):
        np_W = W.cpu().numpy()
        for i in range(W.shape[0]):
            prob = W[i]
            if prob >= 0.99:
                prob = 1
            else:
                prob = 0
            np_W[i] = prob
        return np_W

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
        self.LN = nn.LayerNorm(512)
        self.LN_s = nn.LayerNorm(512)  ################ 512-->256
        self.LN_p = nn.LayerNorm(512)
        self.LN_c = nn.LayerNorm(512)
        self.node_GRU = nn.GRUCell(self.ndim, self.ndim)
        self.edge_GRU = nn.GRUCell(self.ndim, self.ndim)

    def forward(self, G, s_idx, o_idx):
        # 不乘weight
        insnum = G.h.shape[0]
        Hs, Ho, Hp = G.h[s_idx], G.h[o_idx], G.h_edge
        #Mn, Mp = self.message_ori(Hs, Ho, Hp, s_idx, o_idx, insnum)
        Mp = self.edge_message(Hs, Ho,  G.type)
        Mn = self.node_message(Hp, insnum)

        G.h = self.node_GRU(Mn, G.h)
        G.h_edge = self.edge_GRU(Mp, G.h_edge)

        G.h_outputs.append(G.h.view(1, -1, self.ndim))
        G.h_edge_outputs.append(G.h_edge.view(1, -1, self.ndim))
        return G


    def message_ori(self, Hs, Ho, Hp, s_idx, o_idx, insnum):
        Ms = self.LN(self.phio(Ho) + self.phip(Hp))
        Mo = self.LN(self.phis(Hs) + self.phip(Hp))
        Mp = self.LN(self.phis(Hs) + self.phio(Ho))
        Mn = self.average_pooling(Ms, Mo, s_idx, o_idx, insnum)
        return Mn, Mp

    def edge_message(self, Hs, Ho,  type):  # s->i  o->j  weight: r_ij
        # entity -> predicate.
        Mp = self.phis(Hs) + self.phio(Ho)
        Mp = self.LN(Mp)

        return Mp  # subj,obj传递给predicate的信息

    def node_message(self, Hp, insnum):  # s->i  o->j  weight: r_ij
        # predicate传递给subj的信息
        w_Hij = self.phip(Hp)       # wHij    (Ne,)*(Ne,C)
        # predicate -> subj
        wHij_matrix = w_Hij.view(insnum, insnum-1, Hp.shape[1]) #(9,8,512)
        wHij_sum = torch.sum(wHij_matrix,dim=1)#.squeeze(dim=1) 主动s->o的加和
        Ms = wHij_sum #+wHji_sum
        return Ms

    def message(self, Hs, Ho, Hp, s_idx, o_idx, insnum):
        Ms = self.LN(self.phio(Ho) + self.phip(Hp))
        Mo = self.LN(self.phis(Hs) + self.phip(Hp))
        Mp = self.LN(self.phis(Hs) + self.phio(Ho))
        Mn = self.average_pooling(Ms, Mo, s_idx, o_idx, insnum)
        return Mn, Mp

    def message_fei(self, Hs, Ho, Hp, s_idx, o_idx, insnum,type):
        Ms = self.phio(Ho) + self.phip(Hp)
        if (type == "link"): Ms = self.LN(Ms)
        if (type == "support"):  Ms = self.LN_s(Ms)
        if (type == "proximity"):  Ms = self.LN_p(Ms)
        if (type == "comparative"):  Ms = self.LN_c(Ms)
        Mo = self.phis(Hs) + self.phip(Hp)
        if (type == "link"): Mo = self.LN(Mo)
        if (type == "support"):  Mo = self.LN_s(Mo)
        if (type == "proximity"):  Mo = self.LN_p(Mo)
        if (type == "comparative"):  Mo = self.LN_c(Mo)
        Mp = self.LN(self.phis(Hs) + self.phio(Ho))
        if (type == "link"): Mp = self.LN(Mp)
        if (type == "support"):  Mp = self.LN_s(Mp)
        if (type == "proximity"):  Mp = self.LN_p(Mp)
        if (type == "comparative"):  Mp = self.LN_c(Mp)
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



class TypeGraphEncoder(nn.Module):
    def __init__(self, ndim, nlayer=3):
    #def __init__(self, ndim=512, nlayer=5):
        super(TypeGraphEncoder, self).__init__()
        self.nlayer = nlayer
        self.ndim = ndim
        self.sgconv = nn.ModuleList([SceneGraphConv_weight(ndim=self.ndim) for i in range(self.nlayer)])

        self.LN = nn.LayerNorm(self.ndim)

    '''
          o_idx 1 2 3 4 5 6 7 8 0 2 3 4 5 6 7 8 0 1 3 4 5 6 7 8 0 1 2 4 5 6 7 8 0 1 ......
    s_idx
    0 
    0
    0...
    '''
    def forward(self, G):
        s_idx, o_idx = G.edge_index[0, :].contiguous(), G.edge_index[1, :].contiguous() #
        edge_weight = G.edge_weight
        for i in range(self.nlayer):
            G = self.sgconv[i](G, s_idx, o_idx)
        G.h_outputs = torch.cat(G.h_outputs, dim=0)
        G.h_edge_outputs = torch.cat(G.h_edge_outputs, dim=0)
        G.h = self.LN(G.h_outputs.sum(dim=0))
        G.h_edge = self.LN(G.h_edge_outputs.sum(dim=0))
        return G.h, G.h_edge


class SceneGraphConv_weight(nn.Module):
    def __init__(self, ndim):
        super(SceneGraphConv_weight, self).__init__()

        self.ndim = ndim
        self.phis = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.phio = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.phip = MLP(mlp=[self.ndim, self.ndim, self.ndim])
        self.LN = nn.LayerNorm(512)
        self.LN_s = nn.LayerNorm(512)################ 512-->256
        self.LN_p = nn.LayerNorm(512)
        self.LN_c = nn.LayerNorm(512)
        self.node_GRU = nn.GRUCell(self.ndim, self.ndim)
        self.edge_GRU = nn.GRUCell(self.ndim, self.ndim)

    def forward(self, G, s_idx, o_idx):
        insnum = G.h.shape[0]
        edge_weight = G.edge_weight
        Hp = G.h_edge
        #re_edge_weight = re_edge(edge_weight,G.remap)
        #re_Hp = re_edge(Hp,G.remap)

        entity = G.h
        Hs, Ho = G.h[s_idx], G.h[o_idx]

        Mp = self.edge_message(Hs, Ho, edge_weight ,G.type)
        Mn = self.node_message(Hp,  insnum, edge_weight)
        #Mn = self.node_message_full(Hs, Ho, Hp, s_idx, o_idx, insnum, edge_weight)
        G.h = self.node_GRU(Mn, G.h)
        G.h_outputs.append(G.h.view(1, -1, self.ndim))

        G.h_edge = self.edge_GRU(Mp, G.h_edge)
        G.h_edge_outputs.append(G.h_edge.view(1, -1, self.ndim))
        return G

    def edge_message(self, Hs, Ho,edge_weight, type): # s->i  o->j  weight: r_ij
        # entity -> predicate.
        edge_weight = edge_weight.view(edge_weight.shape[0],1)
        Mp = self.phis(edge_weight * Hs) + self.phio(edge_weight * Ho)
        if (type == "link"): Mp = self.LN(Mp)
        if (type == "support"):  Mp = self.LN_s(Mp)
        if (type == "proximity"):  Mp = self.LN_p(Mp)
        if (type == "comparative"):  Mp = self.LN_c(Mp)
        return Mp        # subj,obj传递给predicate的信息

    def node_message(self, Hp, insnum, edge_weight):  # s->i  o->j  weight: r_ij
        edge_weight = edge_weight.view(edge_weight.shape[0],1)
        #re_edge_weight = re_edge_weight.view(re_edge_weight.shape[0],1)
        w_Hij = edge_weight * self.phip(Hp)       # wHij    (Ne,)*(Ne,C)
        #w_Hji = re_edge_weight * self.phip(re_Hp)
        # predicate -> subj
        wHij_matrix = w_Hij.view(insnum, insnum-1, Hp.shape[1]) #(9,8,512)
        wHij_sum = torch.sum(wHij_matrix,dim=1)#.squeeze(dim=1) 主动s->o的加和
        #wHji_matrix = w_Hji.view(insnum, insnum - 1, 512)  # (9,8,512)
        #wHji_sum = torch.sum(wHji_matrix, dim=1)  #(9,512)
        Ms = wHij_sum #+wHji_sum
        #Ms = torch.mean(wHij_sum+wHji_sum,dim=1)
        return Ms



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


class gen_heter_Emb(nn.Module):
    def __init__(self):
        super(gen_heter_Emb, self).__init__()

        self.pred_mlp_s = MLP(mlp=[512, 512, 512])
        self.pred_mlp_p = MLP(mlp=[512 + 128, 512, 512])
        self.pred_mlp_c = MLP(mlp=[512 + 64, 512, 512])
        self.pred_mlp_l = MLP(mlp=[512 * 3, 512, 512])

        self.pos_fc = FullyConnectedNet(5, 64, 128)
        self.geom_fc = FullyConnectedNet(4, 32, 64)
        self.geom_fc2 = FullyConnectedNet(64, 128, 256)

    def forward(self, insnum, pred_codes, pc_geom_info):  # (Nn,1024,3)
        # 输入预训好的边特征　
        # 输出不同类型的关系嵌入：　
        #    Support: visual_i - visual-j
        #    Proximity: bbox offset \\ or 质心距离、方位特征向量
        #    Comparative: 物体size\symmetry属性oneot编码向量的concat　先暂用visual_i - visual-j　+ h V

        pred_idx = torch.LongTensor([[i, j] for i in range(insnum) for j in range(insnum) if i != j])
        diff_codes = pred_codes

        # Geometric features
        edges = pred_idx
        bboxes, lwhV, centroid = pc_geom_info
        min_bboxes = bboxes[:, 0]
        max_bboxes = bboxes[:, 1]
        centroid_i = centroid[edges[:, 0],:2]
        centroid_j = centroid[edges[:, 1],:2]
        dx = centroid_i[:, 0] - centroid_j[:, 0]  ##质心  #(No,1)
        dy = centroid_i[:, 1] - centroid_j[:, 1]  ##(No,1)
        #dz = centroid_i[:, 2] - centroid_j[:, 2]  # (No,1)
        bboxes_offset_xy = torch.stack([dx, dy], dim=1).to(torch.float32)  # (Ne,3)
        distance = torch.zeros([pred_codes.shape[0], 1]).cuda()  # #(Ne,1)
        direction = torch.zeros([pred_codes.shape[0], 2]).cuda()  # (Ne,3)

        for i in range(distance.shape[0]):
            distance[i] = torch.norm(centroid_i[i] - centroid_j[i], p=2)  # 需要高斯挤压到0~1 #(Ne,)
            direction[i] = (centroid_i[i] - centroid_j[i]) / torch.norm(centroid_i[i] - centroid_j[i], p=2)  # (Ne,2)
            #print("direction:", direction[i])
            #print("dx, dy, dz: ", bboxes_offset[i])
        pos_features = torch.cat((bboxes_offset_xy, distance, direction), dim=1)  # (Ne,2+1+2)

        l, w, h, V = lwhV[:, 0], lwhV[:, 1], lwhV[:, 2], lwhV[:, 3]
        d_l = torch.log(l[edges[:, 0]] / l[edges[:, 1]])
        d_w = torch.log(w[edges[:, 0]] / w[edges[:, 1]])
        d_h = torch.log(h[edges[:, 0]] / h[edges[:, 1]])
        d_V = torch.log(V[edges[:, 0]] / V[edges[:, 1]])  # .reshape(-1, 1)
        geom_features = torch.stack([d_l, d_w, d_h, d_V], dim=1).to(torch.float32)  # (Nn,C_g)
        # geom_codes = self.geom_fc2(geom_codes) #256

        support_codes = self.pred_mlp_s(diff_codes)  # (Ne,256)

        pos_codes = self.pos_fc(pos_features)  # MLP 或FCL (Ne, 7)--> (Ne, 128)
        proximity_features = torch.cat((diff_codes, pos_codes), dim=1)  # 512+128  640
        proximity_codes = self.pred_mlp_p(proximity_features)  # (Ne, 640) -> (Ne,512)

        geom_codes = self.geom_fc(geom_features)  # (Ne, 3)--> (Ne, 64)
        comp_features = torch.cat((diff_codes, geom_codes), dim=1)  # (Ne,512+64) 578
        comp_codes = self.pred_mlp_c(comp_features)  # (Ne, 578) ->(Ne,512)

        link_codes = torch.cat((support_codes, proximity_codes, comp_codes), dim=1)
        link_codes = self.pred_mlp_l(link_codes)  # (Ne,512)

        multi_codes = [link_codes, support_codes, proximity_codes, comp_codes]  # 512 256 128 256
        return multi_codes

class gen_heter_Emb_no(nn.Module):
    def __init__(self):
        super(gen_heter_Emb_no, self).__init__()

        self.pred_mlp_s = MLP(mlp=[512, 512, 512])
        self.pred_mlp_p = MLP(mlp=[512, 512, 512])
        self.pred_mlp_c = MLP(mlp=[512, 512, 512])
        self.pred_mlp_l = MLP(mlp=[512 * 3, 512, 512])


    def forward(self, insnum, pred_codes, pc_geom_info):  # (Nn,1024,3)
        # 输入预训好的边特征　
        # 输出不同类型的关系嵌入：　
        #    Support: visual_i - visual-j
        #    Proximity: bbox offset \\ or 质心距离、方位特征向量
        #    Comparative: 物体size\symmetry属性oneot编码向量的concat　先暂用visual_i - visual-j　+ h V

        pred_idx = torch.LongTensor([[i, j] for i in range(insnum) for j in range(insnum) if i != j])
        diff_codes = pred_codes
        support_codes = self.pred_mlp_s(diff_codes)  # (Ne,256)

        proximity_codes = self.pred_mlp_p(diff_codes)  # (Ne, 640) -> (Ne,512)

        comp_codes = self.pred_mlp_c(diff_codes)  # (Ne, 578) ->(Ne,512)

        link_codes = torch.cat((support_codes, proximity_codes, comp_codes), dim=1)
        link_codes = self.pred_mlp_l(link_codes)  # (Ne,512)

        multi_codes = [link_codes, support_codes, proximity_codes, comp_codes]  # 512 256 128 256
        return multi_codes

class gen_heter_Emb_p(nn.Module):
    def __init__(self):
        super(gen_heter_Emb_p, self).__init__()

        self.pred_mlp_s = MLP(mlp=[512, 512, 512])
        self.pred_mlp_p = MLP(mlp=[512 + 128, 512, 512])
        self.pred_mlp_c = MLP(mlp=[512, 512, 512])
        self.pred_mlp_l = MLP(mlp=[512 * 3, 512, 512])

        self.pos_fc = FullyConnectedNet(5, 64, 128)

    def forward(self, insnum, pred_codes, pc_geom_info):  # (Nn,1024,3)
        # 输入预训好的边特征　
        # 输出不同类型的关系嵌入：　
        #    Support: visual_i - visual-j
        #    Proximity: bbox offset \\ or 质心距离、方位特征向量
        #    Comparative: 物体size\symmetry属性oneot编码向量的concat　先暂用visual_i - visual-j　+ h V

        pred_idx = torch.LongTensor([[i, j] for i in range(insnum) for j in range(insnum) if i != j])
        diff_codes = pred_codes

        # Geometric features
        edges = pred_idx
        bboxes, lwhV, centroid = pc_geom_info
        min_bboxes = bboxes[:, 0]
        max_bboxes = bboxes[:, 1]
        centroid_i = centroid[edges[:, 0],:2]
        centroid_j = centroid[edges[:, 1],:2]
        dx = centroid_i[:, 0] - centroid_j[:, 0]  ##质心  #(No,1)
        dy = centroid_i[:, 1] - centroid_j[:, 1]  ##(No,1)
        #dz = centroid_i[:, 2] - centroid_j[:, 2]  # (No,1)
        bboxes_offset_xy = torch.stack([dx, dy], dim=1).to(torch.float32)  # (Ne,3)
        distance = torch.zeros([pred_codes.shape[0], 1]).cuda()  # #(Ne,1)
        direction = torch.zeros([pred_codes.shape[0], 2]).cuda()  # (Ne,3)

        for i in range(distance.shape[0]):
            distance[i] = torch.norm(centroid_i[i] - centroid_j[i], p=2)  # 需要高斯挤压到0~1 #(Ne,)
            direction[i] = (centroid_i[i] - centroid_j[i]) / torch.norm(centroid_i[i] - centroid_j[i], p=2)  # (Ne,2)
            #print("direction:", direction[i])
            #print("dx, dy, dz: ", bboxes_offset[i])
        pos_features = torch.cat((bboxes_offset_xy, distance, direction), dim=1)  # (Ne,2+1+2)


        support_codes = self.pred_mlp_s(diff_codes)  # (Ne,256)

        pos_codes = self.pos_fc(pos_features)  # MLP 或FCL (Ne, 7)--> (Ne, 128)
        proximity_features = torch.cat((diff_codes, pos_codes), dim=1)  # 512+128  640
        proximity_codes = self.pred_mlp_p(proximity_features)  # (Ne, 640) -> (Ne,512)


        comp_codes = self.pred_mlp_c(diff_codes)  # (Ne, 578) ->(Ne,512)

        link_codes = torch.cat((support_codes, proximity_codes, comp_codes), dim=1)
        link_codes = self.pred_mlp_l(link_codes)  # (Ne,512)

        multi_codes = [link_codes, support_codes, proximity_codes, comp_codes]  # 512 256 128 256
        return multi_codes

class gen_heter_Emb_c(nn.Module):
    def __init__(self):
        super(gen_heter_Emb_c, self).__init__()

        self.pred_mlp_s = MLP(mlp=[512, 512, 512])
        self.pred_mlp_p = MLP(mlp=[512, 512, 512])
        self.pred_mlp_c = MLP(mlp=[512 + 64, 512, 512])
        self.pred_mlp_l = MLP(mlp=[512 * 3, 512, 512])

        self.geom_fc = FullyConnectedNet(4, 32, 64)
        self.geom_fc2 = FullyConnectedNet(64, 128, 256)

    def forward(self, insnum, pred_codes, pc_geom_info):  # (Nn,1024,3)
        # 输入预训好的边特征　
        # 输出不同类型的关系嵌入：　
        #    Support: visual_i - visual-j
        #    Proximity: bbox offset \\ or 质心距离、方位特征向量
        #    Comparative: 物体size\symmetry属性oneot编码向量的concat　先暂用visual_i - visual-j　+ h V

        pred_idx = torch.LongTensor([[i, j] for i in range(insnum) for j in range(insnum) if i != j])
        diff_codes = pred_codes

        # Geometric features
        edges = pred_idx
        bboxes, lwhV, centroid = pc_geom_info



        l, w, h, V = lwhV[:, 0], lwhV[:, 1], lwhV[:, 2], lwhV[:, 3]
        d_l = torch.log(l[edges[:, 0]] / l[edges[:, 1]])
        d_w = torch.log(w[edges[:, 0]] / w[edges[:, 1]])
        d_h = torch.log(h[edges[:, 0]] / h[edges[:, 1]])
        d_V = torch.log(V[edges[:, 0]] / V[edges[:, 1]])  # .reshape(-1, 1)
        geom_features = torch.stack([d_l, d_w, d_h, d_V], dim=1).to(torch.float32)  # (Nn,C_g)
        # geom_codes = self.geom_fc2(geom_codes) #256

        support_codes = self.pred_mlp_s(diff_codes)  # (Ne,256)

        proximity_codes = self.pred_mlp_p(diff_codes)  # (Ne, 640) -> (Ne,512)

        geom_codes = self.geom_fc(geom_features)  # (Ne, 3)--> (Ne, 64)
        comp_features = torch.cat((diff_codes, geom_codes), dim=1)  # (Ne,512+64) 578
        comp_codes = self.pred_mlp_c(comp_features)  # (Ne, 578) ->(Ne,512)

        link_codes = torch.cat((support_codes, proximity_codes, comp_codes), dim=1)
        link_codes = self.pred_mlp_l(link_codes)  # (Ne,512)

        multi_codes = [link_codes, support_codes, proximity_codes, comp_codes]  # 512 256 128 256
        return multi_codes


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size1,  output_size):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, output_size)
        #self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        #self.fc5 = nn.Linear(hidden_size2, output_size)
    def forward(self, x):
        out = nn.functional.relu(self.fc1(x))
        out = nn.functional.relu(self.fc2(out))
        #out = nn.functional.relu(self.fc3(x))
        #out = nn.functional.relu(self.fc4(out))
        out = self.fc3(out)
        return out
