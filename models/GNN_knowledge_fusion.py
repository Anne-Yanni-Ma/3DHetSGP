import torch
import torch.nn as nn
import torch.nn.functional as F
import pathmagic  # noqa
import numpy as np
from utils import FocalLoss, MLP
from gnn_models import GraphEncoderKnowledgeFusion
from graph import SceneGraph


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.assimilation = 2
        self.knode = torch.Tensor(np.load('./data/meta_embedding/meta_embedding_node.npy')).cuda()
        self.kedge = torch.Tensor(np.load('./data/meta_embedding/meta_embedding_edge.npy')).cuda()
        self.gnn = GraphEncoderKnowledgeFusion(ndim=512, nlayer=5)
        self.node_mlp = MLP(mlp=[512, 512, 512])
        self.knode_mlp = MLP(mlp=[512, 512, 512])
        self.edge_mlp = MLP(mlp=[512, 512, 512])
        self.kedge_mlp = MLP(mlp=[512, 512, 512])
        self.node_classifer = MLP(mlp=[512, 256, 256, 160])
        self.edge_classifer = MLP(mlp=[512, 256, 128, 27])
        self.conv1d = torch.nn.Conv1d(256, 256, 1)


    def forward(self, obj_codes, pred_codes): #obj_codes(Nn,C_konw) pred_codes (Ne, C)
        insnum = obj_codes.shape[0] # Nn
        edge_index = self.prepare_edges(insnum) # (2, Nn*(Nn-1))
        g = SceneGraph(x=obj_codes, edge_index=edge_index, edge_attr=pred_codes, edge_weight=torch.ones(pred_codes.shape[0]))  #edge_weight(Ne,)
        knode = torch.zeros(insnum, 512).cuda()    # ([7, 512])
        kedge = torch.zeros(insnum*(insnum-1), 512).cuda()  # ([42, 512])

        for i in range(self.assimilation):
            node_embed, edge_embed = self.gnn(g, knode, kedge)  # ([7, 512])  # ([42, 512])

            node_weight, edge_weight = self.node_classifer(node_embed), self.edge_classifer(edge_embed) # ([7, 160])  # ([42, 27])
            node_weight, edge_weight = F.softmax(node_weight, dim=1), F.softmax(edge_weight, dim=1) #
            knode, kedge = self.select_knowledge(node_weight, edge_weight)
            g.reset_graph()
        return node_weight, edge_weight  # ([7, 160])  # ([42, 27])

    def select_knowledge(self, node_weight, edge_weight):
        node_topk_inds = node_weight.topk(k=5).indices
        node_inds = torch.zeros(node_weight.shape).cuda()
        for i in range(node_topk_inds.shape[0]):
            node_inds[i, node_topk_inds[i]] = 1
        knode = torch.mm(node_inds, self.knode)

        edge_topk_inds = edge_weight.topk(k=5).indices
        edge_inds = torch.zeros(edge_weight.shape).cuda()
        for i in range(edge_topk_inds.shape[0]):
            edge_inds[i, edge_topk_inds[i]] = 1
        kedge = torch.mm(edge_inds, self.kedge)
        return knode, kedge

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



class get_loss(nn.Module):
    def __init__(self, alpha, beta, gamma, obj_w, pred_w):
        super(get_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.focal_loss_obj = FocalLoss(class_num=160, alpha=obj_w, gamma=gamma, size_average=True, use_softmax=False)
        self.focal_loss_pred = FocalLoss(class_num=27, alpha=pred_w, gamma=gamma, size_average=True, use_softmax=False)

    def forward(self, node_output, edge_output, gt_obj, gt_rel):
        objgt_onehot = self.prepare_onehot_objgt(gt_obj)  #([7, 160])
        predgt_onehot = self.prepare_onehot_predgt(gt_obj, gt_rel)  # ([42, 27])
        obj_loss = self.focal_loss_obj(node_output, objgt_onehot)
        pred_loss = self.focal_loss_pred(edge_output, predgt_onehot)
        #print("obj_loss : pred_loss, ", obj_loss, " , ", pred_loss)  #   5~10 : 1
        loss = self.alpha * obj_loss + self.beta * pred_loss ## alpha:1 beta: 0.1
        return loss

    def prepare_onehot_predgt(self, gt_obj, gt_rel):  #gt_obj (No,), gt_rel(Ne,3)
        insnum = gt_obj.shape[0]
        onehot_gt = torch.zeros((insnum * insnum - insnum, 27)).cuda()

        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            if idx_i < idx_j:
                onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(gt_rel[i, 2])] = 1
            elif idx_i > idx_j:
                onehot_gt[int(idx_i * (insnum-1) + idx_j), int(gt_rel[i, 2])] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt

    def prepare_onehot_objgt(self, gt_obj):
        insnum = gt_obj.shape[0]
        onehot = torch.zeros(insnum, 160).float().cuda()
        for i in range(insnum):
            onehot[i, gt_obj[i]] = 1
        return onehot
