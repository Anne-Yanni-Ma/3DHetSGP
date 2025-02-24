import torch
import torch.nn as nn
import torch.nn.functional as F
import pathmagic  # noqa
import numpy as np
from utils import FocalLoss, MLP
from graph import SceneGraph


support_label = [1,14,15,16,17,18,19,20,21,22,23,24,25,26]  # len: 14
suppport_map = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
proximity_label = [2,3,4,5,6,7]
proximity_map = [1,2,3,4,5,6]
comparative_label = [8,9,10,11,12,13] # 6个
comparative_map = [1,2,3,4,5,6]
cp_label = [2,3,4,5,6,7,8,9,10,11,12,13] #12个
ppred_w =  torch.Tensor([0.25, 1, 1, 1, 1, 1, 1]).cuda()

def re_edge(edge_f,map):
    redge_f = edge_f.clone()
    for i in range(edge_f.shape[0]):
        redge_f[i] = edge_f[int(map[i])]
    return redge_f


class get_loss_l(nn.Module):
    def __init__(self, alpha, beta, gamma,pred_w):
        super(get_loss_l, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.focal_loss_link = FocalLoss(class_num=2, alpha=pred_w, gamma=1.0, size_average=True, use_softmax=False)

    def forward(self, edge_output_l, gt_obj, gt_rel):
        link_gt =  self.prepare_linkgt(gt_obj,gt_rel)
        #link_loss =  F.binary_cross_entropy_with_logits(edge_output_l, link_gt.float()) # 不考虑none 0.7342
        link_loss = self.focal_loss_link(edge_output_l,link_gt)
        return link_loss

    def prepare_linkgt(self, obj_gt, rel_gt):
        insnum = obj_gt.shape[0]
        #onehot_gt = torch.zeros((insnum * insnum - insnum, 2)).long().cuda()
        link_gt = torch.zeros((insnum * insnum - insnum, 2)).long().cuda()

        for i in range(rel_gt.shape[0]):
            idx_i = rel_gt[i, 0]
            idx_j = rel_gt[i, 1]
            if idx_i < idx_j:
                link_gt[int(idx_i * (insnum - 1) + idx_j - 1),1] = 1
            elif idx_i > idx_j:
                link_gt[int(idx_i * (insnum - 1) + idx_j),1] = 1
        for i in range(link_gt.shape[0]):
            if link_gt[i,1]==0:
                link_gt[i,0]=1
        return link_gt




class get_loss_s(nn.Module):
    def __init__(self, alpha, beta, gamma, pred_w):
        super(get_loss_s, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss_support = FocalLoss(class_num=15, alpha=pred_w, gamma=1.0, size_average=True, use_softmax=False)

    def forward(self,  edge_output_s, gt_obj, gt_rel):

        support_gt =self.prepare_onehot_geter(gt_obj,gt_rel, support_label)
        zeros_s = torch.zeros(support_gt.shape).cuda()
        if (torch.equal(support_gt[:, 1:], zeros_s[:, 1:])):
            support_loss = 0
        else:
            #support_loss = self.focal_loss_support(edge_output_s, support_gt)
            support_loss =  F.binary_cross_entropy_with_logits(edge_output_s, support_gt.float()) # 0.6940
        return support_loss


    def prepare_onehot_geter(self, gt_obj, gt_rel, part_label):  #gt_obj (No,), gt_rel(Ne,3)
        insnum = gt_obj.shape[0]
        prednum = len(part_label)
        onehot_gt = torch.zeros((insnum * insnum - insnum, prednum+1)).cuda()

        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            pred_gt = gt_rel[i,2]
            if(int(pred_gt) in part_label):
                pred_gt_map = part_label.index(pred_gt)+1 #
                if idx_i < idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(pred_gt_map)] = 1
                elif idx_i > idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j), int(pred_gt_map)] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt


class get_loss_p(nn.Module):
    def __init__(self, alpha, beta, gamma, pred_w):
        super(get_loss_p, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss_proximity = FocalLoss(class_num=7, alpha=pred_w, gamma=1.0, size_average=True, use_softmax=False)


    def forward(self,  edge_output_prox, gt_obj, gt_rel):
        proximity_gt = self.prepare_onehot_geter(gt_obj,gt_rel, proximity_label)

        zeros_prox = torch.zeros(proximity_gt.shape).cuda()
        if (torch.equal(proximity_gt[:, 1:], zeros_prox[:, 1:])):
            proximity_loss = 0
        else:
            proximity_loss = self.focal_loss_proximity(edge_output_prox, proximity_gt)
            #proximity_loss =   F.binary_cross_entropy(edge_output_prox, proximity_gt.float() , weight=ppred_w)


        return proximity_loss

    def prepare_onehot_geter(self, gt_obj, gt_rel, part_label):  #gt_obj (No,), gt_rel(Ne,3)
        insnum = gt_obj.shape[0]
        prednum = len(part_label)
        onehot_gt = torch.zeros((insnum * insnum - insnum, prednum+1)).cuda()

        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            pred_gt = gt_rel[i,2]
            if(int(pred_gt) in part_label):
                pred_gt_map = part_label.index(pred_gt)+1
                if idx_i < idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(pred_gt_map)] = 1
                elif idx_i > idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j), int(pred_gt_map)] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt

class get_loss_p_sft(nn.Module):
    def __init__(self, alpha, beta, gamma, pred_w):
        super(get_loss_p_sft, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss_proximity = FocalLoss(class_num=7, alpha=pred_w, gamma=1.0, size_average=True, use_softmax=False)


    def forward(self,  edge_output_prox, gt_obj, gt_rel):
        proximity_gt = self.prepare_onehot_geter(gt_obj,gt_rel, proximity_label)

        zeros_prox = torch.zeros(proximity_gt.shape).cuda()
        if (torch.equal(proximity_gt[:, 1:], zeros_prox[:, 1:])):
            proximity_loss = 0
        else:
            proximity_loss = self.focal_loss_proximity(edge_output_prox, proximity_gt)

        return proximity_loss

    def prepare_onehot_geter(self, gt_obj, gt_rel, part_label):  #gt_obj (No,), gt_rel(Ne,3)
        insnum = gt_obj.shape[0]
        prednum = len(part_label)
        onehot_gt = torch.zeros((insnum * insnum - insnum, prednum+1)).cuda()

        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            pred_gt = gt_rel[i,2]
            if(int(pred_gt) in part_label):
                pred_gt_map = part_label.index(pred_gt)+1
                if idx_i < idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(pred_gt_map)] = 1
                elif idx_i > idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j), int(pred_gt_map)] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt


class get_loss_c(nn.Module):
    def __init__(self, alpha, beta, gamma, pred_w):
        super(get_loss_c, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss_comp = FocalLoss(class_num=7, alpha=pred_w, gamma=1.0, size_average=True, use_softmax=False)

    def forward(self,  edge_output_comp, gt_obj, gt_rel):
        comparative_gt = self.prepare_onehot_geter(gt_obj,gt_rel, comparative_label)
        zeros_comp = torch.zeros(comparative_gt.shape).cuda()
        if (torch.equal(comparative_gt[:, 1:], zeros_comp[:, 1:])):
            comparative_loss = 0
        else:
            comparative_loss =  self.focal_loss_comp(edge_output_comp, comparative_gt)
            #comparative_loss = F.binary_cross_entropy_with_logits(edge_output_comp, comparative_gt.float()) #0.711
        return comparative_loss

    def prepare_onehot_geter(self, gt_obj, gt_rel, part_label):  #gt_obj (No,), gt_rel(Ne,3)
        insnum = gt_obj.shape[0]
        prednum = len(part_label)
        onehot_gt = torch.zeros((insnum * insnum - insnum, prednum+1)).cuda()

        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            pred_gt = gt_rel[i,2]
            if(int(pred_gt) in part_label):
                pred_gt_map = part_label.index(pred_gt)+1
                if idx_i < idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(pred_gt_map)] = 1
                elif idx_i > idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j), int(pred_gt_map)] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt


class get_loss_obj(nn.Module):
    def __init__(self, alpha, beta, gamma, obj_w):
        super(get_loss_obj, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss_obj = FocalLoss(class_num=160, alpha=obj_w, gamma=gamma, size_average=True, use_softmax=False)

    def forward(self, node_output, gt_obj, gt_rel):
        objgt_onehot = self.prepare_onehot_objgt(gt_obj)  #([7, 160])

        obj_loss = self.focal_loss_obj(node_output, objgt_onehot)

        return obj_loss


    def prepare_onehot_objgt(self, gt_obj):
        insnum = gt_obj.shape[0]
        onehot = torch.zeros(insnum, 160).float().cuda()
        for i in range(insnum):
            onehot[i, gt_obj[i]] = 1
        return onehot


class get_loss_all(nn.Module):
    def __init__(self, alpha, beta, gamma, obj_w):
        super(get_loss_all, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.focal_loss_obj = FocalLoss(class_num=160, alpha=obj_w, gamma=gamma, size_average=True, use_softmax=False)
        self.focal_loss_link = FocalLoss(class_num=2, alpha=None, gamma=1.0, size_average=True, use_softmax=False)
        self.focal_loss_support = FocalLoss(class_num=15, alpha=None, gamma=1.0, size_average=True, use_softmax=False)
        self.focal_loss_proximity = FocalLoss(class_num=7, alpha=None, gamma=1.0, size_average=True, use_softmax=False)


    def forward(self, node_output, multi_edge_output, gt_obj, gt_rel,type_output):
        edge_output_l, edge_output_s, edge_output_prox, edge_output_comp = multi_edge_output
        objgt_onehot = self.prepare_onehot_objgt(gt_obj)  #([7, 160])
        link_gt =  self.prepare_linkgt(gt_obj,gt_rel)
        support_gt =self.prepare_onehot_geter(gt_obj,gt_rel, support_label)
        proximity_gt = self.prepare_onehot_geter(gt_obj,gt_rel, proximity_label)
        comparative_gt = self.prepare_onehot_geter(gt_obj,gt_rel, comparative_label)
        #edge_output_s= torch.sigmoid(edge_output_s)
        #edge_output_prox= torch.sigmoid(edge_output_prox)
        obj_loss = self.focal_loss_obj(node_output, objgt_onehot)
        link_loss =  F.binary_cross_entropy_with_logits(edge_output_l, link_gt.float())
        zeros_s = torch.zeros(support_gt.shape).cuda()
        zeros_prox = torch.zeros(proximity_gt.shape).cuda()

        support_loss =  F.binary_cross_entropy_with_logits(edge_output_s, support_gt.float())
        proximity_loss =  self.focal_loss_proximity(edge_output_prox, proximity_gt)
        comparative_loss = self.focal_loss_proximity(edge_output_comp, comparative_gt)

        pred_loss = link_loss  + support_loss + proximity_loss + comparative_loss #+ type_loss

        loss = self.alpha * obj_loss + self.beta * pred_loss ## alpha:1 beta: 0.1
        return loss


    def prepare_onehot_geter(self, gt_obj, gt_rel, part_label):  #gt_obj (No,), gt_rel(Ne,3)
        insnum = gt_obj.shape[0]
        prednum = len(part_label)
        onehot_gt = torch.zeros((insnum * insnum - insnum, prednum+1)).cuda()

        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            pred_gt = gt_rel[i,2]
            if(int(pred_gt) in part_label):
                pred_gt_map = part_label.index(pred_gt)+1 # label映射为label map, 空出0 for none
                if idx_i < idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(pred_gt_map)] = 1
                elif idx_i > idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j), int(pred_gt_map)] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt


    def prepare_onehot_support(self, gt_obj, gt_rel):  #gt_obj (No,), gt_rel(Ne,3)
        insnum = gt_obj.shape[0]
        onehot_gt = torch.zeros((insnum * insnum - insnum, 15)).cuda()

        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            pred_gt = gt_rel[i,2]
            if(pred_gt in support_label):
                if idx_i < idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(gt_rel[i, 2])] = 1
                elif idx_i > idx_j:
                    onehot_gt[int(idx_i * (insnum-1) + idx_j), int(gt_rel[i, 2])] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt


    def prepare_linkgt(self, obj_gt, rel_gt):
        insnum = obj_gt.shape[0]
        #onehot_gt = torch.zeros((insnum * insnum - insnum, 2)).long().cuda()
        link_gt = torch.zeros((insnum * insnum - insnum, 2)).long().cuda()
        for i in range(rel_gt.shape[0]):
            idx_i = rel_gt[i, 0]
            idx_j = rel_gt[i, 1]
            if idx_i < idx_j:
                link_gt[int(idx_i * (insnum - 1) + idx_j - 1),1] = 1
            elif idx_i > idx_j:
                link_gt[int(idx_i * (insnum - 1) + idx_j),1] = 1
        for i in range(link_gt.shape[0]):
            if link_gt[i,1]==0:
                link_gt[i,0]=1
        return link_gt


    def prepare_typegt(self, obj_gt, rel_gt):
        insnum = obj_gt.shape[0]
        # onehot_gt = torch.zeros((insnum * insnum - insnum, 2)).long().cuda()
        type_gt = torch.zeros((insnum * insnum - insnum, 3)).long().cuda()
        # none, support, c_p

        for i in range(rel_gt.shape[0]):
            idx_i = rel_gt[i, 0]
            idx_j = rel_gt[i, 1]
            predgt =  rel_gt[i, 2]
            if (predgt in support_label):
                type_id = 1
            elif (predgt in cp_label):
                type_id = 2
            else: type_id = 0
            if idx_i < idx_j:
                type_gt[int(idx_i * (insnum - 1) + idx_j - 1), type_id] = 1
            elif idx_i > idx_j:
                type_gt[int(idx_i * (insnum - 1) + idx_j), type_id] = 1

        for i in range(insnum * insnum - insnum):
            if torch.sum(type_gt[i, :]) == 0:
                type_gt[i, 0] = 1

        return type_gt


    def prepare_onehot_objgt(self, gt_obj):
        insnum = gt_obj.shape[0]
        onehot = torch.zeros(insnum, 160).float().cuda()
        for i in range(insnum):
            onehot[i, gt_obj[i]] = 1
        return onehot
