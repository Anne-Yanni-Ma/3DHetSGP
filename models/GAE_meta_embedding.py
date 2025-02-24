import torch
import torch.nn as nn
import torch.nn.functional as F
import pathmagic  # noqa
from utils import MLP, FocalLoss
from gnn_models import GraphEncoder
from graph import SceneGraph

class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        # Graph Autoencoder
        self.node_onehot_encoder = MLP(mlp=[160, 256, 256, 512])
        self.edge_onehot_encoder = MLP(mlp=[27, 128, 256, 512])
        self.gnn_encoder = GraphEncoder(ndim=512, nlayer=5)
        self.node_decoder = MLP(mlp=[512,256, 256, 160])
        self.edge_decoder = MLP(mlp=[512,256, 128, 27])
        # Prototype Representation
        self.node_prototype = nn.Embedding(160, 512)
        self.edge_prototype = nn.Embedding(27, 512)

        torch.nn.init.xavier_normal_(self.node_prototype.weight.data)
        torch.nn.init.xavier_normal_(self.edge_prototype.weight.data)

    def forward(self, obj_onehot, pred_onehot, edge_index):
        obj_codes = self.node_onehot_encoder(obj_onehot)  #(7,160)  (7,512)  .([42, 512])
        pred_codes = self.edge_onehot_encoder(pred_onehot) #(42,27)  （42,512）  #([1722, 512])
        g = SceneGraph(x=obj_codes, edge_index=edge_index, edge_attr=pred_codes, edge_weight=torch.ones(pred_codes.shape[0]))
        node_codes, edge_codes = self.gnn_encoder(g) # (7,512) （42,512）

        node_logits = self.node_decoder(node_codes)
        edge_logits = self.edge_decoder(edge_codes)
        node_output = F.softmax(node_logits, dim=1) #(7,160)
        edge_output = F.softmax(edge_logits, dim=1) #(42,27)
        return node_output, edge_output, node_codes, edge_codes, self.node_prototype.weight, self.edge_prototype.weight, obj_onehot, pred_onehot


class get_loss(nn.Module):
    def __init__(self, w_focal=1, w_prot=1):
        super(get_loss, self).__init__()
        self.w_focal = w_focal
        self.w_prot = w_prot

        self.focal_loss_obj = FocalLoss(class_num=160, alpha=None, gamma=2, size_average=True, use_softmax=False)
        self.focal_loss_pred = FocalLoss(class_num=27, alpha=None, gamma=2, size_average=True, use_softmax=False)

    def forward(self, node_output, edge_output, node_codes, edge_codes, node_meta_embedding, edge_meta_embedding, node_gt, edge_gt):
        obj_loss = self.focal_loss_obj(node_output, node_gt)
        pred_loss = self.focal_loss_pred(edge_output, edge_gt)
        node_prototype_distance = self.meta_embedding_similarity(node_codes, node_meta_embedding, node_gt) # node_codes(Nn,512)  node_meta_embedding(160,512) node_gt(Nn,160)
        edge_prototype_distance = self.meta_embedding_similarity(edge_codes, edge_meta_embedding, edge_gt)

        loss_focal = obj_loss + pred_loss
        loss_meta_embedding = node_prototype_distance + edge_prototype_distance

        loss = (self.w_focal * loss_focal) + (self.w_prot * loss_meta_embedding)
        return loss

    def meta_embedding_similarity(self, codes, prototype, onehot):
        ''' code: N * 512
            prototype: n * 512
            onehot: N * n
        '''
        # node_codes(Nn,512)  node_meta_embedding(160,512) node_gt(Nn,160)
        N = onehot.shape[0] # node_num
        distance = 0

        for i in range(N):

            idx = onehot[i].bool()
            prot = torch.sum(prototype[idx], dim=0)
            distance += F.pairwise_distance(codes[i].view(1, -1), prot.view(1, -1), p=2) # dist( Tensor(512),Tensor(512))
        distance = distance / N   #60.4

        return distance






