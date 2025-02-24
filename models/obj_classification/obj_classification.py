import numpy as np
import torch
import torch.nn as nn
import pathmagic  # noqa
from models.pred_classification.pointnet import PointNetEncoderwoBN
from models.pred_classification.utils import FocalLoss, MLP
from pointnet2_ops.pointnet2_utils import FurthestPointSampling


def fps_sampling(xyz, npoints):
    # xyz: B, N, 3
    fps = FurthestPointSampling()
    idx = fps.apply(xyz, npoints).long()      # B, N
    return idx.long()


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.latent_dimension = 512
        self.obj_pointnet_lv1 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=self.latent_dimension)
        self.obj_pointnet_lv2 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=self.latent_dimension)
        self.obj_pointnet_lv3 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=self.latent_dimension)
        self.obj_mlp = MLP(mlp=[512*3, 1024, 1024, 512])
        self.obj_classifer = MLP(mlp=[512, 256, 256, 160])

    def forward(self, pc_mat):
        insnum = pc_mat.shape[0]
        pc_mat = self.normalize_regularization(pc_mat) #(B,N,3)
        if self.training:
            pc_mat = torch.bmm(pc_mat, self.random_rotation(insnum))
        idx_256 = fps_sampling(pc_mat, 256)
        idx_128 = fps_sampling(pc_mat, 128)
        pc_256 = torch.cat([pc_mat[i, idx_256[i], :].view(1, 256, 3) for i in range(insnum)], dim=0).contiguous()
        pc_128 = torch.cat([pc_mat[i, idx_128[i], :].view(1, 128, 3) for i in range(insnum)], dim=0).contiguous()
        obj_codes_lv1 = self.obj_pointnet_lv1(pc_mat.transpose(2, 1).contiguous())
        obj_codes_lv2 = self.obj_pointnet_lv2(pc_256.transpose(2, 1).contiguous())
        obj_codes_lv3 = self.obj_pointnet_lv3(pc_128.transpose(2, 1).contiguous())
        obj_codes = torch.cat([obj_codes_lv1, obj_codes_lv2, obj_codes_lv3], dim=1)
        obj_codes = self.obj_mlp(obj_codes)
        obj_output = self.obj_classifer(obj_codes)
        return obj_output, obj_codes

    def normalize_regularization(self, pc_mat):
        maxs = torch.max(pc_mat, dim=1, keepdim=True)[0]
        mins = torch.min(pc_mat, dim=1, keepdim=True)[0]
        offsets = (maxs + mins) / 2
        scale = torch.max((maxs - mins), dim=2)[0].view(-1, 1, 1).contiguous()
        pc_mat -= offsets
        pc_mat /= scale
        return pc_mat

    def random_rotation(self, insnum):
        rm = []
        for i in range(insnum):
            rotation_angle = np.random.uniform(-1, 1) * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
            rotation_matrix = torch.Tensor(rotation_matrix).cuda()
            rm.append(rotation_matrix)
        rm = torch.stack(rm, dim=0)
        return rm


class get_loss(nn.Module):
    def __init__(self, gamma, obj_w, use_weight=False):
        super(get_loss, self).__init__()
        if use_weight:
            self.focal_loss = FocalLoss(class_num=160, alpha=obj_w, gamma=gamma, size_average=True)
        else:
            self.focal_loss = FocalLoss(class_num=160, alpha=None, gamma=gamma, size_average=True)

    def forward(self, obj_output, gt_obj):
        gt_obj = self.prepare_objgt(gt_obj)
        obj_loss = self.focal_loss(obj_output, gt_obj)
        loss = obj_loss
        return loss

    def prepare_objgt(self, obj_gt):
        insnum = obj_gt.shape[0]
        onehot = torch.zeros(insnum, 160).float().cuda()
        for i in range(insnum):
            onehot[i, obj_gt[i]] = 1
        return onehot
