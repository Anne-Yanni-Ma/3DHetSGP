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
        self.pred_pointnet_lv1 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=512)
        self.pred_pointnet_lv2 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=512)
        self.pred_pointnet_lv3 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=512)
        self.pred_mlp = MLP(mlp=[512*3, 1024, 1024, 512])
        self.pred_mlp2 = MLP(mlp=[515, 512, 512])
        self.pred_classifer = MLP(mlp=[512, 256, 128, 27])

    def forward(self, pc_mat):  # (Nn,1024,3)
        insnum = pc_mat.shape[0] # Nn
        pc_mat = self.normalize_regularize(pc_mat)
        # PointNet stage
        pc_center = torch.mean(pc_mat, dim=1)  # 按列求平均值 返回形状是(行数,1)  pc_center  (Nn,1,3)
        idx_256 = fps_sampling(pc_mat, 256) # (Nn,256,3)
        idx_128 = fps_sampling(pc_mat, 128) # (Nn,128,3)
        pc_256 = torch.cat([pc_mat[i, idx_256[i], :].view(1, 256, 3) for i in range(insnum)], dim=0).contiguous()
        pc_128 = torch.cat([pc_mat[i, idx_128[i], :].view(1, 128, 3) for i in range(insnum)], dim=0).contiguous()
        pc_codes_lv1 = self.pred_pointnet_lv1(pc_mat.transpose(2, 1).contiguous())
        pc_codes_lv2 = self.pred_pointnet_lv2(pc_256.transpose(2, 1).contiguous())
        pc_codes_lv3 = self.pred_pointnet_lv3(pc_128.transpose(2, 1).contiguous())
        pc_codes = self.pred_mlp(torch.cat([pc_codes_lv1, pc_codes_lv2, pc_codes_lv3], dim=1))
        pred_idx = torch.LongTensor([[i, j] for i in range(insnum) for j in range(insnum) if i != j])
        pc_codes_i = torch.cat([pc_codes[pred_idx[:, 0]], pc_center[pred_idx[:, 0]]], dim=1)
        pc_codes_j = torch.cat([pc_codes[pred_idx[:, 1]], pc_center[pred_idx[:, 1]]], dim=1)
        diff_codes = pc_codes_i - pc_codes_j
        pred_codes = self.pred_mlp2(diff_codes)
        # Classification
        pred_output = self.pred_classifer(pred_codes)  # pred_output(Ne,27)    pred_codes (Ne, C)                 # (n*n) * 27
        return pred_output, pred_codes

    def normalize_regularize(self, pc_mat):
        maxs = torch.max(pc_mat, dim=1, keepdim=True).values
        mins = torch.min(pc_mat, dim=1, keepdim=True).values
        maxs = torch.max(maxs, dim=0).values
        mins = torch.min(mins, dim=0).values
        offsets = (maxs + mins) / 2
        scale = torch.max((maxs - mins), dim=1)[0]
        pc_mat -= offsets
        pc_mat /= scale
        return pc_mat


class get_loss(nn.Module):
    def __init__(self, gamma, pred_w, use_weight=True):
        super(get_loss, self).__init__()
        self.gamma = gamma
        if use_weight:
            self.focal_loss = FocalLoss(class_num=27, alpha=pred_w, gamma=gamma, size_average=True)
        else:
            self.focal_loss = FocalLoss(class_num=27, alpha=None, gamma=gamma, size_average=True)

    def forward(self, pred_output, obj_gt, rel_gt):
        pred_gt = self.prepare_predgt(obj_gt, rel_gt)
        loss = self.focal_loss(pred_output, pred_gt)
        return loss

    def prepare_predgt(self, obj_gt, rel_gt):
        insnum = obj_gt.shape[0]
        onehot_gt = torch.zeros((insnum * insnum - insnum, 27)).long().cuda()
        for i in range(rel_gt.shape[0]):
            idx_i = rel_gt[i, 0]
            idx_j = rel_gt[i, 1]
            if idx_i < idx_j:
                onehot_gt[int(idx_i * (insnum-1) + idx_j - 1), int(rel_gt[i, 2])] = 1
            elif idx_i > idx_j:
                onehot_gt[int(idx_i * (insnum-1) + idx_j), int(rel_gt[i, 2])] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt
