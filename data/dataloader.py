import torch
import numpy as np
from torch.utils.data import Dataset
#from vtkplotter import show, Points


with open('./data/classes.txt', 'r') as f:
    classes = f.readlines()
    for i in range(len(classes)):
        classes[i] = classes[i].strip()
f.close()


with open('./data/relationships.txt', 'r') as f:
    relationships = f.readlines()
    for i in range(len(relationships)):
        relationships[i] = relationships[i].strip()
f.close()


def calculate_geom_info(obj_points):
    bboxes = []
    lwhV = []
    centroid = []
    info = []
    # 3Dbbox
    for idx in range(len(obj_points)):
        bbox = [obj_points[idx][:, :3].min(0),obj_points[idx][:, :3].max(0)]
        bboxes.append(np.array(bbox))
        l = bbox[1][0] - bbox[0][0]
        w = bbox[1][1] - bbox[0][1]
        h = bbox[1][2] - bbox[0][2]
        x0, y0, z0 = bbox[0][0]+l/2, bbox[0][1]+w/2, bbox[0][2]+h/2,
        V = l*w*h
        lwhV.append(np.array([l, w, h, V]))
        centroid.append(np.array([x0, y0, z0]))
    info = np.array(bboxes), np.array(lwhV), np.array(centroid)
    #info = bboxes, lwhV, centroid
    return info

def visualize(mat):
    insnum = mat.shape[0]
    rand_color = np.random.rand(insnum, 3)
    pc_mat = []
    color_mat = []
    for i in range(insnum):
        pc_mat.append(mat[i])
        c = rand_color[i].reshape((1, 3)).repeat(512, axis=0)
        color_mat.append(c)
    pc_mat = np.vstack(pc_mat)
    color_mat = np.vstack(color_mat)
    pc = Points(pc_mat, c=color_mat)
    show(pc, interactive=1)


class DataLoader_3DSSG(Dataset):
    def __init__(self, training=True, shuffle=False, norm=False, half=False, per25=False):
        self.training = training
        self.norm = norm

        if shuffle:#False
            self.training_txt = 'XX'
            self.test_txt = 'XX'
        else:
            if half:#False
                self.training_txt = '/home/ma/dataset/3DSSG/3DSSG_subset/train/training_txt.txt'
            elif per25:#True
                self.training_txt = '/home/ma/dataset/3DSSG/3DSSG_subset/train/training_txt.txt'
            else:
                self.training_txt = '/home/ma/dataset/3DSSG/3DSSG_subset/train/training_txt.txt'
            self.test_txt = "/home/ma/dataset/3DSSG/3DSSG_subset/test/testing_txt.txt"


        self.training_list = []
        self.test_list = []

        with open(self.training_txt, 'r') as f:
            self.training_list = f.readlines()
            for i in range(len(self.training_list)):
                self.training_list[i] = self.training_list[i].strip()
        f.close()

        with open(self.test_txt, 'r') as f:
            self.test_list = f.readlines()
            for i in range(len(self.test_list)):
                self.test_list[i] = self.test_list[i].strip()
        f.close()

        self.training_len = len(self.training_list)
        self.testing_len = len(self.test_list)

        self.obj_w = torch.Tensor(np.load('./data/obj_w.npy')).cuda()
        self.pred_w = torch.Tensor([0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1]).cuda()

    def __len__(self):
        if self.training:
            return self.training_len
        else:
            return self.testing_len

    def __getitem__(self, index):
        if self.training:
            folder = self.training_list[index]
        else:
            folder = self.test_list[index]

        obj_gt = np.load(folder +'/gt_obj.npy')
        #print(folder)
        rel_gt = np.load(folder + '/gt_relationships.npy')  # (25, 3)
        #print(rel_gt)
        pc_mat = np.load(folder + '/pointcloud_1024_ins.npy')[:, :, 0:3]
        #print(pc_mat)
        pc_geom_info = calculate_geom_info(pc_mat)


        if self.norm:
            pc_mat = self.normalize(pc_mat)
        return torch.Tensor(pc_mat), pc_geom_info, torch.IntTensor(obj_gt), torch.IntTensor(rel_gt)
        #return torch.Tensor(pc_mat), torch.IntTensor(obj_gt), torch.IntTensor(rel_gt)

    def visualize(self, index):
        if self.training:
            folder = self.training_list[index]
        else:
            folder = self.test_list[index]
        obj_gt = np.load(folder + '/gt_obj.npy')
        rel_gt = np.load(folder + '/gt_relationships.npy')
        pc_mat = np.load(folder + '/pointcloud_1024_ins.npy')[:, :, 0:6]
        for i in range(rel_gt.shape[0]):
            print(classes[obj_gt[rel_gt[i, 0]]] + '->' + classes[obj_gt[rel_gt[i, 1]]] + '=' + relationships[rel_gt[i, 2]])
        pc_mat = pc_mat.reshape(-1, 6)
        pc = Points(pc_mat[:, 0:3], c=pc_mat[:, 3:6])
        show(pc, interactive=1)

    def normalize(self, pc_mat):
        xyz = pc_mat[:, :, 0:3]
        maxs = np.max(np.max(xyz, axis=0), axis=0)
        mins = np.min(np.min(xyz, axis=0), axis=0)
        offsets = (maxs + mins) / 2
        scale = (maxs - mins).max()
        pc_mat[:, :, 0:3] -= offsets
        pc_mat[:, :, 0:3] /= scale
        mins = np.min(np.min(xyz, axis=0), axis=0)
        mins[0] = 0
        mins[1] = 0
        pc_mat[:, :, 0:3] -= mins
        return pc_mat


if __name__ == "__main__":
    dataset3dssg_train = DataLoader_3DSSG(training=False)
    pc_mat, obj_gt, rel_gt = dataset3dssg_train.__getitem__(173)
    print(obj_gt)
    print(rel_gt)
    dataset3dssg_train.visualize(173)
    print(dataset3dssg_train.obj_w)
