import os
import sys
import json
import numpy as np
import torch
from pointnet2_ops.pointnet2_utils import FurthestPointSampling


with open('./data/classes.txt', 'r') as f:
    classes = f.readlines()
    for i in range(len(classes)):
        classes[i] = classes[i].strip()
f.close()

PATH_DATA = '/home/ma/dataset/3DSSG/'
# 3RScan data
PATH_R3Scan = '/home/ma/dataset/3RScan'
# output
#PATH_OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# read data from 3RScan

# read gt from 3DSSG-subset

def read_obj(filename):
    """ read point cloud from OBJ file"""
    with open(filename) as file:
        point_cloud = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                point_cloud.append((float(strs[1]), float(strs[2]), float(strs[3])))
        point_cloud = np.array(point_cloud)
    return point_cloud

def pc_normalize(pc):
    pc_ = pc[:,:3]
    centroid = np.mean(pc_, axis=0)
    pc_ = pc_ - centroid
    m = np.max(np.sqrt(np.sum(pc_ ** 2, axis=1)))
    pc_ = pc_ / m
    if pc.shape[1] > 3:
        pc = np.concatenate((pc_, pc[:,3].reshape(-1,1)), axis=1)
    else:
        pc = pc_
    return pc

def fps_sampling(xyz, npoints):
    # xyz: B, N, 3
    fps = FurthestPointSampling()
    idx = fps.apply(xyz, npoints).long()      # B, N
    return idx.long()


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    '''if N < npoint:
        return point'''
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def normalize_regularization(pc_mat):
    maxs = np.max(pc_mat, axis=1)[0]
    mins = np.min(pc_mat, axis=1)[0]
    offsets = (maxs + mins) / 2
    scale = np.max((maxs - mins), dim=2)[0].view(-1, 1, 1).contiguous()
    pc_mat -= offsets
    pc_mat /= scale
    return pc_mat


def process_onescan(relationships_scan):
    scan = relationships_scan["scan"]
    scan_id = scan + "-" + str(hex(relationships_scan["split"]))[-1]
    print(scan_id)
    #scan_id = relationships_scan["scan"] + "-" + str(hex(relationships_scan["split"]))[-1]

    # # avoid duplicate computing
    # path = os.path.join(CONF.PATH.R3Scan, "{}/data_dict_{}.json".format(scan_id[:-2], scan_id[-1]))
    # if os.path.exists(path):
    #     return

    # load class and relationships dict
    word2idx = {}
    index = 0
    file = open(os.path.join(PATH_DATA, '3DSSG_subset/classes.txt'), 'r')

    category = file.readline()[:-1]
    while category:
        word2idx[category] = index
        category = file.readline()[:-1]
        index += 1

    rel2idx = {}
    index = 0
    file = open(os.path.join(PATH_DATA, "3DSSG_subset/relationships.txt"), 'r')
    category = file.readline()[:-1]
    while category:
        rel2idx[category] = index
        category = file.readline()[:-1]
        index += 1

    # read point cloud from OBJ file
    scan = scan_id[:-2]
    pc_array = read_obj(os.path.join(PATH_R3Scan, "{}/mesh.refined.obj".format(
        scan)))  # (100775,3) scan '752cc578-920c-26f5-8d8d-8a4239658074'
    # group points in the same segment
    segments = {}  # key:segment id, value: points belong to this segment
    with open(os.path.join(PATH_R3Scan, "{}/mesh.refined.0.010000.segs.json".format(scan)), 'r') as f:
        seg_indices = json.load(f)["segIndices"]  # (100775)
        for index, i in enumerate(seg_indices):
            if i not in segments:
                segments[i] = []
            segments[i].append(pc_array[index])
        # segments (2505)
    # group points of the same object
    # filter the object which does not belong to this split
    obj_id_list = []
    for k, _ in relationships_scan["objects"].items():
        obj_id_list.append(int(k))

    with open(os.path.join(PATH_R3Scan, "{}/semseg.v2.json".format(scan)), 'r') as f:
        seg_groups = json.load(f)["segGroups"]
        objects = {}  # object mapping to its belonging points
        obb = {}  # object bboxes in this scan split, size equals objects num
        labels = {}  # { id: 'category name', 6:'trash can'}
        seg2obj = {}  # mapping between segment and object id
        for o in seg_groups:
            id = o["id"]
            if id not in obj_id_list:  # no corresponding relationships in this split
                continue
            if o["label"] not in word2idx:  # Categories not under consideration
                continue
            labels[id] = o["label"]
            segs = o["segments"]
            objects[id] = []
            obb[id] = o["obb"]
            for i in segs:
                seg2obj[i] = id
                for j in segments[i]:
                    objects[id] = j.reshape(1, -1) if len(objects[id]) == 0 else np.concatenate(
                        (objects[id], j.reshape(1, -1)), axis=0)
    # sample and normalize point cloud
    obj_sample = 1024  # CONF.SCALAR.OBJ_PC_SAMPLE  # 1000
    for obj_id, obj_pc in objects.items():
        #if(len(obj_pc)<1024) : pc = obj_pc
        #else:
        #obj_pc = torch.Tensor(obj_pc).cuda()#,device ="cuda")
        #obj_pc = obj_pc.unsqueeze(0)
        pc = farthest_point_sample(obj_pc, obj_sample)  # obj_pc (16823,3) obj_sample 1000 --> pc (1000,3)
        #pc.numpy()
        #objects[obj_id] = pc_normalize(pc)
        objects[obj_id] = pc

    objects_id = []
    objects_cat = []
    objects_pc = []
    objects_num = []

    for k, v in objects.items():
        objects_id.append(k)
        objects_cat.append(word2idx[labels[k]])
        objects_num = objects_num + [len(v)]
        #objects_pc = v if not len(objects_pc) else np.concatenate((objects_pc, v), axis=0)
        v=v.tolist()
        objects_pc.append(v)

    # predicate input of PointNet, including points in the union bounding box of subject and object
    # here consider every possible combination between objects, if there doesn't exist relation in the training file,
    # add the relation with the predicate id replaced by 0
    triples = []
    pairs = []
    relationships_triples = relationships_scan["relationships"]
    for triple in relationships_triples:
        if (triple[0] not in objects_id) or (triple[1] not in objects_id) or (triple[0] == triple[1]):
            continue
        triples.append(triple[:3])
        if triple[:2] not in pairs:
            pairs.append(triple[:2])
    for i in objects_id:
        for j in objects_id:
            if i == j or [i, j] in pairs:
                continue
            triples.append([i, j, 0])  # supplement the 'none' relation
            pairs.append(([i, j]))

    s = 0
    o = 0
    try:
        # union_point_cloud = []
        predicate_cat = []
        predicate_num = []
        for rel in pairs:
            s, o = rel
            # union_pc = []
            pred_cls = np.zeros(len(rel2idx))
            for triple in triples:
                if rel == triple[:2]:
                    pred_cls[triple[2]] = 1
            predicate_cat.append(pred_cls.tolist())

    except KeyError:
        print(scan_id)
        print(obb.keys())
        print(s, o, '\n')
        return


    object_id2idx = {}  # convert object id to the index in the tensor
    for index, v in enumerate(objects_id):
        object_id2idx[v] = index
    s, o = np.split(np.array(pairs), 2, axis=1)  # All have shape (T, 1)
    s, o = [np.squeeze(x, axis=1) for x in [s, o]]  # Now have shape (T,)

    for index, v in enumerate(s):
        s[index] = object_id2idx[v]  # s_idx
    for index, v in enumerate(o):
        o[index] = object_id2idx[v]  # o_idx
    edges = np.stack((s, o), axis=1)  # edges is used for the input of the GCN module]

    gt_triples = []
    for triple in triples:
        if(triple[2]!=0):
            s_id, o_id = triple[:2]
            s_index = object_id2idx[s_id]
            o_index = object_id2idx[o_id]
            gt_triple = [s_index, o_index, triple[2]]
            gt_triples.append(gt_triple)

    # # since point cloud in 3DSGG has been processed, there is no need to sample any more => actually need
    # point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)

    data_dict = {}
    data_dict["scan_id"] = scan_id
    data_dict["objects_cat"] = objects_cat  # object category
    data_dict["objects_pc"] = objects_pc#.tolist()  # corresponding point cloud
    data_dict["gt_triples"] = gt_triples
    data_dict["objects_num"] = objects_num
    data_dict["predicate_cat"] = predicate_cat  # predicate id
    data_dict["objects_count"] = len(objects_cat)
    data_dict["predicate_count"] = len(predicate_cat)

    return data_dict


def process_onescan_gt(relationships_scan):
    scan = relationships_scan["scan"]
    scan_id = scan + "-" + str(hex(relationships_scan["split"]))[-1]
    print(scan_id)

    # load class and relationships dict
    word2idx = {}
    index = 0
    file = open(os.path.join(PATH_DATA, '3DSSG_subset/classes.txt'), 'r')
    category = file.readline()[:-1]
    while category:
        word2idx[category] = index
        category = file.readline()[:-1]
        index += 1

    rel2idx = {}
    index = 0
    file = open(os.path.join(PATH_DATA, "3DSSG_subset/relationships.txt"), 'r')
    category = file.readline()[:-1]
    while category:
        rel2idx[category] = index
        category = file.readline()[:-1]
        index += 1

    # read point cloud from OBJ file
    scan = scan_id[:-2]
    pc_array = read_obj(os.path.join(PATH_R3Scan, "{}/mesh.refined.obj".format(
        scan)))  # (100775,3) scan '752cc578-920c-26f5-8d8d-8a4239658074'
    # group points in the same segment
    segments = {}  # key:segment id, value: points belong to this segment
    with open(os.path.join(PATH_R3Scan, "{}/mesh.refined.0.010000.segs.json".format(scan)), 'r') as f:
        seg_indices = json.load(f)["segIndices"]  # (100775)
        for index, i in enumerate(seg_indices):
            if i not in segments:
                segments[i] = []
            segments[i].append(pc_array[index])

    # group points of the same object
    # filter the object which does not belong to this split
    obj_id_list = []
    for k, _ in relationships_scan["objects"].items():
        obj_id_list.append(int(k))

    with open(os.path.join(PATH_R3Scan, "{}/semseg.v2.json".format(scan)), 'r') as f:
        seg_groups = json.load(f)["segGroups"]
        objects = {}  # object mapping to its belonging points
        obb = {}  # object bboxes in this scan split, size equals objects num
        labels = {}  # { id: 'category name', 6:'trash can'}
        seg2obj = {}  # mapping between segment and object id
        for o in seg_groups:
            id = o["id"]
            if id not in obj_id_list:  # no corresponding relationships in this split
                continue  #[16, 1, 20, 6, 41, 25, 42, 11, 12]
            if o["label"] not in word2idx:  # Categories not under consideration
                continue
            labels[id] = o["label"]
            segs = o["segments"]
            objects[id] = []
            obb[id] = o["obb"]
            for i in segs:
                seg2obj[i] = id
                for j in segments[i]:
                    objects[id] = j.reshape(1, -1) if len(objects[id]) == 0 else np.concatenate(
                        (objects[id], j.reshape(1, -1)), axis=0)

    objects_id = []
    objects_cat = []
    objects_num = []

    for k, v in objects.items():
        objects_id.append(k)
        objects_cat.append(word2idx[labels[k]])
        objects_num = objects_num + [len(v)]

    # predicate input of PointNet, including points in the union bounding box of subject and object
    # here consider every possible combination between objects, if there doesn't exist relation in the training file,
    # add the relation with the predicate id replaced by 0
    triples = []
    pairs = []
    relationships_triples = relationships_scan["relationships"]
    for triple in relationships_triples:
        if (triple[0] not in objects_id) or (triple[1] not in objects_id) or (triple[0] == triple[1]):
            continue
        triples.append(triple[:3])
        if triple[:2] not in pairs:
            pairs.append(triple[:2])
    for i in objects_id:
        for j in objects_id:
            if i == j or [i, j] in pairs:
                continue
            triples.append([i, j, 0])  # supplement the 'none' relation
            pairs.append(([i, j]))

    s = 0
    o = 0
    try:
        # union_point_cloud = []
        predicate_cat = []
        predicate_num = []
        for rel in pairs:
            s, o = rel
            # union_pc = []
            pred_cls = np.zeros(len(rel2idx))
            for triple in triples:
                if rel == triple[:2]:
                    pred_cls[triple[2]] = 1
            predicate_cat.append(pred_cls.tolist())

    except KeyError:
        print(scan_id)
        print(obb.keys())
        print(s, o, '\n')
        return


    object_id2idx = {}  # convert object id to the index in the tensor
    for index, v in enumerate(objects_id):
        object_id2idx[v] = index
    s, o = np.split(np.array(pairs), 2, axis=1)  # All have shape (T, 1)
    s, o = [np.squeeze(x, axis=1) for x in [s, o]]  # Now have shape (T,)

    for index, v in enumerate(s):
        s[index] = object_id2idx[v]  # s_idx
    for index, v in enumerate(o):
        o[index] = object_id2idx[v]  # o_idx
    edges = np.stack((s, o), axis=1)  # edges is used for the input of the GCN module]

    gt_triples = []
    for triple in triples:
        if(triple[2]!=0):
            s_id, o_id = triple[:2]
            s_index = object_id2idx[s_id]
            o_index = object_id2idx[o_id]
            gt_triple = [s_index, o_index, triple[2]]
            gt_triples.append(gt_triple)

    # # since point cloud in 3DSGG has been processed, there is no need to sample any more => actually need
    # point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)

    data_dict = {}
    data_dict["scan_id"] = scan_id
    data_dict["objects_cat"] = objects_cat  # object category
    #data_dict["objects_pc"] = objects_pc#.tolist()  # corresponding point cloud
    data_dict["gt_triples"] = gt_triples
    data_dict["objects_num"] = objects_num
    data_dict["predicate_cat"] = predicate_cat  # predicate id
    data_dict["objects_count"] = len(objects_cat)
    data_dict["predicate_count"] = len(predicate_cat)

    return data_dict



def write_into_json(relationship, dataset_type):
    only_gt= False
    if(only_gt):
        data_dict = process_onescan_gt(relationship)
    else:
        data_dict = process_onescan(relationship)
    if data_dict is None:
        return

    scan_id = data_dict["scan_id"]
    scan_name = scan_id[:-2]
    scan_folder = os.path.join(PATH_DATA, "3DSSG_subset", dataset_type, scan_name)
    if not os.path.exists(scan_folder):
        os.mkdir(scan_folder)
    scan_list_path = os.path.join(PATH_DATA, "3DSSG_subset", dataset_type, "{}ing_txt.txt".format(dataset_type))


    datapath = scan_folder + '/' + 'data_{}'.format(scan_id[-1])
    if not os.path.exists(datapath):
        os.mkdir(datapath)
    with open(scan_list_path, 'a') as f:
        f.write(datapath+"\n")

    gt_obj_path = os.path.join(datapath, "gt_obj.npy")
    gt_rel_path = os.path.join(datapath, "gt_relationships.npy")
    pointcloud_ins_path = os.path.join(datapath, "pointcloud_1024_ins.npy")

    np_gt_obj = data_dict["objects_cat"]
    np_gt_rel = data_dict["gt_triples"]
    np_pc = data_dict["objects_pc"]

    len_shape_of_rel = len(np.array(np_gt_rel).shape)
    size_of_rel = np.array(np_gt_rel).size
    # print(shape_of_rel) # np_gt_rel.shape (25, 3)
    if (size_of_rel == 0):
        print("data_", scan_id[-1], " shoud skip")
        return
    if (len_shape_of_rel < 2):
        print(np_gt_rel)
        np_rel_tmp = np_gt_rel[None]
        np_gt_rel = np_rel_tmp
        print(np_gt_rel)
    np.save(gt_obj_path, np_gt_obj)
    np.save(gt_rel_path, np_gt_rel)
    np.save(pointcloud_ins_path, np_pc)


def process_dataset(relationships, dataset_type):

    for scan_i in relationships:
        write_into_json(scan_i, dataset_type)
        print("------Processing ", scan_i["scan"], " --------------")
    print("Done!")

def check_process_dataset(relationships):

    for scan_i in relationships:
        if(scan_i["scan"]=="422885e0-192d-25fc-844a-62e395291839"):
            process_onescan_gt(scan_i)
            print("------Processing ", scan_i["scan"], " --------------")
    print("Done!")


if __name__ == '__main__':
    relationships_train = json.load(open(os.path.join(PATH_DATA, "3DSSG_subset/relationships_train.json")))["scans"]
    relationships_val = json.load(open(os.path.join(PATH_DATA, "3DSSG_subset/relationships_validation.json")))["scans"]
    process_dataset(relationships_train, dataset_type='train')
    process_dataset(relationships_val, dataset_type='test')
    #check_process_dataset(relationships_train)


