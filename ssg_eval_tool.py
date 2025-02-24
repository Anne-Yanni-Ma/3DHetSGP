import numpy as np
import torch.nn.functional as F
import torch
import pathmagic  # noqa
from functools import reduce

support_label = [1,14,15,16,17,18,19,20,21,22,23,24,25,26]  # len: 14
suppport_map = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
proximity_label = [2,3,4,5,6,7] # 不除去inside  6个
proximity_map = [1,2,3,4,5,6]
comparative_label = [8,9,10,11,12,13] # 6个
comparative_map = [1,2,3,4,5,6]
cp_label = [2,3,4,5,6,7,8,9,10,11,12,13]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2021)


with open('./data/classes.txt', 'r') as f:
    classes = f.readlines()
    for i in range(len(classes)):
        classes[i] = classes[i].strip()
f.close()


with open('./data/relationships-know3d-ori.txt', 'r') as f:
    relationships = f.readlines()
    for i in range(len(relationships)):
        relationships[i] = relationships[i].strip()
f.close()


class Object_Accuracy():
    def __init__(self, len_dataset, show_acc_category=False, need_softmax=True):
        self.len_dataset = len_dataset
        self.show_acc_category = show_acc_category
        self.need_softmax = need_softmax

        self.acc_overall = 0.0
        self.acc_mean = 0.0
        self.acc_category = np.zeros(160)

        self.count_all = 0.0
        self.count_category = np.zeros(160)

    def calculate_accuray(self, obj_output, gt_obj):
        gt_obj = gt_obj.cpu().numpy()
        if self.need_softmax:
            obj_scores = F.softmax(obj_output, dim=1).cpu().numpy()
        else:
            obj_scores = obj_output.cpu().numpy()
        obj_output = obj_scores.argmax(1)
        correct = np.sum(obj_output == gt_obj)

        self.count_all += gt_obj.shape[0]
        self.acc_overall += correct
        for i in range(obj_output.shape[0]):
            if obj_output[i] == gt_obj[i]:
                self.acc_category[int(obj_output[i])] += 1
            self.count_category[int(gt_obj[i])] += 1

    def final_update(self):
        self.acc_overall /= self.count_all
        for i in range(160):
            if self.count_category[i] == 0:
                self.count_category[i] = 99999
        self.acc_category /= self.count_category
        self.acc_mean = self.acc_category.sum() / 160

    def print_string(self):
        if self.show_acc_category:
            res = ''
            res += ('Object Acc_o: %f; ' % self.acc_overall) + ('Object Acc_m: %f; ' % self.acc_mean)
            res += '\n'
            for i in range(self.acc_category.shape[0]):
                res += ('%s: ' % classes[i]) + ('%f\n' % self.acc_category[i])
            return res
        else:
            return ('Object Acc_o: %f; ' % self.acc_overall) + ('Object Acc_m: %f; ' % self.acc_mean)

    def reset(self):
        self.__init__()


class Object_Recall():
    def __init__(self, len_dataset, need_softmax=True):
        self.recall = {1: 0, 5: 0, 10: 0}
        self.topk_obj = []
        self.need_softmax = need_softmax

    def calculate_recall(self, obj_output, obj_gt):
        if self.need_softmax:
            obj_output = F.softmax(obj_output, dim=1)
        topk = torch.topk(obj_output, k=10, dim=1).indices
        for i in range(obj_gt.shape[0]):
            flag = False
            for j in range(10):
                if obj_gt[i] == topk[i, j]:
                    self.topk_obj.append(j+1)
                    flag = True
                    break
            if flag is False:
                self.topk_obj.append(10+1)

    def final_update(self):
        ntopk_obj = np.asarray(self.topk_obj)
        len_ntop_obj = len(ntopk_obj)
        self.recall[1] = (ntopk_obj <= 1).sum() / len_ntop_obj
        self.recall[5] = (ntopk_obj <= 5).sum() / len_ntop_obj
        self.recall[10] = (ntopk_obj <= 10).sum() / len_ntop_obj

    def print_string(self):
        return ('Object R@1: %f; ' % self.recall[1]) + ('Object R@5: %f; ' % self.recall[5]) + ('Object R@10: %f; ' % self.recall[10])

    def reset(self):
        self.__init__()


class Heter_Predicate_Recall():
    def __init__(self, len_dataset, need_softmax=True):
        self.recall = {1: 0, 3: 0, 5: 0}
        self.recall_s = {1: 0, 3: 0, 5: 0}
        self.recall_p = {1: 0, 3: 0, 5: 0}
        self.recall_c = {1: 0, 3: 0, 5: 0}

        self.topk_pred = []
        self.topk_pred_s = []
        self.topk_pred_p = []
        self.topk_pred_c = []
        self.need_softmax = need_softmax
        self.recall_category = np.zeros(26)
        self.recall_category_s = np.zeros(14)
        self.recall_category_p = np.zeros(6)
        self.recall_category_c = np.zeros(6)

        self.recall_count = np.zeros(26) + 1e-6
        self.recall_count_s = np.zeros(14) + 1e-6
        self.recall_count_p = np.zeros(6) + 1e-6
        self.recall_count_c = np.zeros(6) + 1e-6

    def calculate_Separate_recall(self, insnum, edge_output,gt_rel):
        # 计算 Proximity Comparative type的recall
        pred_output_s = torch.zeros((insnum * insnum - insnum, 15)).cuda()
        pred_output_p = torch.zeros((insnum * insnum - insnum, 7)).cuda()
        pred_output_c = torch.zeros((insnum * insnum - insnum, 7)).cuda()

        pred_output_s[:,1] =  edge_output[:,1]
        pred_output_s[:, 2:15] = edge_output[:, 14:27]
        pred_output_p[:,1:7], pred_output_c[:,1:7] =  edge_output[:,2:8], edge_output[:,8:14]


        if self.need_softmax:
            pred_output_s = F.softmax(pred_output_s, dim=1)
            pred_output_p = F.softmax(pred_output_p, dim=1)
            pred_output_c = F.softmax(pred_output_c, dim=1)
        topk_s = torch.topk(pred_output_s[:,1:], k=5, dim=1).indices
        topk_p = torch.topk(pred_output_p[:,1:], k=5, dim=1).indices
        topk_c = torch.topk(pred_output_c[:,1:], k=5, dim=1).indices

        # Support Recall
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            predgt = int(gt_rel[i, 2])

            if predgt not in support_label:
                continue

            predgt_s = support_label.index(predgt)
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum - 1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum - 1) + idx_j
            flag = False
            self.recall_count_s[predgt_s] += 1
            for j in range(5):
                if predgt_s == topk_s[idx, j]:
                    self.recall_category_s[predgt_s] += 1
                    self.topk_pred_s.append(j + 1)
                    flag = True
                    break
            if flag is False:
                self.topk_pred_s.append(5 + 1)

        # Proximity Recall
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            predgt =int(gt_rel[i, 2])

            if predgt not in proximity_label:
                continue

            predgt_p = proximity_label.index(predgt)
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum-1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum-1) + idx_j
            flag = False
            self.recall_count_p[predgt_p] += 1
            for j in range(5):
                if predgt_p == topk_p[idx, j]:
                    self.recall_category_p[predgt_p] += 1
                    self.topk_pred_p.append(j+1)
                    flag = True
                    break
            if flag is False:
                self.topk_pred_p.append(5+1)

        ## Comparative Recall
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            predgt = int(gt_rel[i, 2])
            if predgt not in comparative_label:
                continue

            predgt_c = comparative_label.index(predgt)
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum - 1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum - 1) + idx_j
            flagg = False
            self.recall_count_c[predgt_c] += 1
            for j in range(5):
                if predgt_c == topk_c[idx, j]:
                    self.recall_category_c[predgt_c] += 1
                    self.topk_pred_c.append(j + 1)
                    flagg = True
                    break
            if flagg is False:
                self.topk_pred_c.append(5 + 1)

        return self.topk_pred_s, self.topk_pred_p, self.topk_pred_c

    def final_update(self):
        ntopk_pred = np.asarray(self.topk_pred)
        len_ntop_pred = len(ntopk_pred)
        self.recall[1] = (ntopk_pred <= 1).sum() / len_ntop_pred
        self.recall[3] = (ntopk_pred <= 3).sum() / len_ntop_pred
        self.recall[5] = (ntopk_pred <= 5).sum() / len_ntop_pred
        self.recall_category = self.recall_category / self.recall_count

        ntopk_pred_s = np.asarray(self.topk_pred_s)
        len_ntop_pred_s = len(ntopk_pred_s)
        self.recall_s[1] = (ntopk_pred_s <= 1).sum() / len_ntop_pred_s
        self.recall_s[3] = (ntopk_pred_s <= 3).sum() / len_ntop_pred_s
        self.recall_s[5] = (ntopk_pred_s <= 5).sum() / len_ntop_pred_s
        self.recall_category_s = self.recall_category_s / self.recall_count_s

        ntopk_pred_p = np.asarray(self.topk_pred_p)
        len_ntop_pred_p = len(ntopk_pred_p)
        self.recall_p[1] = (ntopk_pred_p <= 1).sum() / len_ntop_pred_p
        self.recall_p[3] = (ntopk_pred_p <= 3).sum() / len_ntop_pred_p
        self.recall_p[5] = (ntopk_pred_p <= 5).sum() / len_ntop_pred_p
        self.recall_category_p = self.recall_category_p / self.recall_count_p

        ntopk_pred_c = np.asarray(self.topk_pred_c)
        len_ntop_pred_c = len(ntopk_pred_c)
        self.recall_c[1] = (ntopk_pred_c <= 1).sum() / len_ntop_pred_c
        self.recall_c[3] = (ntopk_pred_c <= 3).sum() / len_ntop_pred_c
        self.recall_c[5] = (ntopk_pred_c <= 5).sum() / len_ntop_pred_c
        self.recall_category_c = self.recall_category_c / self.recall_count_c

    def print_string(self):
        res = ('Predicate R@1: %f; ' % self.recall[1]) + ('Predicate R@3: %f; ' % self.recall[3]) + ('Predicate R@5: %f; ' % self.recall[5]) + '\n'
        res += ('S_Predicate R@1: %f; ' % self.recall_s[1]) + ('S_Predicate R@3: %f; ' % self.recall_s[3]) + ('S_Predicate R@5: %f; ' % self.recall_s[5])+ '\n'
        res += ('P_Predicate R@1: %f; ' % self.recall_p[1]) + ('P_Predicate R@3: %f; ' % self.recall_p[3]) + ('P_Predicate R@5: %f; ' % self.recall_p[5])+ '\n'
        res += ('C_Predicate R@1: %f; ' % self.recall_c[1]) + ('C_Predicate R@3: %f; ' % self.recall_c[3]) + ('C_Predicate R@5: %f; ' % self.recall_c[5])+ '\n'

        return res

    def reset(self):
        self.__init__()

class Predicate_Recall():
    def __init__(self, len_dataset, need_softmax=True):
        self.recall = {1: 0, 3: 0, 5: 0}
        self.topk_pred = []
        self.need_softmax = need_softmax
        self.recall_category = np.zeros(26)
        self.recall_count = np.zeros(26) + 1e-6

    def calculate_recall(self, insnum, pred_output, gt_rel):
        if self.need_softmax:
            pred_output = F.softmax(pred_output, dim=1)
        topk = torch.topk(pred_output, k=5, dim=1).indices
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum-1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum-1) + idx_j
            flag = False
            self.recall_count[int(gt_rel[i, 2])-1] += 1
            for j in range(5):
                if gt_rel[i, 2] == topk[idx, j]:
                    self.recall_category[int(gt_rel[i, 2])-1] += 1
                    self.topk_pred.append(j+1)
                    flag = True
                    break
            if flag is False:
                self.topk_pred.append(5+1)
        return self.topk_pred

    def final_update(self):
        ntopk_pred = np.asarray(self.topk_pred)
        len_ntop_pred = len(ntopk_pred)
        self.recall[1] = (ntopk_pred <= 1).sum() / len_ntop_pred
        self.recall[3] = (ntopk_pred <= 3).sum() / len_ntop_pred
        self.recall[5] = (ntopk_pred <= 5).sum() / len_ntop_pred
        self.recall_category = self.recall_category / self.recall_count

    def print_string(self):
        return ('Predicate R@1: %f; ' % self.recall[1]) + ('Predicate R@3: %f; ' % self.recall[3]) + ('Predicate R@5: %f; ' % self.recall[5])

    def reset(self):
        self.__init__()


class Predicate_Recall_Onehot():
    def __init__(self, len_dataset, need_softmax=True):
        self.recall = {1: 0, 3: 0, 5: 0}
        self.topk_pred = []
        self.need_softmax = need_softmax
        self.recall_category = np.zeros(26)
        self.recall_count = np.zeros(26) + 1e-6

    def calculate_recall(self, insnum, pred_output, gt_rel):
        if self.need_softmax:
            pred_output = F.softmax(pred_output, dim=1)
        topk = torch.topk(pred_output, k=5, dim=1).indices
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum-1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum-1) + idx_j
            flag = False
            self.recall_count[int(gt_rel[i, 2])-1] += 1
            for j in range(5):
                if gt_rel[i, 2] == topk[idx, j]:
                    self.topk_pred.append(j+1)
                    self.recall_category[int(gt_rel[i, 2])-1] += 1
                    flag = True
                    break
            if flag is False:
                self.topk_pred.append(5+1)
        return self.topk_pred

    def final_update(self):
        ntopk_pred = np.asarray(self.topk_pred)
        len_ntop_pred = len(ntopk_pred)
        self.recall[1] = (ntopk_pred <= 1).sum() / len_ntop_pred
        self.recall[3] = (ntopk_pred <= 3).sum() / len_ntop_pred
        self.recall[5] = (ntopk_pred <= 5).sum() / len_ntop_pred
        self.recall_category = self.recall_category / self.recall_count

    def print_string(self):
        return ('Predicate R@1: %f; ' % self.recall[1]) + ('Predicate R@3: %f; ' % self.recall[3]) + ('Predicate R@5: %f; ' % self.recall[5])

    def reset(self):
        self.__init__()


class Predicate_Accuracy():
    def __init__(self, len_dataset, need_softmax=True):
        self.len_dataset = len_dataset
        self.binary_acc = 0
        self.binary_recall = 0
        self.acc = 0
        self.acc_mean = 0.0
        self.acc_category = np.zeros(27)
        self.count_category = np.zeros(27)
        self.need_softmax = need_softmax
        self.Rel_result =[]

    def calculate_accuracy_binary(self, insnum, pred_output, gt_rel):
        if self.need_softmax:
            pred_output = F.softmax(pred_output, dim=1)
        pred_label = torch.argmax(pred_output, dim=1)  # for example tensor([ 0,  0,  0, 16, 20, 16,  0,  0,  0,  0,  0,  0], device='cuda:0')
        gt = torch.zeros(pred_label.shape[0])
        correct_count = 0
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum-1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum-1) + idx_j
            gt[idx] = 1
        for i in range(gt.shape[0]):
            if pred_label[i] == 0 and gt[i] == 0:
                correct_count += 1
            elif pred_label[i] != 0 and gt[i] == 1:
                correct_count += 1
        self.binary_acc = self.binary_acc + (correct_count / pred_label.shape[0])

    def calculate_recall_binary(self, insnum, pred_output, gt_rel):
        if self.need_softmax:
            pred_output = F.softmax(pred_output, dim=1)
        pred_label = torch.argmax(pred_output, dim=1)
        gt = torch.zeros(pred_label.shape[0])
        related_count = 0
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum-1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum-1) + idx_j
            gt[idx] = 1
        related_total = (gt == 1).sum().float().data
        if related_total == 0:
            self.binary_recall = self.binary_recall + 0
        else:
            for i in range(gt.shape[0]):
                if pred_label[i] != 0 and gt[i] == 1:
                    related_count += 1
            self.binary_recall = self.binary_recall + (related_count / related_total)

    def calculate_accuracy(self, insnum, pred_output, gt_rel):
        pred_output = F.softmax(pred_output, dim=1)
        pred_label = torch.argmax(pred_output, dim=1)
        onehot = torch.zeros((insnum * insnum - insnum, 27)).cuda()
        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]  #obj
            idx_j = gt_rel[i, 1]  #suj
            if idx_i < idx_j:
                onehot[int(idx_i * (insnum-1) + idx_j - 1), int(gt_rel[i, 2])] = 1
            elif idx_i > idx_j:
                onehot[int(idx_i * (insnum-1) + idx_j), int(gt_rel[i, 2])] = 1
            self.count_category[int(gt_rel[i, 2])] += 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot[i, :]) == 0:
                onehot[i, 0] = 1
                self.count_category[0] += 1  # 无边的关系加到 none类别  注意3dssg的none类别不在第0位
        count = 0
        for i in range(pred_label.shape[0]):
            idx = pred_label[i]
            if onehot[i, idx] == 1:
                count += 1
                self.acc_category[idx] += 1
                if(idx!=0):
                    #print("第-",i,"/", insnum, "-条边正确")
                    subj = i//(insnum-1)
                    x = i % (insnum-1)
                    if(subj<x+1): obj=x+1
                    elif(subj>x): obj=x
                    #print("subj:", subj, "obj:",obj, "predicate：",int(idx))
                    #print(classes[subj] + '->' + classes[obj] + '=' + relationships[int(idx)])
                    rel_pred = classes[subj], classes[obj] ,relationships[int(idx)]
                    if(rel_pred not in self.Rel_result):
                        self.Rel_result.append(rel_pred)


        self.acc = self.acc + (count / pred_label.shape[0])

    def final_update(self):
        self.binary_acc = self.binary_acc / self.len_dataset
        self.binary_recall = self.binary_recall / self.len_dataset
        self.acc = self.acc / self.len_dataset
        for i in range(27):
            if self.count_category[i] == 0:
                self.count_category[i] = 1
        print(self.acc_category)
        self.acc_category = self.acc_category / self.count_category
        self.acc_mean = self.acc_category.mean()
        Rel_result = self.Rel_result
        '''for rel in Rel_result:
            subj,obj,pred = rel
            print(subj + '->' + obj + '=' + pred)'''
        print("########### sum of correct rel:", len(Rel_result))

    def print_string(self):
        res = 'Predicate Binary Acc: %f; ' % self.binary_acc + 'Relation Binary Recall: %f; ' % self.binary_recall + 'Relation Acc: %f; ' % self.acc + '\n'
        res += f'Predicate mean acc: {self.acc_mean}'
        return res

    def reset(self):
        self.__init__()


class Relation_Recall():
    def __init__(self, len_dataset, need_softmax=True):
        self.recall = {20: 0, 50: 0, 100: 0}
        self.ngc_recall = {20: 0, 50: 0, 100: 0}
        self.m_recall = {20: 0, 50: 0, 100: 0}
        self.m_recall_cat = np.zeros((4, 26))
        self.len_dataset = len_dataset
        self.need_softmax = need_softmax
        self.Rel_acc = []

    def calculate_recall(self, obj_output, pred_output, gt_obj, gt_rel):
        obj_num = gt_obj.shape[0]
        # prepare predicted relation triplet
        if self.need_softmax:
            obj_scores = F.softmax(obj_output, dim=1).cpu().numpy()
            pred_scores = F.softmax(pred_output, dim=1).cpu().numpy()
        else:
            obj_scores = obj_output.cpu().numpy()
            pred_scores = pred_output.cpu().numpy()
        obj_inds = np.array([[i, j] for i in range(obj_num) for j in range(obj_num) if i != j])
        obj_score = obj_scores.max(1)
        pred_score = pred_scores[:, 1:].max(1)
        obj_output = obj_scores.argmax(1)
        pred_output = pred_scores[:, 1:].argmax(1) + 1
        triplet, triplet_score = _triplet(obj_inds, obj_output, pred_output, obj_score, pred_score)

        sorted_triplet = triplet[np.argsort(-triplet_score)]
        # prepare gt relation triplet
        if gt_rel.shape[0] == 0:
            for k in self.ngc_recall:
                self.recall[k] += 0
            return
        gt_obj_inds = gt_rel[:, :2].cpu().numpy()
        gt_obj = gt_obj.cpu().numpy()
        gt_pred = gt_rel[:, 2].cpu().numpy()
        gt_triplet = _triplet(gt_obj_inds, gt_obj, gt_pred, None, None)

        # calculate matches
        pred_to_gt = _compute_pred_matches(gt_triplet, sorted_triplet)

        for i in range(gt_triplet.shape[0]):
            self.m_recall_cat[0, gt_triplet[i, 1]-1] += 1
        # calculate recalls
        value = 0
        for k in self.recall:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k]).astype('int16')
            rec_i = float(len(match)) / float(gt_rel.shape[0])
            acc_rel = rec_i
            self.recall[k] += rec_i
            if k == 50:
                value = rec_i
            if k == 20:
                for i in range(match.shape[0]):
                    self.m_recall_cat[1, gt_triplet[match[i], 1]-1] += 1
            if k == 50:
                for i in range(match.shape[0]):
                    self.m_recall_cat[2, gt_triplet[match[i], 1]-1] += 1
            if k == 100:
                for i in range(match.shape[0]):
                    self.m_recall_cat[3, gt_triplet[match[i], 1]-1] += 1
        #print("与边匹配准确度：　", acc_rel)
        #self.Rel_acc.append(round(acc_rel,2))

    def calculate_ngc_recall(self, obj_output, pred_output, gt_obj, gt_rel):
        obj_num = gt_obj.shape[0]
        # prepare predicted relation triplet
        if self.need_softmax:
            obj_scores = F.softmax(obj_output, dim=1).cpu().numpy()
        else:
            obj_scores = obj_output.cpu().numpy()
        obj_inds = np.array([[i, j] for i in range(obj_num) for j in range(obj_num) if i != j])
        obj_score = obj_scores.max(1)
        obj_output = obj_scores.argmax(1)
        if self.need_softmax:
            pred_scores = F.softmax(pred_output, dim=1).cpu().numpy()
        else:
            pred_scores = pred_output.cpu().numpy()

        obj_scores_per_rel = obj_score[obj_inds].prod(1)
        ngc_overall_scores = obj_scores_per_rel[:, None] * pred_scores[:, 1:]
        ngc_overall_scores_1d = ngc_overall_scores.reshape(1, -1)
        ngc_score_inds = np.argsort(-ngc_overall_scores_1d)[0][:100]
        ngc_score_inds = np.unravel_index(ngc_score_inds, pred_scores[:, 1:].shape)
        ngc_score_inds = np.column_stack((ngc_score_inds[0], ngc_score_inds[1]))
        ngc_obj_inds = obj_inds[ngc_score_inds[:, 0]]
        ngc_pred_scores = pred_scores[ngc_score_inds[:, 0], ngc_score_inds[:, 1]+1]
        triplet, triplet_score = _triplet(ngc_obj_inds, obj_output, ngc_score_inds[:, 1]+1, obj_score, ngc_pred_scores)
        # prepare gt relation triplet
        if gt_rel.shape[0] == 0:
            for k in self.ngc_recall:
                self.ngc_recall[k] += 0
            return
        gt_obj_inds = gt_rel[:, :2].cpu().numpy()
        gt_obj = gt_obj.cpu().numpy()
        gt_pred = gt_rel[:, 2].cpu().numpy()
        gt_triplet = _triplet(gt_obj_inds, gt_obj, gt_pred, None, None)

        # calculate matches
        pred_to_gt = _compute_pred_matches(gt_triplet, triplet)

        # calculate recalls
        for k in self.ngc_recall:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rel.shape[0])
            self.ngc_recall[k] += rec_i
        return self.ngc_recall

    def final_update(self):
        for k in self.recall:
            self.recall[k] /= self.len_dataset
            self.ngc_recall[k] /= self.len_dataset
            for i in range(26):
                if self.m_recall_cat[0, i] != 0:
                    if k == 20:
                        self.m_recall_cat[1, i] /= self.m_recall_cat[0, i]
                    if k == 50:
                        self.m_recall_cat[2, i] /= self.m_recall_cat[0, i]
                    if k == 100:
                        self.m_recall_cat[3, i] /= self.m_recall_cat[0, i]
            if k == 20:
                self.m_recall[k] = self.m_recall_cat[1].mean()
            if k == 50:
                self.m_recall[k] = self.m_recall_cat[2].mean()
            if k == 100:
                self.m_recall[k] = self.m_recall_cat[3].mean()

    def print_string(self):
        recall_res = 'Rel R@20: %f; ' % (self.recall[20]) + 'Rel R@50: %f; ' % (self.recall[50]) + 'Rel R@100: %f;' % (self.recall[100]) +'\n'
        recall_res_ngc = 'Rel R@20: %f; ' % (self.ngc_recall[20]) + 'Rel R@50: %f; ' % (self.ngc_recall[50]) + 'Rel R@100: %f;' % (self.ngc_recall[100])
        recall_res_m = 'Rel R@20: %f; ' % (self.m_recall[20]) + 'Rel R@50: %f; ' % (self.m_recall[50]) + 'Rel R@100: %f;' % (self.m_recall[100])
        return recall_res, recall_res_ngc, recall_res_m

    def reset(self):
        self.__init__()


def _triplet(obj_inds, obj_output, pred_output, obj_score, pred_score):
    s = obj_output[obj_inds[:, 0]]
    o = obj_output[obj_inds[:, 1]]
    if obj_score is not None and pred_score is not None:
        s_score = obj_score[obj_inds[:, 0]]
        o_score = obj_score[obj_inds[:, 1]]
        return np.column_stack((s, pred_output, o)), s_score*pred_score*o_score
    else:
        return np.column_stack((s, pred_output, o))


def _compute_pred_matches(gt_triplets, pred_triplets):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_triplets.shape[0])]
    for gt_ind, keep_inds in zip(np.where(gt_has_match)[0], keeps[gt_has_match]):
        for i in np.where(keep_inds)[0]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res
