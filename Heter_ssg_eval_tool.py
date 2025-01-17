import numpy as np
import torch.nn.functional as F
import torch
from functools import reduce

support_label = [1,14,15,16,17,18,19,20,21,22,23,24,25,26]  # len: 14
support_label_np = np.array([1,14,15,16,17,18,19,20,21,22,23,24,25,26])  # len: 14
support_map = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
proximity_label = [2,3,4,5,6,7] # 不除去inside  6个
proximity_label_np = np.array([2,3,4,5,6,7])
proximity_map = [1,2,3,4,5,6]
comparative_label = [8,9,10,11,12,13] # 6个
comparative_label_np = np.array([8,9,10,11,12,13]) # 6个
comparative_map = [1,2,3,4,5,6]
cp_label = [2,3,4,5,6,7,8,9,10,11,12,13]
cp_label_np = np.array([2,3,4,5,6,7,8,9,10,11,12,13])



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


with open('./data/relationships.txt', 'r') as f:
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

class Heter_Predicate_Accuracy():
    def __init__(self,len_dataset, need_softmax=True):
        self.len_dataset = len_dataset
        self.binary_acc = 0
        self.binary_recall = 0
        self.acc = 0
        self.acc_s = 0
        self.acc_p = 0
        self.acc_c = 0
        self.lendata_s = 0

        self.lendata_c =0
        self.lendata_p = 0


        self.acc_category = np.zeros(27)
        self.acc_category_1 = np.zeros(27)
        self.acc_category_type_1 = np.zeros(3)


        self.count_category = np.zeros(27)
        self.count_category_1 = np.zeros(27)
        self.count_category_type_1 = np.zeros(3)


        self.type_acc_1 = [0, 0]
        self.need_softmax = need_softmax

        self.avg_support_ratio = 0
        self.avg_proximity_ratio = 0
        self.avg_comparative_ratio = 0

    def calculate_Heter_accuracy(self, insnum, heter_edge_output, gt_rel):
        # 计算单type的准确度 正确类别的个数/type下类别数
        pred_output_s, pred_output_p, pred_output_c =  heter_edge_output
        #pred_output_l, pred_output_s, pred_output_p, pred_output_c = pred_output_l[:,1:], pred_output_s[:,1:], pred_output_p[:,1:], pred_output_c[:,1:]

        pred_output_s = F.softmax(pred_output_s, dim=1)
        pred_label_s = torch.argmax(pred_output_s, dim=1)

        pred_output_p = F.softmax(pred_output_p, dim=1)
        #torch.sigmoid(logit_p)
        pred_label_p = torch.argmax(pred_output_p, dim=1)

        # train得到的comparative output
        pred_output_c =  F.softmax(pred_output_c, dim=1)
        pred_label_c = torch.argmax(pred_output_c, dim=1)

        num_gt_s=0
        num_gt_p=0
        num_gt_c=0

        onehot = torch.zeros((insnum * insnum - insnum, 27)).cuda()
        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]  # obj
            idx_j = gt_rel[i, 1]  # suj
            predgt =int(gt_rel[i, 2])
            if predgt in support_label:
                num_gt_s +=1
            if predgt in proximity_label:
                num_gt_p +=1
            if predgt in comparative_label:
                num_gt_c +=1
            if idx_i < idx_j:
                onehot[int(idx_i * (insnum - 1) + idx_j - 1), int(gt_rel[i, 2])] = 1
            elif idx_i > idx_j:
                onehot[int(idx_i * (insnum - 1) + idx_j), int(gt_rel[i, 2])] = 1
            self.count_category[int(gt_rel[i, 2])] += 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot[i, :]) == 0:
                onehot[i, 0] = 1
                self.count_category[0] += 1  # 无边的关系加到 none类别  注意3dssg的none类别不在第0位

        count_s = 0
        count_p = 0
        count_c = 0
        # support acc
        for i in range(pred_label_s.shape[0]):
            idx = support_label[pred_label_s[i]-1]  # pred_gt_map = support_label.index(pred_gt)+1
            if onehot[i, idx] == 1:
                count_s += 1
                self.acc_category[idx] += 1

        for i in range(pred_label_p.shape[0]):
            idx = proximity_label[pred_label_p[i]-1]
            if onehot[i, idx] == 1:
                count_p += 1
                self.acc_category[idx] += 1

        for i in range(pred_label_c.shape[0]):
            idx = comparative_label[pred_label_c[i]-1]
            if onehot[i, idx] == 1:
                count_c += 1
                self.acc_category[idx] += 1

        if(num_gt_s!=0):
            self.acc_s = self.acc_s + (count_s / num_gt_s)
            self.lendata_s+= 1
        if(num_gt_p!=0):
            self.acc_p = self.acc_p + (count_p / num_gt_p)
            self.lendata_p += 1
        if (num_gt_c != 0):
            self.acc_c = self.acc_c + (int(count_c) / num_gt_c)
            self.lendata_c +=1
        #print("counts: ",count_s, "countp: ",count_p,"countc: ",count_c)
        #print("acc_s: ", self.acc_s, "acc_p: ", self.acc_p,"acc_c: ", self.acc_c)
        #print("num_gt_s: ", num_gt_s, "num_gt_p: ", num_gt_p,"num_gt_c: ",num_gt_c)


    def final_update(self):
        self.acc = self.acc / self.len_dataset
        self.acc_s = self.acc_s / self.lendata_s
        self.acc_p = self.acc_p / self.lendata_p
        self.acc_c = self.acc_c / self.lendata_c

        self.acc_category_avg = self.acc_category / self.count_category
        self.acc_mean = self.acc_category.mean()

    def print_string(self):
        acc_cat_ratio = self.acc_category_avg #27
        sum_support_ratio = acc_cat_ratio[1] # supported_by
        sum_proximity_ratio = 0
        sum_comparative_ratio = 0

        for i in range(14,27): #(14,27)
            sum_support_ratio += acc_cat_ratio[i]

        for i in range(2,7): #(2,8)
            sum_proximity_ratio += acc_cat_ratio[i]

        for i in range(8,14): #(8,14)
            sum_comparative_ratio += acc_cat_ratio[i]

        self.avg_support_ratio = sum_support_ratio/14
        self.avg_proximity_ratio = sum_proximity_ratio/6
        self.avg_comparative_ratio = sum_comparative_ratio/6

        res = ' Relation:    Acc_support: %f; ' % self.acc_s +  ' Acc_proximity: %f; ' % self.acc_p + ' Acc_comparative: %f; ' % self.acc_c +  '\n'
        #res += f' 27 Predicate mean acc: {self.acc_mean}'
        #res += f' 26 Predicate per acc: {self.acc_category_avg[1:]}' +  '\n'
        res += f' 26 Predicate pred count: { self.acc_category[1:]}'+  '\n'
        #res += f' 26 Predicate gt count: { self.count_category[1:]}' +  '\n'
        res += f' 3 types Acc mean:  Support: {self.avg_support_ratio}; '  + 'Proximity:  %f ;   ' %(self.avg_proximity_ratio ) + 'Comparative: %f ;  ' %(self.avg_comparative_ratio)

        return res

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

    def calculate_Heter_recall(self, insnum, heter_edge_output, gt_rel):
        # 计算multi_pred_outputs的recall

        pred_output_s, pred_output_p, pred_output_c =  heter_edge_output
        # (Ne,2)     (Ne,15) 1,14~26   (Ne,7) 2~7    (Ne,7) 8~13
        pred_output = torch.zeros((insnum * insnum - insnum, 27)).cuda()

        pred_output[:,1] = pred_output_s[:,1]
        pred_output[:, 14:27] = pred_output_s[:,2:15]
        pred_output[:,2:8] = pred_output_p[:,1:7]
        pred_output[:,8:14] = pred_output_c[:,1:7]
        #pred_output = F.softmax(pred_output, dim=1)

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

    def calculate_Heter_recall_w(self, insnum, heter_edge_output, gt_rel):
        # 计算multi_pred_outputs的recall

        pred_output_s, pred_output_p, pred_output_c =  heter_edge_output
        # (Ne,2)     (Ne,15) 1,14~26   (Ne,7) 2~7    (Ne,7) 8~13
        pred_output = torch.zeros((insnum * insnum - insnum, 27)).cuda()
        link_w, support_w, p_w, comp_w = types_w
        '''pred_output_s = F.softmax(pred_output_s, dim=1)
        pred_output_p = F.softmax(pred_output_p, dim=1)
        pred_output_c = F.softmax(pred_output_c, dim=1)
        '''
        pred_output_s = support_w.view(support_w.shape[0], 1) * pred_output_s
        pred_output_p = p_w.view(p_w.shape[0], 1) * pred_output_p
        pred_output_c = comp_w.view(comp_w.shape[0], 1) * pred_output_c
        pred_output[:,1] = pred_output_s[:,1]
        pred_output[:, 14:27] = pred_output_s[:,2:15]
        pred_output[:,2:8] = pred_output_p[:,1:7]
        pred_output[:,8:14] = pred_output_c[:,1:7]
        #pred_output = F.softmax(pred_output, dim=1)

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


    def calculate_Single_recall(self, insnum, heter_edge_output,gt_rel):
        # 计算 Proximity Comparative type的recall
        pred_output_s, pred_output_p, pred_output_c =  heter_edge_output
        #self.need_softmax = True
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

class Heter_Relation_Recall():
    def __init__(self, len_dataset, need_softmax=True):
        self.recall = {20: 0, 50: 0, 100: 0}
        self.ngc_recall = {20: 0, 50: 0, 100: 0}
        self.m_recall = {20: 0, 50: 0, 100: 0}
        self.m_recall_cat = np.zeros((4, 26))
        self.len_dataset = len_dataset
        self.need_softmax = need_softmax

    def calculate_Hierarchi_Heter_recall(self, obj_output, heter_edge_output, gt_obj, gt_rel,):
        obj_num = gt_obj.shape[0]
        # prepare predicted relation triplet
        pred_output_s, pred_output_p, pred_output_c = heter_edge_output
        nps, npp, npc = pred_output_s.cpu().numpy(), pred_output_p.cpu().numpy(),pred_output_c.cpu().numpy()

        obj_scores = obj_output.cpu().numpy()
        obj_inds = np.array([[i, j] for i in range(obj_num) for j in range(obj_num) if i != j])
        obj_score = obj_scores.max(1)
        obj_output = obj_scores.argmax(1)
        pred_output_cp = torch.zeros((obj_num * obj_num - obj_num, 13)).cuda()
        pred_output_cp[:, 1:7] = pred_output_p[:, 1:7]
        pred_output_cp[:, 7:13] = pred_output_c[:, 1:7]

        pred_scores_s = pred_output_s.cpu().numpy()
        pred_scores_p = pred_output_p.cpu().numpy()
        pred_scores_c = pred_output_c.cpu().numpy()
        pred_scores_cp = pred_output_cp.cpu().numpy()

        pred_score_s = pred_scores_s[:, 1:].max(1)
        pred_score_p = pred_scores_p[:, 1:].max(1)
        pred_score_c = pred_scores_c[:, 1:].max(1)
        pred_score_cp = pred_scores_cp[:, 1:].max(1)

        #pred_output_s = pred_scores_s[:, 1:].argmax(1)# + 1
        pred_output_s = support_label_np[pred_scores_s[:, 1:].argmax(1)]# + 1
        pred_output_p = proximity_label_np[pred_scores_p[:, 1:].argmax(1)]# + 1
        pred_output_c = comparative_label_np[pred_scores_c[:, 1:].argmax(1)]# + 1
        pred_output_cp = cp_label_np[pred_scores_cp[:, 1:].argmax(1)]# + 1

        triplet_s, triplet_score_s = _triplet(obj_inds, obj_output, pred_output_s, obj_score, pred_score_s)
        triplet_p, triplet_score_p = _triplet(obj_inds, obj_output, pred_output_p, obj_score, pred_score_p)
        triplet_c, triplet_score_c = _triplet(obj_inds, obj_output, pred_output_c, obj_score, pred_score_c)
        triplet_cp, triplet_score_cp = _triplet(obj_inds, obj_output, pred_output_cp, obj_score, pred_score_cp)

        sorted_triplet_s = triplet_s[np.argsort(-triplet_score_s)]
        sorted_triplet_p = triplet_p[np.argsort(-triplet_score_p)]
        sorted_triplet_c = triplet_c[np.argsort(-triplet_score_c)]
        sorted_triplet_cp = triplet_cp[np.argsort(-triplet_score_cp)]

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
        pred_to_gt_s = _compute_pred_matches(gt_triplet, sorted_triplet_s)
        pred_to_gt_p = _compute_pred_matches(gt_triplet, sorted_triplet_p)
        pred_to_gt_c = _compute_pred_matches(gt_triplet, sorted_triplet_c)
        pred_to_gt_cp = _compute_pred_matches(gt_triplet, sorted_triplet_cp)

        for i in range(gt_triplet.shape[0]):
            self.m_recall_cat[0, gt_triplet[i, 1]-1] += 1
        # calculate recalls
        value = 0
        rel_edges = gt_rel[:, :2]
        num_single_rel = torch.unique(rel_edges, dim=0).shape[0]
        for k in self.recall:
            # the following code are copied from Neural-MOTIFS
            match_s = reduce(np.union1d, pred_to_gt_s[:k]).astype('int16')
            match_p = reduce(np.union1d, pred_to_gt_p[:k]).astype('int16')
            match_c = reduce(np.union1d, pred_to_gt_c[:k]).astype('int16')
            match_cp = reduce(np.union1d, pred_to_gt_cp[:k]).astype('int16')
            len_match = len(match_s)+len(match_cp)#len(match_p) + len(match_c)#len(match_cp)
            rec_i = float(len_match) / (float(gt_rel.shape[0])*1)
            #rec_i = float(len_match) / float(num_single_rel)
            acc_rel = round(rec_i,2)
            self.recall[k] += rec_i
            if k == 50:
                value = rec_i
                print(acc_rel)

            if k == 20:
                for i in range(match_s.shape[0]):
                    self.m_recall_cat[1, gt_triplet[match_s[i], 1]-1] += 1
                for i in range(match_cp.shape[0]):
                    self.m_recall_cat[1, gt_triplet[match_cp[i], 1]-1] += 1
                '''for i in range(match_p.shape[0]):
                    self.m_recall_cat[1, gt_triplet[match_p[i], 1] - 1] += 1
                for i in range(match_c.shape[0]):
                    self.m_recall_cat[1, gt_triplet[match_c[i], 1] - 1] += 1'''
            if k == 50:
                for i in range(match_s.shape[0]):
                    self.m_recall_cat[2, gt_triplet[match_s[i], 1] - 1] += 1
                for i in range(match_cp.shape[0]):
                    self.m_recall_cat[2, gt_triplet[match_cp[i], 1] - 1] += 1
                '''for i in range(match_c.shape[0]):
                    self.m_recall_cat[1, gt_triplet[match_c[i], 1] - 1] += 1'''
            if k == 100:
                for i in range(match_s.shape[0]):
                    self.m_recall_cat[3, gt_triplet[match_s[i], 1] - 1] += 1
                for i in range(match_cp.shape[0]):
                    self.m_recall_cat[3, gt_triplet[match_cp[i], 1] - 1] += 1
                '''for i in range(match_c.shape[0]):
                    self.m_recall_cat[1, gt_triplet[match_c[i], 1] - 1] += 1'''
        #print("与边匹配准确度：　", acc_rel)
        #self.Rel_acc.append(round(acc_rel,2))

    #def calculate_recall(self, obj_output, pred_output, gt_obj, gt_rel):
    def calculate_Heter_recall(self, obj_output, heter_edge_output, gt_obj, gt_rel,):
        obj_num = gt_obj.shape[0]
        # prepare predicted relation triplet
        pred_output_s, pred_output_p, pred_output_c = heter_edge_output
        nps, npp, npc = pred_output_s.cpu().numpy(), pred_output_p.cpu().numpy(),pred_output_c.cpu().numpy()

        # (Ne,2)     (Ne,15) 1,14~26   (Ne,7) 2~7    (Ne,7) 8~13
        pred_output = torch.zeros((obj_num * obj_num - obj_num, 27)).cuda()
        pred_output[:, 1] = pred_output_s[:, 1]
        pred_output[:, 14:27] = pred_output_s[:, 2:15]
        pred_output[:, 2:8] = pred_output_p[:, 1:7]
        pred_output[:, 8:14] = pred_output_c[:, 1:7]
        #pred_output = F.softmax(pred_output, dim=1) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pred_output_single = torch.zeros((obj_num * obj_num - obj_num, 4)).cuda()
        pred_output_single[:, 1] = torch.sum(pred_output_s[:, 1:15], dim=1)
        pred_output_single[:, 2] = torch.sum(pred_output_p[:, 1:7], dim=1)
        pred_output_single[:, 3] = torch.sum(pred_output_c[:, 1:7], dim=1)

        np_pred = pred_output_single.cpu().numpy()

        if self.need_softmax:
            obj_scores = F.softmax(obj_output, dim=1).cpu().numpy()
            pred_scores = F.softmax(pred_output, dim=1).cpu().numpy()
        else:
            obj_scores = obj_output.cpu().numpy()
            pred_scores = pred_output.cpu().numpy()
            pred_scores_single = pred_output_single.cpu().numpy()
        obj_inds = np.array([[i, j] for i in range(obj_num) for j in range(obj_num) if i != j])
        obj_score = obj_scores.max(1)
        pred_score = pred_scores[:, 1:].max(1)
        pred_score_single = pred_scores_single[:, 1:].max(1)

        obj_output = obj_scores.argmax(1)
        pred_output = pred_scores[:, 1:].argmax(1) + 1
        triplet, triplet_score = _triplet(obj_inds, obj_output, pred_output, obj_score, pred_score)
        '''obj_score = np.ones(obj_num)
        obj_output = gt_obj.cpu().numpy()
        triplet, triplet_score = _triplet(obj_inds, obj_output, pred_output, obj_score, pred_score)'''

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

        pred_tri_20 = sorted_triplet[:20]
        ii =0
        '''for rel in pred_tri_20:
            ii += 1
            s,p,o = rel
            print(ii,':',classes[s],'--', relationships[p], '--', classes[o])'''
        # calculate matches
        pred_to_gt = _compute_pred_matches(gt_triplet, sorted_triplet)
        #print(pred_to_gt[:20])

        # calculate recalls
        value = 0
        rel_edges = gt_rel[:, :2]
        num_single_rel = torch.unique(rel_edges,dim=0).shape[0]

        for k in self.recall:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k]).astype('int16')
            rec_i = float(len(match)) / float(gt_rel.shape[0])
            #rec_i = float(len(match)) / float(num_single_rel)
            self.recall[k] += rec_i
            if k == 50:
                value = rec_i
                #print(rec_i)


    def calculate_Heter_mean_recall(self, obj_output, heter_edge_output, gt_obj, gt_rel,):
        obj_num = gt_obj.shape[0]
        # prepare predicted relation triplet
        pred_output_s, pred_output_p, pred_output_c = heter_edge_output
        #pred_output_p = F.softmax(pred_output_p, dim=1)
        nps, npp, npc = pred_output_s.cpu().numpy(), pred_output_p.cpu().numpy(),pred_output_c.cpu().numpy()

        # (Ne,2)     (Ne,15) 1,14~26   (Ne,7) 2~7    (Ne,7) 8~13
        pred_output = torch.zeros((obj_num * obj_num - obj_num, 27)).cuda()
        pred_output[:, 1] = pred_output_s[:, 1]
        pred_output[:, 14:27] = pred_output_s[:, 2:15]
        pred_output[:, 2:8] = pred_output_p[:, 1:7]
        pred_output[:, 8:14] = pred_output_c[:, 1:7]

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
        '''obj_score = np.ones(obj_num)
        obj_output = gt_obj.cpu().numpy()
        triplet, triplet_score = _triplet(obj_inds, obj_output, pred_output, obj_score, pred_score)'''

        sorted_triplet = triplet[np.argsort(-triplet_score)]
        # prepare gt relation triplet

        gt_obj_inds = gt_rel[:, :2].cpu().numpy()
        gt_obj = gt_obj.cpu().numpy()
        gt_pred = gt_rel[:, 2].cpu().numpy()
        gt_triplet = _triplet(gt_obj_inds, gt_obj, gt_pred, None, None)

        pred_tri_20 = sorted_triplet[:20]
        ii =0
        '''for rel in pred_tri_20:
            ii += 1
            s,p,o = rel
            print(ii,':',classes[s],'--', relationships[p], '--', classes[o])'''
        # calculate matches
        pred_to_gt = _compute_pred_matches(gt_triplet, sorted_triplet)
        #print(pred_to_gt[:20])
        for i in range(gt_triplet.shape[0]):
            self.m_recall_cat[0, gt_triplet[i, 1]-1] += 1
        # calculate recalls
        value = 0
        rel_edges = gt_rel[:, :2]
        num_single_rel = torch.unique(rel_edges,dim=0).shape[0]

        for k in self.m_recall:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k]).astype('int16')
            rec_i = float(len(match)) / float(gt_rel.shape[0])
            #rec_i = float(len(match)) / float(num_single_rel)

            if k == 20:
                for i in range(match.shape[0]):
                    self.m_recall_cat[1, gt_triplet[match[i], 1]-1] += 1
            if k == 50:
                for i in range(match.shape[0]):
                    self.m_recall_cat[2, gt_triplet[match[i], 1]-1] += 1
            if k == 100:
                for i in range(match.shape[0]):
                    self.m_recall_cat[3, gt_triplet[match[i], 1]-1] += 1



    def calculate_Heter_ngc_recall(self, obj_output, heter_edge_output, gt_obj, gt_rel):
        obj_num = gt_obj.shape[0]
        # prepare predicted relation triplet
        if self.need_softmax:
            obj_scores = F.softmax(obj_output, dim=1).cpu().numpy()
        else:
            obj_scores = obj_output.cpu().numpy()
        obj_inds = np.array([[i, j] for i in range(obj_num) for j in range(obj_num) if i != j])
        obj_score = obj_scores.max(1)
        obj_output = obj_scores.argmax(1)

        pred_output_s, pred_output_p, pred_output_c = heter_edge_output
        # (Ne,2)     (Ne,15) 1,14~26   (Ne,7) 2~7    (Ne,7) 8~13
        pred_output = torch.zeros((obj_num * obj_num - obj_num, 27)).cuda()
        pred_output[:, 1] = pred_output_s[:, 1]
        pred_output[:, 14:27] = pred_output_s[:, 2:15]
        pred_output[:, 2:8] = pred_output_p[:, 1:7]
        pred_output[:, 8:14] = pred_output_c[:, 1:7]
        #pred_output = F.softmax(pred_output, dim=1) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!　注掉！

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
        recall_res = 'Rel R@20: %f; ' % (self.recall[20]) + 'Rel R@50: %f; ' % (self.recall[50]) + 'Rel ' \
                                                                                                   'R@100: %f;' % (self.recall[100])
        recall_res_ngc = 'Rel R@20: %f; ' % (self.ngc_recall[20]) + 'Rel R@50: %f; ' % (self.ngc_recall[50]) + 'Rel R@100: %f;' % (self.ngc_recall[100])
        recall_res_m = 'Rel R@20: %f; ' % (self.m_recall[20]) + 'Rel R@50: %f; ' % (self.m_recall[50]) + 'Rel R@100: %f;' % (self.m_recall[100])
        return recall_res, recall_res_ngc, recall_res_m

    def reset(self):
        self.__init__()


class Predicate_Accuracy():
    def __init__(self, len_dataset, need_softmax=True):
        self.len_dataset = len_dataset
        self.binary_acc = 0
        self.binary_recall = 0
        self.binary_acc_1 =0
        self.acc = 0
        self.acc_1 = 0
        self.acc_type_1 = 0

        self.acc_mean = 0.0
        self.acc_mean_1 = 0.0
        self.acc_mean_type_1 = 0.0
        self.acc_category = np.zeros(27)
        self.acc_category_1 = np.zeros(27)
        self.acc_category_type_1 = np.zeros(3)
        self.count_category = np.zeros(27)
        self.count_category_1 = np.zeros(27)
        self.count_category_type_1 = np.zeros(3)
        self.type_acc_1 = [0,0]
        self.need_softmax = need_softmax

    def calculate_accuracy_binary(self, insnum, pred_output, gt_rel):
        if self.need_softmax:
            pred_output = F.softmax(pred_output, dim=1)
        pred_label = torch.argmax(pred_output, dim=1)  # for example tensor([ 0,  0,  0, 16, 20, 16,  0,  0,  0,  0,  0,  0], device='cuda:0')
        gt = torch.zeros(pred_label.shape[0])
        correct_count = 0
        correct_count_1 = 0

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
        #print("binary_acc_all: ", self.binary_acc)

        pairs = gt_rel[:, :2]
        real_edge = torch.unique(pairs, dim=0)
        edge_num_1 = real_edge.shape[0]
        correct_edge_id = []

        for i in range(gt.shape[0]):
            if pred_label[i] != 0 and gt[i] == 1 and i not in correct_edge_id:
                correct_count_1 += 1
                correct_edge_id.append(i)
        self.binary_acc_1 = self.binary_acc_1 + (correct_count_1 / edge_num_1)
        #print("binary_acc_1: ", self.binary_acc_1)

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
        self.acc = self.acc + (count / pred_label.shape[0])

    def calculate_accuracy_1(self, insnum, pred_output, gt_rel):
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
            self.count_category_1[int(gt_rel[i, 2])] += 1

        pairs = gt_rel[:, :2]
        real_edge = torch.unique(pairs, dim=0)
        edge_num_1 = real_edge.shape[0]

        count = 0
        for i in range(pred_label.shape[0]):
            idx = pred_label[i]
            if idx!=0 and onehot[i, idx] == 1:
                count += 1
                self.acc_category_1[idx] += 1
        self.acc_1 = self.acc_1 + (count / gt_rel.shape[0])

    def calculate_accuracy_type_1(self, insnum, pred_output, gt_rel):
        pred_output = F.softmax(pred_output, dim=1)
        pred_label = torch.argmax(pred_output, dim=1)
        gt = torch.zeros(pred_label.shape[0])

        support_count_1 = 0
        cp_count_1 = 0
        correct_count_1 =0
        pairs = gt_rel[:, :2]
        real_edge_id = []
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            predgt = int(gt_rel[i, 2])
            if (predgt in support_label):
                type_id = 1
            elif (predgt in cp_label):
                type_id = 2
            else:
                type_id = 0 # impossible
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum - 1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum - 1) + idx_j
            gt[idx] = type_id
            if idx not in real_edge_id:
                real_edge_id.append(idx)
            self.count_category_type_1[type_id] += 1

        for i in range(gt.shape[0]):
            idx = pred_label[i]
            if idx in support_label:
                type_idx = 1
            elif idx in cp_label:
                type_idx = 2
            else:
                type_idx = 0

            if type_idx ==1 and gt[i] == 1:
                support_count_1 += 1
                self.acc_category_type_1[type_idx] += 1
            elif type_idx ==2 and gt[i]==2:
                cp_count_1 += 1
                self.acc_category_type_1[type_idx] += 1
            correct_count_1 = support_count_1 + cp_count_1
        self.acc_type_1 = self.acc_type_1 + (correct_count_1 / gt_rel.shape[0])

        '''onehot = torch.zeros((insnum * insnum - insnum, 3)).cuda()
        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]  #obj
            idx_j = gt_rel[i, 1]  #suj
            predgt = gt_rel[i, 2]
            if (predgt in support_label):
                type_id = 1
            elif (predgt in cp_label):
                type_id = 2
            else:
                type_id = 0
            if idx_i < idx_j:
                onehot[int(idx_i * (insnum-1) + idx_j - 1), type_id] = 1
            elif idx_i > idx_j:
                onehot[int(idx_i * (insnum-1) + idx_j), type_id] = 1
            self.count_category_type_1[type_id] += 1

        pairs = gt_rel[:, :2]
        real_edge = torch.unique(pairs, dim=0)
        edge_num_1 = real_edge.shape[0]

        count = 0
        for i in range(pred_label.shape[0]):
            idx = pred_label[i]
            if idx in support_label:
                type_idx = 1
            elif idx in cp_label:
                type_idx = 2
            else:
                type_idx = 0
            # 27 转换为3大类type(none, support, com&pro)
            if type_idx!=0 and onehot[i, type_idx] == 1:
                count += 1
                self.acc_category_type_1[type_idx] += 1
        self.acc_type_1 = self.acc_type_1 + (count / gt_rel.shape[0])'''

    def final_update(self):
        self.binary_acc = self.binary_acc / self.len_dataset
        self.binary_acc_1 = self.binary_acc_1 / self.len_dataset
        self.binary_recall = self.binary_recall / self.len_dataset
        self.acc = self.acc / self.len_dataset
        self.acc_1 = self.acc_1 / self.len_dataset
        self.acc_type_1 = self.acc_type_1 / self.len_dataset

        for i in range(27):
            if self.count_category[i] == 0:
                self.count_category[i] = 1
        for i in range(27):
            if self.count_category_1[i] == 0:
                self.count_category_1[i] = 1
        for i in range(3):
            if self.count_category_type_1[i] == 0:
                self.count_category_type_1[i] = 1
        self.acc_category = self.acc_category / self.count_category
        self.acc_mean = self.acc_category.mean()
        self.acc_category_1 = self.acc_category_1 / self.count_category_1
        self.acc_mean_1 = self.acc_category_1[1:].mean()
        self.acc_category_type_1 = self.acc_category_type_1 / self.count_category_type_1
        self.acc_mean_type_1 = self.acc_category_type_1[1:].mean()
        self.type_acc_1 = self.acc_category_type_1[1:]

    def print_string(self):
        res = 'Predicate Binary Acc all: %f; ' % self.binary_acc + 'Relation Binary Recall: %f; ' % self.binary_recall + 'Relation Acc: %f; ' % self.acc + '\n'
        res += f'Predicate mean acc: {self.acc_mean}'
        res += 'Predicate Binary Acc 1: %f; ' % self.binary_acc_1
        res += ' 26 Relation Acc 1(only edge): %f; ' % self.acc_1
        res += f' 26 Predicate  mean acc 1(only edge): {self.acc_mean_1}'+ '\n'
        res += ' 2 type Relation Acc 1(only edge): %f; ' % self.acc_type_1 + '\n'
        res += f' 2 type Predicate  mean acc 1(only edge): {self.acc_mean_type_1}'+ '\n'
        res += f' 2 type Predicate acc 1(only edge): {self.type_acc_1}' + '\n'

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

class Relation_Recall():
    def __init__(self, len_dataset, need_softmax=True):
        self.recall = {20: 0, 50: 0, 100: 0}
        self.ngc_recall = {20: 0, 50: 0, 100: 0}
        self.m_recall = {20: 0, 50: 0, 100: 0}
        self.m_recall_cat = np.zeros((4, 26))
        self.len_dataset = len_dataset
        self.need_softmax = need_softmax

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
        recall_res = 'Rel R@20: %f; ' % (self.recall[20]) + 'Rel R@50: %f; ' % (self.recall[50]) + 'Rel R@100: %f;' % (self.recall[100])
        recall_res_ngc = 'Rel R@20: %f; ' % (self.ngc_recall[20]) + 'Rel R@50: %f; ' % (self.ngc_recall[50]) + 'Rel R@100: %f;' % (self.ngc_recall[100])
        recall_res_m = 'Rel R@20: %f; ' % (self.m_recall[20]) + 'Rel R@50: %f; ' % (self.m_recall[50]) + 'Rel R@100: %f;' % (self.m_recall[100])
        return recall_res, recall_res_ngc, recall_res_m

    def reset(self):
        self.__init__()

class Edge_Accuracy():
    def __init__(self,len_dataset, need_softmax=True):
        self.len_dataset = len_dataset
        self.type_binary_acc_all = 0
        self.type_binary_acc_1 = 0
        self.link_binary_acc_1 = 0
        self.binary_recall = 0
        self.acc = 0
        self.acc_type_1 =0
        self.acc_mean = 0.0
        self.acc_category = np.zeros(3)
        self.acc_category_type_1 = np.zeros(3)
        self.count_category = np.zeros(3)
        self.count_category_type_1 = np.zeros(3)
        self.type_acc_1 = [0,0]

        self.need_softmax = need_softmax


    def calculate_accuracy_binary_link(self, insnum, link_output, gt_rel):
        link_output = F.softmax(link_output, dim=1)
        link_label = torch.argmax(link_output, dim=1)
        '''al = 2.2
        be = 0.025
        if link_output[:]> 1/al +be:
            link_label[:]=1
        else: link_label[:]=0'''

        gt = torch.zeros(link_label.shape[0])
        correct_count_1 = 0
        pairs = gt_rel[:,:2]
        real_edge = torch.unique(pairs,dim=0)
        edge_num_1 = real_edge.shape[0]
        correct_edge_id = []
        real_edge_id = []
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum-1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum-1) + idx_j
            gt[idx] = 1
            if idx not in real_edge_id:
                real_edge_id.append(idx)

        for i in range(gt.shape[0]):
            if link_label[i] != 0 and gt[i] == 1 and i not in correct_edge_id:
                correct_count_1 += 1
                correct_edge_id.append(i)
        self.link_binary_acc_1 = self.link_binary_acc_1 + (correct_count_1 /edge_num_1)
        #print("link_binary_acc_1: ", self.link_binary_acc_1 )


    def calculate_accuracy_binary_type(self, insnum, type_output, gt_rel):
        type_output = F.softmax(type_output, dim=1)
        type_label = torch.argmax(type_output, dim=1)  # for example tensor([ 0,  0,  0, 16, 20, 16,  0,  0,  0,  0,  0,  0], device='cuda:0')
        gt = torch.zeros(type_label.shape[0])
        correct_count_all = 0
        correct_count_1 = 0

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
            if type_label[i] == 0 and gt[i] == 0:
                correct_count_all += 1
            elif type_label[i] != 0 and gt[i] == 1:
                correct_count_all += 1
        self.type_binary_acc_all = self.type_binary_acc_all + (correct_count_all / type_label.shape[0])
        #print("type_binary_acc_all: ", self.type_binary_acc_all )

        pairs = gt_rel[:, :2]
        real_edge = torch.unique(pairs, dim=0)
        edge_num_1 = real_edge.shape[0]
        correct_edge_id = []

        for i in range(gt.shape[0]):
            if type_label[i] != 0 and gt[i] == 1 and i not in correct_edge_id:
                correct_count_1 += 1
                correct_edge_id.append(i)
        self.type_binary_acc_1 = self.type_binary_acc_1 + (correct_count_1 /edge_num_1)
        #print("type_binary_acc_1: ", self.type_binary_acc_1 )


    def calculate_accuracy(self, insnum, pred_output, gt_rel):
        pred_output = F.softmax(pred_output, dim=1)
        pred_label = torch.argmax(pred_output, dim=1)
        onehot = torch.zeros((insnum * insnum - insnum, 3)).cuda()
        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]  #obj
            idx_j = gt_rel[i, 1]  #suj
            predgt = gt_rel[i, 2]
            if (predgt in support_label):
                type_id = 1
            elif (predgt in cp_label):
                type_id = 2
            else:
                type_id = 0
            if idx_i < idx_j:
                onehot[int(idx_i * (insnum-1) + idx_j - 1), type_id] = 1
            elif idx_i > idx_j:
                onehot[int(idx_i * (insnum-1) + idx_j),type_id] = 1
            self.count_category[type_id] += 1

        '''for i in range(insnum * insnum - insnum):
            if torch.sum(onehot[i, :]) == 0:
                onehot[i, 0] = 1
                self.count_category[0] += 1  # 无边的关系加到 none类别  注意3dssg的none类别不在第0位'''
        pairs = gt_rel[:, :2]
        real_edge = torch.unique(pairs, dim=0)
        edge_num_1 = real_edge.shape[0]

        count = 0
        for i in range(pred_label.shape[0]):
            idx = pred_label[i]
            if idx!=0 and onehot[i, idx] == 1:
                count += 1
                self.acc_category[idx] += 1
        self.acc = self.acc + (count / edge_num_1)

    def calculate_accuracy_type_1(self, insnum, pred_output, gt_rel):
        pred_output = F.softmax(pred_output, dim=1)
        pred_label = torch.argmax(pred_output, dim=1)
        gt = torch.zeros(pred_label.shape[0])

        support_count_1 = 0
        cp_count_1 = 0
        correct_count_1 =0
        real_edge_id = []
        for i in range(gt_rel.shape[0]):
            idx_i = int(gt_rel[i, 0])
            idx_j = int(gt_rel[i, 1])
            predgt = int(gt_rel[i, 2])
            if (predgt in support_label):
                type_id = 1
            elif (predgt in cp_label):
                type_id = 2
            else:
                type_id = 0 # impossible
            idx = 0
            if idx_i < idx_j:
                idx = idx_i * (insnum - 1) + idx_j - 1
            elif idx_i > idx_j:
                idx = idx_i * (insnum - 1) + idx_j
            gt[idx] = type_id
            self.count_category_type_1[type_id] += 1

        for i in range(gt.shape[0]):
            idx = pred_label[i]
            type_idx = idx

            if type_idx ==1 and gt[i] == 1:
                support_count_1 += 1
                self.acc_category_type_1[type_idx] += 1
            elif type_idx ==2 and gt[i]==2:
                cp_count_1 += 1
                self.acc_category_type_1[type_idx] += 1
            correct_count_1 = support_count_1 + cp_count_1
        self.acc_type_1 = self.acc_type_1 + (correct_count_1 / gt_rel.shape[0])

    def final_update(self):
        #self.binary_acc = self.binary_acc / self.len_dataset
        #self.binary_recall = self.binary_recall / self.len_dataset
        self.type_binary_acc_all = self.type_binary_acc_all / self.len_dataset
        self.type_binary_acc_1 = self.type_binary_acc_1 / self.len_dataset
        self.link_binary_acc_1 = self.link_binary_acc_1  / self.len_dataset
        self.acc = self.acc / self.len_dataset
        self.acc_type_1 = self.acc_type_1 / self.len_dataset

        for i in range(3):
            if self.count_category[i] == 0:
                self.count_category[i] = 1
        for i in range(3):
            if self.count_category_type_1[i] == 0:
                self.count_category_type_1[i] = 1
        self.acc_category = self.acc_category / self.count_category
        self.acc_mean = self.acc_category[1:].mean()
        self.acc_category_type_1 = self.acc_category_type_1 / self.count_category_type_1
        self.acc_mean_type_1 = self.acc_category_type_1[1:].mean()
        self.type_acc_1 = self.acc_category_type_1[1:]

    def print_string(self):
        res = 'Predicate type Binary_acc_all: %f; ' % self.type_binary_acc_all
        res +='Predicate type Binary_acc_1: %f; ' % self.type_binary_acc_1
        res +='Predicate link Binary_acc_1: %f; ' % self.link_binary_acc_1 + 'Relation Acc 1(only edge): %f; ' % self.acc + '\n'
        res += f' 3 Predicate mean acc: {self.acc_mean}'
        res += ' 2 type Relation Acc 1(only edge): %f; ' % self.acc_type_1 + '\n'
        res += f' 2 type Predicate  mean acc 1(only edge): {self.acc_mean_type_1}'+ '\n'
        res += f' 2 type Predicate acc 1(only edge): {self.type_acc_1}'

        return res

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
