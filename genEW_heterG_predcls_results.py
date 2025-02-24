import os
import sys
import argparse
import logging
import time
import datetime
import importlib
import shutil
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from data.dataloader_EW import DataLoader_3DSSG
from Heter_ssg_eval_tool import Object_Accuracy, Object_Recall, Heter_Predicate_Accuracy, Heter_Predicate_Recall, Heter_Relation_Recall
from models.obj_classification import obj_classification
from models.op_utils import  gen_edge_feature


lpred_w =  torch.Tensor([0.25, 1]).cuda()
spred_w =  torch.Tensor([0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).cuda()
ppred_w =  torch.Tensor([0.25, 1, 1, 1, 1, 1, 1]).cuda()
cpred_w =  torch.Tensor([0.25, 1, 1, 1, 1, 1, 1]).cuda()
pred_w_type  =  torch.Tensor([0.25, 1, 1, 1]).cuda()

support_label = [1,14,15,16,17,18,19,20,21,22,23,24,25,26]  # len: 14
suppport_map = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
proximity_label = [2,3,4,5,6,7] # 不除去inside  6个
proximity_map = [1,2,3,4,5,6]
comparative_label = [8,9,10,11,12,13] # 6个
comparative_map = [1,2,3,4,5,6]
cp_label = [2,3,4,5,6,7,8,9,10,11,12,13] #12个

# Project directory
PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_DIR, 'models'))
NOWEIGHT = False
NOWEIGHTLOSS = False



# Arguments declearation
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='HeterGNN_newEdge_model_xy', help='model name [default: GNN_knowledge_fusion,HeterGNN_model]')
    parser.add_argument('--task', type=str, default='PredCls', help='Task type [default: PredCls,SGCls ]')
    parser.add_argument('--multiloss', type=str, default='HeterGNN_loss', help='model name [default: GNN_knowledge_fusion,HeterGNN_model]')
    parser.add_argument('--epoch',  default=100, type=int, help='Epoch to run [default: 100]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()


def prepare_onehot_objgt(gt_obj):
    insnum = gt_obj.shape[0]
    onehot = torch.zeros(insnum, 160).float().cuda()
    for i in range(insnum):
        onehot[i, gt_obj[i]] = 1
    return onehot


def save_weights(types_output4, data_path):
    weight_path = os.path.join(data_path, "edge_weights_sg.npy")
    '''link_w, support_w, cp_w = types_weight
    types_weight = torch.stack((link_w, support_w, cp_w),dim=0).clone().detach().cpu()'''
    types_output4 = types_output4.clone().detach().cpu()
    np.save(weight_path, types_output4)
    print("Save this types_output4 in ",data_path, "Done.")

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
        #if pred_gt in cp_label:
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


def Gate_threshold(W, threshold):
    Gate_W = W.clone()
    for i in range(Gate_W.shape[0]):
        prob = Gate_W[i]
        if prob > threshold:
            prob = 1
        else:
            prob = 0
        Gate_W[i] = prob
    return Gate_W


def Gate(W, alp, bel):
    Gate_W = W.clone()
    for i in range(Gate_W.shape[0]):
        prob = Gate_W[i]
        if prob <= bel:
            prob = 0
        elif prob >= 0.95:
            prob = 1
        else:
            prob = prob  # alp*prob - alp*bel
        Gate_W[i] = prob
    return Gate_W


def calculate_tensor_Acc(w_Gated, gt_w,size):
    zero_tag =0
    num_gt = torch.sum(gt_w == 1)
    if(num_gt==0):
        zero_tag= 1
        weight_acc =0
    else:
        count_correct = torch.sum((w_Gated == gt_w) & (gt_w == 1))
        weight_acc = int(count_correct) / int(num_gt)
    count_sim = torch.sum((w_Gated == gt_w))
    weight_M_simarity =  int(count_sim) /  size
    return weight_acc,zero_tag,weight_M_simarity

def gen_Type_outputs(edge_outputs):
    none_weight, edge_output_s, edge_output_p, edge_output_c = edge_outputs#.clone()
    support_weight = torch.sum((edge_output_s[:,1:15]),dim=1)
    p_weight = torch.sum((edge_output_p[:, 1:7]),dim=1)
    c_weight = torch.sum((edge_output_c[:, 1:7]),dim=1)
    #none_weight = edge_output_s[:,0]*edge_output_p[:,0]*edge_output_c[:,0]
    type_output = torch.stack([none_weight, support_weight, p_weight, c_weight], dim=0)
    return type_output

def gen_type_weight(type_output4):
    none_weight, support_weight, p_weight, c_weight = type_output4
    nps, npp, npc = support_weight.cpu().numpy(), p_weight.cpu().numpy(),c_weight.cpu().numpy()
    link_w = torch.ones(none_weight.shape[0]).cuda()
    none_w = Gate_threshold(none_weight, 0.9)
    link_w = link_w-none_w
    support_w = Gate_threshold(support_weight,0.05)
    p_w = Gate_threshold(p_weight,0.05)
    c_w = Gate_threshold(c_weight,0.05)

    types_weight = [link_w, support_w, p_w,c_w]
    return types_weight

def gen_type_weight_lth(type_output4):
    none_weight, support_weight, p_weight, c_weight = type_output4
    nps, npp, npc = support_weight.cpu().numpy(), p_weight.cpu().numpy(),c_weight.cpu().numpy()
    link_weight = torch.ones(none_weight.shape[0]).cuda()-none_weight
    link_w = Gate_threshold(link_weight, 0.5)
    support_w = Gate_threshold(support_weight,0.1)
    p_w = Gate_threshold(p_weight,0.1)
    c_w = Gate_threshold(c_weight,0.1)

    types_weight = [link_w, support_w, p_w,c_w]
    return types_weight

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # cuda
    # torch.backends.cudnn.enabled = False  # disable cudnn for CUDNN_NOT_SUPPORT, may not contiguous
    # ---------------- Create log dir -------------------
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('gen_Edge_weights')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # ---------------- Training set --------------------
    TRAINING_SET = DataLoader_3DSSG(training=True, per25=True)
    TEST_SET = DataLoader_3DSSG(training=False)
    trainDataLoader = torch.utils.data.DataLoader(TRAINING_SET, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_SET, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    # ---------------- Log dataset info --------------------
    log_string("The number of training data is: %d" % len(TRAINING_SET))

    # ---------------- Config network model --------------------
    MODEL = importlib.import_module(args.model) #GNN_knowledge_fusion
    MODEL_LOSS = importlib.import_module(args.multiloss) #GNN_knowledge_fusion

    # initial pretrained node/edge embedder
    address_node = os.path.join(PROJECT_DIR, 'log/obj_classification/2023-03-05_20-52')
    sys.path.append(address_node)
    MODEL_node = importlib.import_module('models.obj_classification.obj_classification')
    node_embedder = MODEL_node.get_model().cuda()
    dict_address_node = os.path.join(address_node, 'checkpoints/last_model.pth')
    checkpoint_node = torch.load(dict_address_node)
    node_embedder.load_state_dict(checkpoint_node['model_state_dict'])
    node_embedder = node_embedder.cuda().eval()

    #  26 pred_classify
    address_edge = os.path.join(PROJECT_DIR, 'log/pred_classification/2023-02-15_17-46')
    sys.path.append(address_edge)
    MODEL_edge = importlib.import_module('models.pred_classification.pred_classification')
    pred_embedder = MODEL_edge.get_model_multi_output().cuda()
    dict_address_edge = os.path.join(address_edge, 'checkpoints/last_model.pth')
    checkpoint_edge = torch.load(dict_address_edge)
    pred_embedder.load_state_dict(checkpoint_edge['model_state_dict'])
    pred_embedder = pred_embedder.cuda().eval()

    #address_network = os.path.join(PROJECT_DIR, 'log/HeterGNN_newEdge_PredCls/2023-07-09_11-55_prox_xy')
    address_network = os.path.join(PROJECT_DIR, 'log/XXXXXXXXXXX/XXXXXXXXX')
    sys.path.append(address_network)
    MODEL_network = importlib.import_module('models.HeterGNN_newEdge_model_xy')
    network = MODEL_network.get_model().cuda()
    dict_address_network = os.path.join(address_network, 'checkpoints/best_mR_model.pth')
    checkpoint_network = torch.load(dict_address_network)
    network.load_state_dict(checkpoint_network['model_state_dict'])
    network = network.cuda().eval()

    obj_w = TRAINING_SET.obj_w
    pred_w = TRAINING_SET.pred_w
    #criterion = MODEL.get_loss(alpha=0.1, beta=1, gamma=2, obj_w=obj_w).cuda()   # cuda
    #criterion_l = MODEL_LOSS.get_loss_l(alpha=0.1, beta=1, gamma=2).cuda()  # cuda  , pred_w=spred_w
    criterion_l = MODEL_LOSS.get_loss_l(alpha=0.1, beta=1, gamma=2, pred_w=lpred_w).cuda()  # cuda  , pred_w=spred_w
    criterion_s = MODEL_LOSS.get_loss_s(alpha=0.1, beta=1, gamma=2, pred_w=spred_w).cuda()   # cuda
    criterion_p = MODEL_LOSS.get_loss_p(alpha=0.1, beta=1, gamma=2, pred_w=ppred_w).cuda()  # cuda
    criterion_c = MODEL_LOSS.get_loss_c(alpha=0.1, beta=1, gamma=2, pred_w=cpred_w).cuda()  # cuda
    criterion_obj = MODEL_LOSS.get_loss_obj(alpha=0.1, beta=1, gamma=2, obj_w=obj_w).cuda()  # cuda
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Conv1d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0

    global_epoch = 0
    best_loss = 999
    best_rel_R20 = 0
    best_rel_R50 = 0
    best_rel_R100 = 0
    best_rel_R20_ngc = 0
    best_rel_R50_ngc = 0
    best_rel_R100_ngc = 0
    best_rel_mR20 = 0
    best_rel_mR50 = 0
    best_rel_mR100 = 0
    #testDataLoader =trainDataLoader
    # ---------------- Start training --------------------
    for epoch in range(start_epoch, 1):

        test_obj_acc = Object_Accuracy(len(testDataLoader), need_softmax=False)
        test_obj_recall = Object_Recall(len(testDataLoader), need_softmax=False)
        test_pred_acc = Heter_Predicate_Accuracy(len(testDataLoader), need_softmax=False)
        test_pred_recall = Heter_Predicate_Recall(len(testDataLoader), need_softmax=False)
        test_rel_recall = Heter_Relation_Recall(len(testDataLoader), need_softmax=False)
        with torch.no_grad():
            loss_sum = 0
            log_string('---- EPOCH %03d TEST ----' % (global_epoch + 1))
            test_bar = tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9)
            sum_support_weight = 0
            sum_proximity_weight = 0
            sum_link_weight = 0
            sum_comparative_weight = 0
            sum_M_simarity_l = 0
            sum_M_simarity_s = 0
            sum_M_simarity_p = 0
            sum_M_simarity_comp = 0

            link_weight_acc_all = []
            support_weight_acc_all = []
            proximity_weight_acc_all = []
            comparative_weight_acc_all = []
            skip_scene_s = 0
            skip_scene_p = 0
            skip_scene_comp = 0
            sum_weight_gtP = 0
            sum_M_simarity_gtP = 0
            for i, data in test_bar:
                pc_mat, pc_geom_info, types_output4, gt_obj, gt_rel = data
                types_output4 = types_output4[0].cuda()
                bboxes, lwhV, centroid = pc_geom_info
                bboxes_minc = bboxes[0]
                lwhV = lwhV[0]
                centroid = centroid[0]
                pc_mat = pc_mat[0]  # (Nn,1024,3)
                gt_obj = gt_obj[0]  # (Nn, )
                gt_rel = gt_rel[0]  # (Ne, 3)
                edge_descriptor = gen_edge_feature(pc_mat).cuda()  # (Ne, 11)

                pc_mat_node = pc_mat.cuda()   # (Nn,1024,3)
                pc_mat_edge = torch.clone(pc_mat_node).cuda()
                gt_obj = gt_obj.cuda().long() # (Nn, )
                gt_rel = gt_rel.cuda().long() # (Ne, 3)
                pc_geom_info = bboxes_minc.cuda(), lwhV.cuda(), centroid.cuda()

                node_embedder = node_embedder.eval()  # (Nn,1024,3)
                pred_embedder = pred_embedder.eval()
                #edge_embedder = edge_embedder.eval()

                if args.task == "PredCls":
                    node_knowledge = torch.Tensor(np.load('./data/meta_embedding/meta_embedding_node.npy')).cuda()
                    obj_onehot = prepare_onehot_objgt(gt_obj)  # (Nn,num_obj)
                    obj_codes = torch.mm(obj_onehot, node_knowledge)  # obj_codes (Nn,C_konw) = (Nn,160) *(160,C_know)
                    obj_output = obj_onehot
                elif args.task == "SGCls":
                    obj_output, obj_codes = node_embedder(pc_mat_node)

                pred_output, pred_codes = pred_embedder(pc_mat_edge)  # Ne,27   Ne,512
                multiW_gt = gen_gt_typeslink(pc_mat_node, gt_rel)
                link_wgt, support_wgt, p_wgt, comp_wgt = multiW_gt
                types_w = types_output4
                multiW_Gated = gen_type_weight(types_w)
                link_wG, support_wG, p_wG, comp_wG = multiW_Gated

                Edge_weights = link_wG, support_wG, p_wG, comp_wG
                network = network.eval() # MODEL: HeterGNN_OneEmb_model

                node_output, multi_logits, multi_edge_output = network(obj_codes, pred_codes, Edge_weights, pc_geom_info,edge_descriptor)  # IN: obj_codes (Nn,C_konw) pred_codes (Ne, 512)
                edge_output_s, edge_output_prox, edge_output_c = multi_edge_output
                loss_l = 0#criterion_l(edge_output_l, gt_obj, gt_rel)
                loss_e = 0
                loss_s = criterion_s(edge_output_s, gt_obj, gt_rel)
                loss_p = criterion_p(edge_output_prox, gt_obj, gt_rel)
                loss_c = criterion_c(edge_output_c, gt_obj, gt_rel)
                loss_obj = criterion_obj(node_output, gt_obj, gt_rel)
                if NOWEIGHTLOSS == True:
                    pred_loss = loss_l + loss_p + loss_c + loss_s
                else:
                    pred_loss = loss_e + loss_l + loss_p + loss_c + loss_s
                loss =   loss_obj + 2 * pred_loss  ## alpha:1 beta: 0.1
                loss_sum = loss_sum + loss.item()
                #edge_output_comp = gen_comp_output(pc_mat.shape[0], Wcomp, pc_geom_info)

                edge_output_s, edge_output_p, edge_output_c = multi_edge_output
                edge_output_l = edge_output_s

                #data_path = TRAINING_SET.training_list[i]
                data_path = TEST_SET.test_list[i]
                save_weights(types_output4,data_path)

                node_output_eval = node_output.clone().detach()
                edge_output_s_eval = edge_output_s.clone().detach()
                edge_output_prox_eval = edge_output_prox.clone().detach()
                edge_output_comp_eval = edge_output_c.clone().detach()

                heter_edge_output_eval = edge_output_s_eval, edge_output_prox_eval, edge_output_comp_eval
                #heter_edge_output_eval = logit_s.clone().detach(), logit_p.clone().detach(), logit_c.clone().detach()

                test_obj_acc.calculate_accuray(node_output_eval, gt_obj)# node_output_eval (Nn,160) gt_obj (Nn,)
                test_obj_recall.calculate_recall(node_output_eval, gt_obj)
                test_pred_acc.calculate_Heter_accuracy(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
                test_pred_recall.calculate_Heter_recall(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
                #test_pred_recall.calculate_Heter_recall_w(pc_mat.shape[0], heter_edge_output_eval, gt_rel,types_w)
                test_pred_recall.calculate_Single_recall(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
                test_rel_recall.calculate_Heter_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)  # (Nn,160) (Ne, 27)  gt_obj (Nn,)  gt_rel (Ne, 3)
                test_rel_recall.calculate_Heter_ngc_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)

                edge_outputs = edge_output_s[:,0], edge_output_s, edge_output_p, edge_output_c
                types_output4 = gen_Type_outputs(edge_outputs)
                types_weight = gen_type_weight(types_output4)
                gt_types_w = gen_gt_typeslink(pc_mat_node, gt_rel)
                gt_link_w, gt_support_w, gt_p_w, gt_comp_w = gt_types_w

                link_w_Gated, support_w_Gated, p_w_Gated, comp_w_Gated = types_weight
                link_weight_acc, _, M_simarity_l = calculate_tensor_Acc(link_w_Gated, gt_link_w, gt_link_w.shape[0])
                link_weight_acc_all.append(round(link_weight_acc, 2))
                sum_link_weight += link_weight_acc
                sum_M_simarity_l += M_simarity_l
                #print("link_weight_acc: ", link_weight_acc)
                #print("M_simarity_l: ", M_simarity_l)

                support_weight_acc, zero_s, M_simarity_s = calculate_tensor_Acc(support_w_Gated, gt_support_w,
                                                                                gt_support_w.shape[0])
                if (zero_s == 1):
                    #print("support_weight_acc = 'none'")
                    support_weight_acc_all.append("none")
                    skip_scene_s += 1
                else:
                    support_weight_acc_all.append(round(support_weight_acc, 2))
                # support_weight_acc_all.append(round(support_weight_acc, 2))
                sum_support_weight += support_weight_acc
                sum_M_simarity_s += M_simarity_s
                #print("support_weight_acc: ",support_weight_acc)
                #print("M_simarity_s: ", M_simarity_s)
                proximity_weight_acc, zero_p, M_simarity_p = calculate_tensor_Acc(p_w_Gated, gt_p_w, gt_p_w.shape[0])
                if (zero_p == 1):
                    proximity_weight_acc_all.append("none")
                    skip_scene_p += 1
                else:
                    proximity_weight_acc_all.append(round(proximity_weight_acc, 2))

                sum_proximity_weight += proximity_weight_acc
                sum_M_simarity_p += M_simarity_p

                comparative_weight_acc, zero_comp, M_simarity_comp = calculate_tensor_Acc(comp_w_Gated, gt_comp_w,
                                                                                          gt_comp_w.shape[0])
                if (zero_comp == 1):
                    #print("comparative_weight_acc = 'none'")
                    comparative_weight_acc_all.append("none")
                    skip_scene_comp += 1
                else:
                    comparative_weight_acc_all.append(round(comparative_weight_acc, 2))

                sum_comparative_weight += comparative_weight_acc
                sum_M_simarity_comp += M_simarity_comp
                # print("comparative_weight_acc: ", comparative_weight_acc)
                # print("M_simarity_comp: ", M_simarity_comp)

            avg_link_weight_acc = sum_link_weight / len(testDataLoader)
            avg_support_weight_acc = sum_support_weight / (len(testDataLoader) - skip_scene_s)
            avg_proximity_weight_acc = sum_proximity_weight / (len(testDataLoader) - skip_scene_p)
            avg_comparative_weight_acc = sum_comparative_weight / (len(testDataLoader) - skip_scene_comp)
            avg_M_simarity_l = sum_M_simarity_l / len(testDataLoader)
            avg_M_simarity_s = sum_M_simarity_s / len(testDataLoader)
            avg_M_simarity_p = sum_M_simarity_p / len(testDataLoader)
            avg_M_simarity_comp = sum_M_simarity_comp / len(testDataLoader)

            print("avg_M_simarity_l: ", avg_M_simarity_l)
            print("avg_link_weight_acc: ", avg_link_weight_acc)
            #print("link_weight_acc_all: ", link_weight_acc_all)

            print("avg_M_simarity_s: ", avg_M_simarity_s)
            print("avg_support_weight_acc: ", avg_support_weight_acc)
            #print("support_weight_acc_all: ", support_weight_acc_all)

            print("avg_M_simarity_p: ", avg_M_simarity_p)
            print("avg_proximity_weight_acc: ", avg_proximity_weight_acc)
            #print("proximity_weight_acc_all", proximity_weight_acc_all)

            print("avg_M_simarity_comp: ", avg_M_simarity_comp)
            print("avg_comparative_weight_acc: ", avg_comparative_weight_acc)
            #print("comparative_weight_acc_all", comparative_weight_acc_all)

            test_obj_acc.final_update()
            test_obj_recall.final_update()
            test_pred_acc.final_update()
            test_pred_recall.final_update()
            test_rel_recall.final_update()
            log_string('Evaluation mean loss: %f' % (loss_sum / len(testDataLoader)))
            log_string('Eval ' + test_obj_acc.print_string())
            log_string('Eval ' + test_obj_recall.print_string())
            log_string('Eval ' + test_pred_acc.print_string())
            log_string('Eval ' + test_pred_recall.print_string())
            test_recall_res, test_recall_res_ngc, test_m_recall_res = test_rel_recall.print_string()
            log_string('Eval ' + test_recall_res)
            log_string('Eval NGC ' + test_recall_res_ngc)
            log_string('Eval mean ' + test_m_recall_res)

            curr_loss = loss_sum / len(testDataLoader)
            curr_rel_R20 = test_rel_recall.recall[20]
            curr_rel_R50 = test_rel_recall.recall[50]
            curr_rel_R100 = test_rel_recall.recall[100]
            curr_rel_R20_ngc = test_rel_recall.ngc_recall[20]
            curr_rel_R50_ngc = test_rel_recall.ngc_recall[50]
            curr_rel_R100_ngc = test_rel_recall.ngc_recall[100]
            curr_rel_mR20 = test_rel_recall.m_recall[20]
            curr_rel_mR50 = test_rel_recall.m_recall[50]
            curr_rel_mR100 = test_rel_recall.m_recall[100]

            if curr_rel_R50 > best_rel_R50 or curr_rel_R100 > best_rel_R100:
                best_rel_R20 = curr_rel_R20
                best_rel_R50 = curr_rel_R50
                best_rel_R100 = curr_rel_R100
                log_string('--Eval-- Best rel: R@20: %f; ' % (best_rel_R20) + 'rel R@50: %f; ' % (
                    best_rel_R50) + 'rel R@100: %f; ' % (best_rel_R100))

            if curr_rel_mR50 > best_rel_mR50 or curr_rel_mR100 > best_rel_mR100:
                best_rel_R20 = curr_rel_R20
                best_rel_R50 = curr_rel_R50
                best_rel_R100 = curr_rel_R100
                best_rel_R20_ngc = curr_rel_R20_ngc
                best_rel_R50_ngc = curr_rel_R50_ngc
                best_rel_R100_ngc = curr_rel_R100_ngc
                best_rel_mR20 = curr_rel_mR20
                best_rel_mR50 = curr_rel_mR50
                best_rel_mR100 = curr_rel_mR100
                best_loss = curr_loss


            log_string('Best mean loss: %f' % (best_loss))
            log_string('Best rel R@20: %f; ' % (best_rel_R20) + 'rel R@50: %f; ' % (best_rel_R50) + 'rel R@100: %f;' % (best_rel_R100))
            log_string('Best NGC rel R@20: %f; ' % (best_rel_R20_ngc) + 'rel R@50: %f; ' % (best_rel_R50_ngc) + 'rel R@100: %f;' % (best_rel_R100_ngc))
            log_string('Best rel mean R@20: %f; ' % (best_rel_mR20) + 'rel R@50: %f; ' % (best_rel_mR50) + 'rel R@100: %f;' % (best_rel_mR100))
        global_epoch += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
