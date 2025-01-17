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
from Heter_ssg_eval_tool_type import Type_Relation_Recall,Set_Relation_Recall

from models.obj_classification import obj_classification
from models.HeterGNN_newEdge_model_xy import gen_gt_typeslink
from models.op_utils import  gen_edge_feature
lpred_w =  torch.Tensor([0.25, 1]).cuda()
spred_w =  torch.Tensor([0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).cuda()
ppred_w =  torch.Tensor([0.25, 1, 1, 1, 1, 1, 1]).cuda()
cpred_w =  torch.Tensor([0.25, 1, 1, 1, 1, 1, 1]).cuda()
pred_w_type  =  torch.Tensor([0.25, 1, 1]).cuda()

# Project directory
PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_DIR, 'models'))
NOWEIGHT = False
NOWEIGHTLOSS = False
EVALWEIGHT = False

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
        elif prob >= 1 / alp + bel:
            prob = 1
        else:
            prob =  alp * prob - alp * bel  # alp*prob - alp*bel

        Gate_W[i] = prob
    return Gate_W

def Gate_2threhold(W, high, low):
    Gate_W = W.clone()
    for i in range(Gate_W.shape[0]):
        prob = Gate_W[i]
        if prob <= low:
            prob = 0
        elif prob >= high:
            prob = 1
        else:
            prob = 0.5  # alp*prob - alp*bel

        Gate_W[i] = prob
    return Gate_W

def gen_type_weight(type_output4):
    none_weight, support_weight, p_weight, c_weight = type_output4#[0],type_output4[1],type_output4[2],type_output4[3]
    #nps, npp, npc = support_weight.cpu().numpy(), p_weight.cpu().numpy(),c_weight.cpu().numpy()
    link_w = torch.ones(none_weight.shape[0]).cuda()
    none_w = Gate_threshold(none_weight, 0.8) # should retrain 0.7
    link_w = link_w-none_w
    support_w = Gate_threshold(support_weight,0.1)# should retrain 0.15 0.2 0.15
    p_w = Gate_threshold(p_weight,0.1)
    c_w = Gate_threshold(c_weight,0.1)

    types_weight = [link_w, support_w, p_w,c_w]
    return types_weight



def add_diag(insnum, M):
    for i in range(insnum):
        for j in range(insnum):
            if i!=j and M[i,j]==1:
                M[j,i]=1
    return M


def prepare_onehot_objgt(gt_obj):
    insnum = gt_obj.shape[0]
    onehot = torch.zeros(insnum, 160).float().cuda()
    for i in range(insnum):
        onehot[i, gt_obj[i]] = 1
    return onehot

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

def multiW_Acc(multiW_Gated, multiW_gt):
    link_w_Gated, support_w_Gated, p_w_Gated, comp_w_Gated = multiW_Gated
    gt_link_w, gt_support_w, gt_p_w, gt_comp_w = multiW_gt
    link_weight_acc, _, M_simarity_l = calculate_tensor_Acc(link_w_Gated, gt_link_w, gt_link_w.shape[0])
    support_weight_acc, zero_s, M_simarity_s = calculate_tensor_Acc(support_w_Gated, gt_support_w, gt_support_w.shape[0])
    proximity_weight_acc, zero_p, M_simarity_p = calculate_tensor_Acc(p_w_Gated, gt_p_w, gt_p_w.shape[0])
    comparative_weight_acc, zero_comp, M_simarity_comp = calculate_tensor_Acc(comp_w_Gated, gt_comp_w, gt_comp_w.shape[0])
    if (zero_s == 1):  support_weight_acc = None
    else: support_weight_acc =  round(support_weight_acc,2)
    if (zero_p == 1):  proximity_weight_acc = None
    else: proximity_weight_acc = round(proximity_weight_acc,2)
    if (zero_comp == 1):  comparative_weight_acc = None
    else: comparative_weight_acc = round(comparative_weight_acc,2)
    print("\n")
    print("Weight_ACC(L,S,P,C): ", link_weight_acc, support_weight_acc,proximity_weight_acc,comparative_weight_acc)
    print("M_Sim(L,S,P,C): ", round(M_simarity_l,2),round(M_simarity_s,2),round(M_simarity_p,2), round(M_simarity_comp,2))

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
    experiment_dir = experiment_dir.joinpath('HeterGNN_itera')
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
    trainDataLoader = torch.utils.data.DataLoader(TRAINING_SET, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_SET, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    # ---------------- Log dataset info --------------------
    log_string("The number of training data is: %d" % len(TRAINING_SET))

    # ---------------- Config network model --------------------
    MODEL = importlib.import_module(args.model)
    MODEL_LOSS = importlib.import_module(args.multiloss)
    shutil.copy(os.path.join(PROJECT_DIR, 'models/%s.py' % args.model), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'train_heterG_predcls_newEdge.py'), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'Heter_ssg_eval_tool.py'), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'models/pointnet.py'), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'models/HeterGNN_newEdge_model_xy.py'), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'models/graph.py'), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'models/utils.py'), str(experiment_dir))

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
    address_edge = os.path.join(PROJECT_DIR, 'log/pred_classification/2023-02-15_17-46') #2023-02-15_17-46
    #address_edge = os.path.join(PROJECT_DIR, 'log/pred_classification_new/2023-08-05_00-39_w0.1_concat') #2023-02-15_17-46
    #address_edge = os.path.join(PROJECT_DIR, 'log/pred_classification_new_ECC/2023-08-05_01-43_dgcnn_w0.1') #2023-02-15_17-46

    sys.path.append(address_edge)
    MODEL_edge = importlib.import_module('models.pred_classification.pred_classification')
    pred_embedder = MODEL_edge.get_model().cuda()
    dict_address_edge = os.path.join(address_edge, 'checkpoints/last_model.pth')
    checkpoint_edge = torch.load(dict_address_edge)
    pred_embedder.load_state_dict(checkpoint_edge['model_state_dict'])
    pred_embedder = pred_embedder.cuda().eval()

    obj_w = TRAINING_SET.obj_w
    pred_w = TRAINING_SET.pred_w
    network = MODEL.get_model().cuda()    # cuda  MODEL: GNN_knowledge_fusion
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
    network = network.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(network.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-8
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

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

    # ---------------- Start training --------------------
    for epoch in range(start_epoch, args.epoch):
        '''Train on scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        network = network.apply(lambda x: bn_momentum_adjust(x, momentum))

        loss_sum = 0
        train_obj_acc = Object_Accuracy(len(trainDataLoader), need_softmax=False)
        train_obj_recall = Object_Recall(len(trainDataLoader), need_softmax=False)

        train_pred_acc = Heter_Predicate_Accuracy(len(trainDataLoader), need_softmax=False)
        train_pred_recall = Heter_Predicate_Recall(len(trainDataLoader), need_softmax=False)
        train_rel_recall = Heter_Relation_Recall(len(trainDataLoader), need_softmax=False)

        # ---------------- Start batch set training --------------------
        bar = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9)
        for i, data in bar:
            #if i>10: break
            pc_mat, pc_geom_info, types_output4, gt_obj, gt_rel = data
            types_output4 = types_output4[0].cuda()
            bboxes, lwhV, centroid = pc_geom_info
            bboxes_minc = bboxes[0]
            # bboxes_maxc = bboxes[1]
            lwhV = lwhV[0]
            centroid = centroid[0]
            pc_mat = pc_mat[0]  # (Nn,1024,3)
            gt_obj = gt_obj[0]  # (Nn, )
            gt_rel = gt_rel[0]  # (Ne, 3)
            #edge_descriptor = gen_edge_feature(pc_mat).cuda() # (Ne, 11)
            edge_descriptor =[]
            pc_mat_node = pc_mat.cuda()  # (Nn,1024,3)
            pc_mat_edge = torch.clone(pc_mat_node).cuda()  # 深拷贝
            gt_obj = gt_obj.cuda().long()  # (Nn, )
            gt_rel = gt_rel.cuda().long()  # (Ne, 3)
            pc_geom_info = bboxes_minc.cuda(), lwhV.cuda(), centroid.cuda()
            insnum = pc_mat_node.shape[0]


            if args.task == "PredCls":
                node_knowledge = torch.Tensor(np.load('./data/meta_embedding/best_node.npy')).cuda()
                obj_onehot = prepare_onehot_objgt(gt_obj)  # (Nn,num_obj)
                obj_codes = torch.mm(obj_onehot, node_knowledge)  # obj_codes (Nn,C_konw) = (Nn,160) *(160,C_know)
                obj_output = obj_onehot
            elif args.task =="SGCls":
                obj_output, obj_codes = node_embedder(pc_mat_node)

            pred_output, pred_codes = pred_embedder(pc_mat_edge)  # Ne,27   Ne,512

            optimizer.zero_grad()
            network = network.train()

            multiW_gt = gen_gt_typeslink(pc_mat_node, gt_rel)
            link_wgt, support_wgt, p_wgt, comp_wgt = multiW_gt
            types_w = types_output4
            multiW_Gated = gen_type_weight(types_w)
            link_wG, support_wG, p_wG, comp_wG = multiW_Gated

            Edge_weights = link_wG, support_wG, p_wG, comp_wG
            #Edge_weights = link_wgt, support_wgt, p_wgt, comp_wgt

            #multiW_Acc(multiW_Gated, multiW_gt)

            #_, pret_obj_codes = node_embedder(pc_mat_node)# SGCls

            node_output,multi_logits, multi_edge_output = network(obj_codes,  pred_codes, Edge_weights, pc_geom_info,edge_descriptor) #IN: obj_codes (Nn,C_konw) pred_codes (Ne, 512)
            edge_output_s, edge_output_prox, edge_output_c = multi_edge_output
            #loss_l = criterion_l(edge_output_l, gt_obj, gt_rel)
            loss_e =  0#criterion_edge(type_output, gt_obj, gt_rel)
            loss_s = criterion_s(edge_output_s, gt_obj, gt_rel)
            loss_p = criterion_p(edge_output_prox, gt_obj, gt_rel)
            loss_c = criterion_c(edge_output_c, gt_obj, gt_rel)
            loss_obj = criterion_obj(node_output, gt_obj, gt_rel)
            if NOWEIGHTLOSS == True:
                pred_loss = loss_p + loss_c + loss_s
            else:
                pred_loss = loss_e + loss_p + loss_c + loss_s
            #loss = criterion(node_output, multi_edge_output, gt_obj, gt_rel,type_output)  # node_output (Nn,160) edge_output (Ne, 27)  are onehot

            loss = loss_obj + 2 * pred_loss  ## alpha:1 beta: 0.1
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.item()

            logit_s, logit_p, logit_c = multi_logits

            node_output_eval = node_output.clone().detach()
            #edge_output_l_eval = edge_output_l.clone().detach()
            edge_output_s_eval = edge_output_s.clone().detach()
            edge_output_prox_eval = edge_output_prox.clone().detach()
            edge_output_comp_eval = edge_output_c.clone().detach()

            # 　Todo: comparative 的 pred_output还未产生
            heter_edge_output_eval = edge_output_s_eval, edge_output_prox_eval, edge_output_comp_eval
            #heter_edge_output_eval = logit_s.clone().detach(), logit_p.clone().detach(), logit_c.clone().detach()
            #train_obj_acc.calculate_accuray(node_output_eval, gt_obj) # node_output_eval (Nn,160) gt_obj (Nn,)
            #train_obj_recall.calculate_recall(node_output_eval, gt_obj)
            train_pred_acc.calculate_Heter_accuracy(pc_mat.shape[0],heter_edge_output_eval, gt_rel)
            #train_pred_recall.calculate_Heter_recall_w(pc_mat.shape[0], heter_edge_output_eval, gt_rel,types_w)
            #train_pred_recall.calculate_Single_recall(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
            train_rel_recall.calculate_Heter_recall(node_output_eval,  heter_edge_output_eval, gt_obj, gt_rel)  #(Nn,160) (Ne, 27)  gt_obj (Nn,)  gt_rel (Ne, 3)
            train_rel_recall.calculate_Heter_ngc_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)
            train_rel_recall.calculate_Heter_mean_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)

        #train_obj_acc.final_update()
        #train_obj_recall.final_update()
        train_pred_acc.final_update()
        #train_pred_recall.final_update()
        train_rel_recall.final_update()

        log_string('Training mean loss: %f' % (loss_sum / len(trainDataLoader)))
        #log_string('Training ' + train_obj_acc.print_string())
        #log_string('Training ' + train_obj_recall.print_string())
        log_string('Training ' + train_pred_acc.print_string())
        #log_string('Training ' + train_pred_recall.print_string())
        train_recall_res, train_recall_res_ngc, train_m_recall_res = train_rel_recall.print_string()
        log_string('Training ' + train_recall_res)
        #log_string('Training NGC ' + train_recall_res_ngc)
        #log_string('Training mean ' + train_m_recall_res)

        test_obj_acc = Object_Accuracy(len(testDataLoader), need_softmax=False)
        test_obj_recall = Object_Recall(len(testDataLoader), need_softmax=False)
        test_pred_acc = Heter_Predicate_Accuracy(len(testDataLoader), need_softmax=False)
        test_pred_recall = Heter_Predicate_Recall(len(testDataLoader), need_softmax=False)
        test_rel_recall = Heter_Relation_Recall(len(testDataLoader), need_softmax=False)
        '''test_rel_recall_set_s = Set_Relation_Recall(len(testDataLoader), need_softmax=False, set_type="support")
        test_rel_recall_set_p = Set_Relation_Recall(len(testDataLoader), need_softmax=False, set_type="proximity")
        test_rel_recall_set_c = Set_Relation_Recall(len(testDataLoader), need_softmax=False, set_type="comparative")'''

        with torch.no_grad():
            loss_sum = 0
            log_string('---- EPOCH %03d TEST ----' % (global_epoch + 1))
            test_bar = tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9)
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
                #edge_descriptor = gen_edge_feature(pc_mat).cuda()  # (Ne, 11)
                edge_descriptor=[]
                pc_mat_node = pc_mat.cuda()   # (Nn,1024,3)
                pc_mat_edge = torch.clone(pc_mat_node).cuda()
                gt_obj = gt_obj.cuda().long() # (Nn, )
                gt_rel = gt_rel.cuda().long() # (Ne, 3)
                pc_geom_info = bboxes_minc.cuda(), lwhV.cuda(), centroid.cuda()
                insnum = pc_mat_node.shape[0]

                node_embedder = node_embedder.eval()  # (Nn,1024,3)
                #pred_embedder = pred_embedder.eval()

                if args.task == "PredCls":
                    node_knowledge = torch.Tensor(np.load('./data/meta_embedding/best_node.npy')).cuda()
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
                #Edge_weights = link_wgt, support_wgt, p_wgt, comp_wgt

                network = network.eval() # MODEL: HeterGNN_OneEmb_model
                node_output, multi_logits, multi_edge_output = network(obj_codes, pred_codes, Edge_weights, pc_geom_info,edge_descriptor)  # IN: obj_codes (Nn,C_konw) pred_codes (Ne, 512)
                edge_output_s, edge_output_prox, edge_output_c = multi_edge_output
                #loss_l = criterion_l(edge_output_l, gt_obj, gt_rel)
                loss_e = 0# criterion_edge(type_output, gt_obj, gt_rel)
                loss_s = criterion_s(edge_output_s, gt_obj, gt_rel)
                loss_p = criterion_p(edge_output_prox, gt_obj, gt_rel)
                loss_c = criterion_c(edge_output_c, gt_obj, gt_rel)
                loss_obj = criterion_obj(node_output, gt_obj, gt_rel)
                if NOWEIGHTLOSS == True:
                    pred_loss = loss_p + loss_c + loss_s
                else:
                    pred_loss = loss_e + loss_p + loss_c + loss_s
                loss =   loss_obj + 2 * pred_loss  ## alpha:1 beta: 0.1

                loss_sum = loss_sum + loss.item()

                node_output_eval = node_output.clone().detach()
                #edge_output_l_eval = edge_output_l.clone().detach()
                edge_output_s_eval = edge_output_s.clone().detach()
                edge_output_prox_eval = edge_output_prox.clone().detach()
                edge_output_comp_eval = edge_output_c.clone().detach()

                logit_s, logit_p, logit_c = multi_logits
                heter_edge_output_eval = edge_output_s_eval, edge_output_prox_eval, edge_output_comp_eval
                #heter_edge_output_eval = logit_s.clone().detach(), logit_p.clone().detach(), logit_c.clone().detach()

                test_obj_acc.calculate_accuray(node_output_eval, gt_obj)# node_output_eval (Nn,160) gt_obj (Nn,)
                test_obj_recall.calculate_recall(node_output_eval, gt_obj)
                test_pred_acc.calculate_Heter_accuracy(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
                test_pred_recall.calculate_Heter_recall(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
                #test_pred_recall.calculate_Hierarchi_Heter_recall(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
                #test_pred_recall.calculate_Heter_recall_w(pc_mat.shape[0], heter_edge_output_eval, gt_rel,types_w)
                test_pred_recall.calculate_Single_recall(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
                test_rel_recall.calculate_Heter_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)  # (Nn,160) (Ne, 27)  gt_obj (Nn,)  gt_rel (Ne, 3)
                test_rel_recall.calculate_Heter_ngc_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)
                '''test_rel_recall_set_s.calculate_Set_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)
                test_rel_recall_set_s.calculate_Set_ngc_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)

                test_rel_recall_set_p.calculate_Set_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)
                test_rel_recall_set_p.calculate_Set_ngc_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)

                test_rel_recall_set_c.calculate_Set_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)
                test_rel_recall_set_c.calculate_Set_ngc_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)'''
                test_rel_recall.calculate_Heter_mean_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)

            test_obj_acc.final_update()
            test_obj_recall.final_update()
            test_pred_acc.final_update()
            test_pred_recall.final_update()
            test_rel_recall.final_update()
            '''test_rel_recall_set_s.final_update()
            test_rel_recall_set_p.final_update()
            test_rel_recall_set_c.final_update()'''

            log_string('Evaluation mean loss: %f' % (loss_sum / len(testDataLoader)))
            log_string('Eval ' + test_obj_acc.print_string())
            log_string('Eval ' + test_obj_recall.print_string())
            log_string('Eval ' + test_pred_acc.print_string())
            log_string('Eval ' + test_pred_recall.print_string())
            test_recall_res, test_recall_res_ngc, test_m_recall_res = test_rel_recall.print_string()
            log_string('Eval ' + test_recall_res)
            log_string('Eval NGC ' + test_recall_res_ngc)
            log_string('Eval mean ' + test_m_recall_res)

            '''test_recall_res_set_s, test_recall_res_ngc_set_s, test_m_recall_res_set_s = test_rel_recall_set_s.print_string()
            log_string('s--Set Eval ' + test_recall_res_set_s)
            log_string('s--Set Eval NGC ' + test_recall_res_ngc_set_s)
            log_string('s--Set Eval mean ' + test_m_recall_res_set_s)

            test_recall_res_set_p, test_recall_res_ngc_set_p, test_m_recall_res_set_p = test_rel_recall_set_p.print_string()
            log_string('p--Set Eval ' + test_recall_res_set_p)
            log_string('p--Set Eval NGC ' + test_recall_res_ngc_set_p)
            log_string('p--Set Eval mean ' + test_m_recall_res_set_p)

            test_recall_res_set_c, test_recall_res_ngc_set_c, test_m_recall_res_set_c = test_rel_recall_set_c.print_string()
            log_string('c--Set Eval ' + test_recall_res_set_c)
            log_string('c--Set Eval NGC ' + test_recall_res_ngc_set_c)
            log_string('c--Set Eval mean ' + test_m_recall_res_set_c)'''

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

                logger.info('Save Best Rel model...')
                savepath = str(checkpoints_dir) + '/best_rel_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'current_loss': curr_loss,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

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
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_mR_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'current_loss': curr_loss,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            log_string('Best mean loss: %f' % (best_loss))
            log_string('Best rel R@20: %f; ' % (best_rel_R20) + 'rel R@50: %f; ' % (best_rel_R50) + 'rel R@100: %f;' % (best_rel_R100))
            log_string('Best NGC rel R@20: %f; ' % (best_rel_R20_ngc) + 'rel R@50: %f; ' % (best_rel_R50_ngc) + 'rel R@100: %f;' % (best_rel_R100_ngc))
            log_string('Best rel mean R@20: %f; ' % (best_rel_mR20) + 'rel R@50: %f; ' % (best_rel_mR50) + 'rel R@100: %f;' % (best_rel_mR100))
        global_epoch += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
