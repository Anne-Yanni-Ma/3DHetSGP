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
pred_w_type  =  torch.Tensor([0.25, 1, 1,1]).cuda()
# Project directory
PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_DIR, 'models'))
NOWEIGHT = False
NOWEIGHTLOSS = False
proximity_label = [2,3,4,5,6,7] # 不除去inside  6个
proximity_map = [1,2,3,4,5,6]

# Arguments declearation
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='HeterGNN_newEdge_model_sg_xy', help='model name [default: GNN_knowledge_fusion,HeterGNN_model]')
    parser.add_argument('--task', type=str, default='SGCls', help='Task type [default: PredCls,SGCls ]')
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


def prepare_onehot_objgt(gt_obj):
    insnum = gt_obj.shape[0]
    onehot = torch.zeros(insnum, 160).float().cuda()
    for i in range(insnum):
        onehot[i, gt_obj[i]] = 1
    return onehot


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
    experiment_dir = experiment_dir.joinpath('Eval_SGCls')
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
    testDataLoader = torch.utils.data.DataLoader(TEST_SET, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
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

    address_edge = os.path.join(PROJECT_DIR, 'log/pred_classification/2023-02-15_17-46')
    sys.path.append(address_edge)
    MODEL_edge = importlib.import_module('models.pred_classification.pred_classification')
    pred_embedder = MODEL_edge.get_model().cuda()
    dict_address_edge = os.path.join(address_edge, 'checkpoints/last_model.pth')
    checkpoint_edge = torch.load(dict_address_edge)
    pred_embedder.load_state_dict(checkpoint_edge['model_state_dict'])
    pred_embedder = pred_embedder.cuda().eval()
    address_network = os.path.join(PROJECT_DIR, 'log/model_SGCls/2024-01-16_21-55_best_noweight')
    sys.path.append(address_network)
    MODEL_network = importlib.import_module('models.HeterGNN_newEdge_model_sg_xy')
    network = MODEL_network.get_model().cuda()
    dict_address_network = os.path.join(address_network, 'checkpoints/best_rel_model.pth')
    checkpoint_network = torch.load(dict_address_network)
    network.load_state_dict(checkpoint_network['model_state_dict'])
    network = network.cuda().eval()
    MODEL_type = importlib.import_module('models.edge_classification.type_classification_MP')

    obj_w = TRAINING_SET.obj_w
    criterion_l = MODEL_LOSS.get_loss_l(alpha=0.1, beta=1, gamma=2, pred_w=lpred_w).cuda()  # cuda
    criterion_edge = MODEL_type.TypeLoss(gamma=2,pred_w=pred_w_type).cuda()
    criterion_s = MODEL_LOSS.get_loss_s(alpha=0.1, beta=1, gamma=2, pred_w=spred_w).cuda()   # cuda
    criterion_p = MODEL_LOSS.get_loss_p(alpha=0.1, beta=1, gamma=2, pred_w=ppred_w).cuda()  # cuda
    criterion_c = MODEL_LOSS.get_loss_c(alpha=0.1, beta=1, gamma=2, pred_w=cpred_w).cuda()  # cuda
    criterion_obj = MODEL_LOSS.get_loss_obj(alpha=0.1, beta=1, gamma=2, obj_w=obj_w).cuda()  # cuda


    global_epoch = 0
    best_loss = 999

    # ---------------- Start training --------------------

    test_obj_acc = Object_Accuracy(len(testDataLoader), need_softmax=False)
    test_obj_recall = Object_Recall(len(testDataLoader), need_softmax=False)
    test_pred_acc = Heter_Predicate_Accuracy(len(testDataLoader), need_softmax=False)
    test_pred_recall = Heter_Predicate_Recall(len(testDataLoader), need_softmax=False)
    test_rel_recall = Heter_Relation_Recall(len(testDataLoader), need_softmax=False)
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
            edge_descriptor = gen_edge_feature(pc_mat).cuda()  # (Ne, 11)

            pc_mat_node = pc_mat.cuda()  # (Nn,1024,3)
            pc_mat_edge = torch.clone(pc_mat_node).cuda()
            gt_obj = gt_obj.cuda().long()  # (Nn, )
            gt_rel = gt_rel.cuda().long()  # (Ne, 3)
            pc_geom_info = bboxes_minc.cuda(), lwhV.cuda(), centroid.cuda()
            insnum = pc_mat_node.shape[0]

            node_embedder = node_embedder.eval()  # (Nn,1024,3)
            pred_embedder = pred_embedder.eval()
            network = network.eval()  # MODEL: HeterGNN_OneEmb_model

            if args.task == "PredCls":
                node_knowledge = torch.Tensor(np.load('./data/meta_embedding/meta_embedding_node.npy')).cuda()
                obj_onehot = prepare_onehot_objgt(gt_obj)  # (Nn,num_obj)
                obj_codes = torch.mm(obj_onehot, node_knowledge)  # obj_codes (Nn,C_konw) = (Nn,160) *(160,C_know)
                obj_output = obj_onehot
            elif args.task == "SGCls":
                obj_output, obj_codes = node_embedder(pc_mat_node)

            pred_output, pred_codes = pred_embedder(pc_mat_edge)  # Ne,27   Ne,512

            types_w = types_output4
            multiW_Gated = gen_type_weight(types_w)
            link_wG, support_wG, p_wG, comp_wG = multiW_Gated
            Edge_weights = link_wG, support_wG, p_wG, comp_wG

            node_output, multi_logits, multi_edge_output = network(obj_codes, pred_codes, Edge_weights, pc_geom_info)
            edge_output_l, edge_output_s, edge_output_prox, edge_output_c, type_output = multi_edge_output
            loss_l = criterion_l(edge_output_l, gt_obj, gt_rel)
            loss_e = criterion_edge(type_output, gt_obj, gt_rel)
            loss_s = criterion_s(edge_output_s, gt_obj, gt_rel)
            loss_p = criterion_p(edge_output_prox, gt_obj, gt_rel)
            loss_c = criterion_c(edge_output_c, gt_obj, gt_rel)
            loss_obj = criterion_obj(node_output, gt_obj, gt_rel)
            if NOWEIGHTLOSS == True:
                pred_loss = loss_p + loss_c + loss_s
            else:
                pred_loss = loss_e + loss_p + loss_c + loss_s
            loss = loss_obj + 2 * pred_loss  ## alpha:1 beta: 0.1

            loss_sum = loss_sum + loss.item()

            node_output_eval = node_output.clone().detach()
            edge_output_s_eval = edge_output_s.clone().detach()
            edge_output_prox_eval = edge_output_prox.clone().detach()
            edge_output_comp_eval = edge_output_c.clone().detach()

            heter_edge_output_eval = edge_output_s_eval, edge_output_prox_eval, edge_output_comp_eval
            test_obj_acc.calculate_accuray(node_output_eval, gt_obj)  # node_output_eval (Nn,160) gt_obj (Nn,)
            test_obj_recall.calculate_recall(node_output_eval, gt_obj)
            test_pred_acc.calculate_Heter_accuracy(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
            test_pred_recall.calculate_Heter_recall(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
            test_pred_recall.calculate_Single_recall(pc_mat.shape[0], heter_edge_output_eval, gt_rel)
            test_rel_recall.calculate_Heter_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)
            # (Nn,160) (Ne, 27)  gt_obj (Nn,)  gt_rel (Ne, 3)
            test_rel_recall.calculate_Heter_ngc_recall(node_output_eval, heter_edge_output_eval, gt_obj, gt_rel)


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



if __name__ == "__main__":
    args = parse_args()
    main(args)
