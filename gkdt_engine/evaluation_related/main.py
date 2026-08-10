import os
import time
import datetime
import argparse
from yacs.config import CfgNode
import pprint
import logging
import numpy as np
import random
import copy
# import pdb

import sys
sys.path.append('.')  # append pwd into system path so that it could find python modules
print(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.backends.cudnn as cudnn
# torch.autograd.set_detect_anomaly(True)
from tensorboardX import SummaryWriter

from config import update_config, create_loggers
from utils.utils import list2str, load_samples, make_grid_images, image_normalize, AverageMeter, summarize_losses
from evaluation_related.build_dataset_coco_eval import build_episode_loader, batch_data_to_device, parse_batch_data
from datasets.transforms import recover_kps
from datasets.dataset_utils import draw_instance, draw_skeletons, draw_markers
from network.openkd_model import get_openkd_model, OpenKDModel
from core.loss_lw import HeatmapLoss, DirectCoordLoss
from core.misc import compute_openkd_heatmap_loss, split_main_aux_heatmaps
from evaluation_metric import PCKMetric, NEMetric
from vis import show_save_episode, save_predictions, save_heatmaps

import misc

#==========
#TODO
from evaluation_metric.write_prediction_to_json import init_coco_prediction_file,add_coco_predictions,save_coco_predictions
import json
#==========

############################################################################################
## main call
############################################################################################
def get_optimizer(cfg, model):
    optimizer_type = cfg.TRAIN.OPTIMIZER
    lr = cfg.TRAIN.LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    if  optimizer_type== 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError

    return optimizer

def validate(cfg, model, test_loader, **kwargs):
    print('==============testing start==============')
    model.eval()  # affects BN & disable Dropout
    torch.set_grad_enabled(False)  # disable grad computation

    square_image_length = cfg.DATASET.SQUARE_IMAGE_LENGTH  # 384

    # define evaluation metrics
    pck_thresh_type = 'bbox'  # 'bbox' or 'img'
    if pck_thresh_type == 'bbox':
        pck_thresh = [0.1]  # [0.1, 0.2]
    else:
        pck_thresh = [0.06]  # [0.06, 0.10], 0.06 * 384 = 23.04 pixels (23 pixels)
    pck_metric = PCKMetric(pck_thresholds=pck_thresh)  # Percentage of Keypoints (PCK)
    ne_metric = NEMetric()  # Normalized Error

    #=========================================================
    #TODO
    # 定义文件路径
    original_gt_file = "/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json"
    output_pred_file = "/project/vonneumann1/cl2025/GKD/experiments/expert_benchmark/pred_file/bilinear_person_keypoints_val2017.json"
    #original_gt_file = "/project/vonneumann1/cl2025/keypoint_datasets/human_pose/human_art/eval_AP_annotations/validation_humanart.json"
    #output_pred_file = "/project/vonneumann1/cl2025/GKD/experiments/evaluation_for_coco/annotations/output_validation_humanart.json"
    
    # 初始化COCO预测文件
    pred_data = init_coco_prediction_file(output_pred_file)
    
    # 加载原始COCO标注
    with open(original_gt_file, 'r') as f:
        original_gt_data = json.load(f)
    original_gt_dict = {ann['id']: ann for ann in original_gt_data['annotations']}
    #=========================================================

    episode_i = 0
    num_test_episodes = len(test_loader)  # in testing/val, we set to rollout one test episode per batch (see build_dataset.py)
    for batch_i, batch_episodes_data in enumerate(test_loader):
        batch_episodes_data = batch_data_to_device(batch_episodes_data, is_gpu=True)

        (queries, query_labels, query_kp_mask, scale_trans_q, bbox_origin_q, w_h_origin_q), \
        (supports, support_labels, support_kp_mask, scale_trans_s, bbox_origin_s, w_h_origin_s), \
        (obj_texts, obj_texts_mask), \
        (kps_texts, kps_texts_mask), \
        (episode_indexes, num_kp_original) = parse_batch_data(batch_episodes_data, other_info=test_loader.dataset.other_info)

        #print("episode_indexes:",episode_indexes) #episode_indexes is current episode_indexes

        outputs = model(queries, supports, support_labels, support_kp_mask, obj_texts, obj_texts_mask, kps_texts, kps_texts_mask)
        predict_heatmaps_list = outputs[0]  # [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict
        heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect = model.openkd_heatmap_fuse(
            predict_heatmaps_list[0],  # {'obj': tensor, 'text': tensor, 'image': tensor}
            support_kp_mask[0],
            kps_texts_mask[0],
            multi_group_supervision=cfg.LOSS.MULTI_GROUP_SUPERVISION,  # True or False
            fusing_operation=cfg.LOSS.OBJ_KP_HEATMAP_FUSION  # 'avg' or 'prod'
        )
        heatmaps_predict = heatmaps_fused  # B2 x N x h x w

        # multiple episode images into one (S=1)
        query_labels = query_labels[0]                                # B2 x N x 2
        valid_kp_mask = query_kp_mask[0] * (fused_mask_sum>0).long()  # B2 x N
        scale_trans_q = scale_trans_q[0]                      # B2 x 6
        bbox_origin_q = bbox_origin_q[0]                      # B2 x 4
        w_h_origin_q = w_h_origin_q[0]                        # B2 x 2

        # coordinates decoding
        B2, N = heatmaps_predict.shape[:2]
        if cfg.LOSS.TYPE == 'direct_coord':  # no need to decode
            predictions = heatmaps_predict
        else:
            H, W = heatmaps_predict.shape[2:]
            predict_score, predict_grids = torch.max(heatmaps_predict.reshape(B2, N, -1), 2)  # B2 x N
            predict_gridxy = torch.FloatTensor(B2, N, 2).cuda()
            predict_gridxy[:, :, 0] = predict_grids % W  # grid x
            predict_gridxy[:, :, 1] = predict_grids // H  # grid y
            # 'MSE', 'cross-entropy'
            predictions = ((predict_gridxy + 0.5) / H - 0.5) * 2

        predictions = predictions * valid_kp_mask.view(B2, N, 1)
        query_labels = query_labels * valid_kp_mask.view(B2, N, 1)

        predictions = predictions.cpu().detach()
        query_labels = query_labels.cpu().detach()
        valid_kp_mask = valid_kp_mask.cpu().detach()
        scale_trans_q, bbox_origin_q, w_h_origin_q = scale_trans_q.cpu(), bbox_origin_q.cpu(), w_h_origin_q.cpu()

        # square distance diff in original image scale
        predictions_o = recover_kps(predictions, square_image_length, scale_trans_q)
        query_labels_o = recover_kps(query_labels, square_image_length, scale_trans_q)

        print("predictions_o.shape:",predictions_o.shape)
        exit()
        #==========================================================
        # TODO: Add coco evaluation codes (For Yuxin). 
        # 使用独立函数生成COCO格式预测结果
        # 添加当前batch的预测结果
        add_coco_predictions(
            pred_data=pred_data,
            gkd_dataset=test_loader.dataset,
            episode_indexes=episode_indexes,
            predictions_o=predictions_o,
            original_gt_dict=original_gt_dict
        )
        #==========================================================


        if pck_thresh_type == 'bbox':
            edges_for_pck = (bbox_origin_q[:, [2, 3]]).numpy()  # B2 x 2
        else:  # == 'img'
            edges_for_pck = w_h_origin_q.numpy()  # B2 x 2
        pck_metric.compute_pck_and_update(predictions_o.numpy(), query_labels_o.numpy(), valid_kp_mask.numpy(), edges_for_pck)
        
        edges_for_ne = w_h_origin_q.numpy()  # B2 x 2
        ne_metric.compute_ne_and_update(predictions_o.numpy(), query_labels_o.numpy(), valid_kp_mask.numpy(), edges_for_ne)

        # increment in episode_i: only 1 episode each time in testing/val.
        episode_i += 1  

        if (episode_i % 20 == 0 or episode_i == num_test_episodes):
            acc_mean, interval = pck_metric.get_mean_accuracy_result()
            ne_mean, ne_interval = ne_metric.get_mean_ne_result()
            print('episode {}/{}, Acc {}, Int. {}, NE {:.6f}, Int. {:.6f}, time: {}'.format(episode_i, num_test_episodes,
                                acc_mean, interval, ne_mean, ne_interval, datetime.datetime.now()))      

    #==========================================================
    # TODO:
    # 循环结束后保存最终结果
    save_coco_predictions(pred_data, output_pred_file)
    #==========================================================

    sum_tps, num_tps = np.sum(pck_metric.tps[0]), len(pck_metric.tps[0])
    print('episode {}/{}, {}/{} ({:.4f}), time: {}'.format(num_test_episodes, num_test_episodes, sum_tps, num_tps, sum_tps/max(num_tps, 1), datetime.datetime.now()))
    print('==============testing end================')
    
    torch.set_grad_enabled(True)  # enable grad computation
    model.train()

    return acc_mean, interval, ne_mean

def main():
    cfg, args = update_config()
    print(cfg)
    # print(pprint.pformat(cfg))
    
    manual_seed = cfg.MANUAL_SEED
    if manual_seed is not None:
        # cudnn.benchmark = False  # require to import torch.backends.cudnn as cudnn
        # cudnn.deterministic = True
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
    
    # two ways to control cfg_str, 2023.08.15
    if len(cfg.AUTONAME.KEYS) == 0:
        cfg_file_basename = os.path.basename(args.cfg_file)
        cfg_str = os.path.splitext(cfg_file_basename)[0]
    else:
        assert len(cfg.AUTONAME.LABELS) == len(cfg.AUTONAME.KEYS)
        cfg_str = ''
        for k, key_tmp in enumerate(cfg.AUTONAME.KEYS):
            label_tmp = cfg.AUTONAME.LABELS[k]
            value_tmp = eval('cfg.'+key_tmp)
            str_value = list2str(value_tmp) if isinstance(value_tmp, (list, tuple)) else str(value_tmp)
            if label_tmp == '':  # if label is an empty str, just continue
                continue
            cfg_str += (label_tmp+str_value)
    print('==>cfg_str: ', cfg_str)

    output_model_dir, logger, tb_writer = create_loggers(cfg, cfg_str)
    # logger.info(cfg)

    print("==>Preparing model")
    openkd_model = get_openkd_model(cfg)
    optimizer = get_optimizer(cfg, openkd_model)
    if cfg.LOSS.TYPE in ['MSE', 'sigmoid-bce', 'cross-entropy']:  # supervised by GT heatmap
        loss_func = HeatmapLoss(cfg)
    elif cfg.LOSS.TYPE == 'direct_coord':  # supervised by GT keypoints
        loss_func = DirectCoordLoss(cfg)
    else:
        raise NotImplementedError

    checkpoint_file_wo_ext = os.path.join(output_model_dir, cfg_str)
    checkpoint_file = checkpoint_file_wo_ext + '.%s'%cfg.LOAD_CHECKPOINT_TYPE  # e.g., /output/openkd.best
    if cfg.RESUME and os.path.exists(checkpoint_file):
        # TODO: why model weights are in cuda, optimizer weights in cpu? We try to use map_location to solve optimizer's problem but not work.
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))  
        openkd_model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        cfg.START_EPOCH = checkpoint['epoch'] + 1  # set epoch to start training / val
        print("==>Checkpoint loaded: '{}'".format(checkpoint_file))
    else:
        print("==>Checkpoint not found: '{}'".format(checkpoint_file))
        cfg.START_EPOCH = 0  # set epoch to start training / val
    print("==>Model's start epoch:", cfg.START_EPOCH)
    print("==>Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in openkd_model.parameters() if p.requires_grad]):,}")

    if torch.cuda.is_available():
        openkd_model = openkd_model.cuda()
        loss_func = loss_func.cuda()

    print("==>Preparing data")
    episode_type = cfg.DATASET.EPISODE_TYPE  # "one_class", "mix_class"
    k_shot = cfg.TRAIN.NUM_TRAIN_SHOT
    m_query = cfg.TRAIN.NUM_TRAIN_QUERY
    k_shot_test = cfg.TEST.NUM_TEST_SHOT
    m_query_test = cfg.TEST.NUM_TEST_QUERY
    num_episodes = cfg.TRAIN.NUM_EPISODES  # -1 use traverse all episodes; >0 (e.g., 10,000), use fix number of episodes 
    num_episodes_test = cfg.TEST.NUM_EPISODES  # -1 use traverse all episodes; >0 (e.g., 1000), use fix number of episodes 

    # seen or unseen species, base or novel kps
    test_loader = build_episode_loader(cfg, k_shot_test, m_query_test, episode_type, num_episodes_test, phase='test')

    print('==>Final testing! Loading %s model'%cfg.LOAD_CHECKPOINT_TYPE)
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        openkd_model.load_state_dict(checkpoint['model'])
        print("==>Checkpoint loaded: '{}'".format(checkpoint_file))
    else:
        print("==>Checkpoint not found: '{}'".format(checkpoint_file))

    print('==>Test seen/unseen species, base/novel kps')
    eval_results = validate(cfg, openkd_model, test_loader)
    acc, ne = eval_results[0], eval_results[2]

if __name__ == '__main__':
    main()