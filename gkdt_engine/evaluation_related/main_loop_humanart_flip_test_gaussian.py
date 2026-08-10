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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# import torch.backends.cudnn as cudnn
# torch.autograd.set_detect_anomaly(True)
from tensorboardX import SummaryWriter

from config import update_config, create_loggers
from utils.utils import list2str, load_samples, make_grid_images, image_normalize, AverageMeter, summarize_losses
from evaluation_related.build_dataset_coco_eval import build_episode_loader, batch_data_to_device, parse_batch_data
from datasets.transforms import recover_kps
from datasets.dataset_utils import draw_instance, draw_skeletons, draw_markers
from network.gkd_model import get_gkd_model
from core.loss_lw import HeatmapLoss, DirectCoordLoss
from core.misc import compute_openkd_heatmap_loss, split_main_aux_heatmaps
from evaluation_metric import PCKMetric, NEMetric
from vis import show_save_episode, save_predictions, save_heatmaps

import misc

from network.mae_vit.vit import get_mae_vit_model
import cv2

#==========
#TODO
from evaluation_metric.write_prediction_to_json import init_coco_prediction_file,add_coco_predictions,save_coco_predictions
import json
#==========

def flip_back(flip_output, flip_pairs):
    # 创建副本用于交换操作
    flip_output_original = flip_output.clone()
    
    # 先交换对称关键点的位置
    for pair in flip_pairs:
        if pair[0] < flip_output.shape[1] and pair[1] < flip_output.shape[1]:
            # 交换对称关键点的热图
            flip_output_original[:, pair[0]] = flip_output[:, pair[1]]
            flip_output_original[:, pair[1]] = flip_output[:, pair[0]]
    
    # 再水平翻转热图
    flip_output_original = torch.flip(flip_output_original, dims=[-1])
    
    # 热图对齐校正（shift heatmap）
    # 由于下采样时的取整操作，翻转后可能会有1像素的偏移
    #flip_output_original[:, :, :, 1:] = flip_output_original[:, :, :, :-1]
    
    return flip_output_original


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords

def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        #print("heatmaps.shape:",heatmaps.shape)  #heatmaps.shape: (12, 17, 96, 96)
        preds, maxvals = _get_max_preds(heatmaps)

        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    return preds, maxvals

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
    pck_metric = PCKMetric(pck_thresholds=pck_thresh)  # Percentage of Correct Keypoints (PCK)
    ne_metric = NEMetric()  # Normalized Error

    #=========================================================
    #TODO
    # 定义文件路径
    original_gt_file = "/project/vonneumann1/cl2025/keypoint_datasets/human_pose/human_art/annotations/validation_humanart.json"
    output_pred_file = f"/project/vonneumann1/cl2025/GKD/experiments/evaluation_for_coco/humanart_result/main4_neg_samples_bilinear_tvm/output_{cfg.DATASET.TEST_DATA[0][1]}"
    print("output_pred_file:",output_pred_file)

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


        #print("queries.shape:",queries.shape)  #queries.shape: torch.Size([1, 12, 3, 384, 384])

        outputs = model(queries, supports, support_labels, support_kp_mask, obj_texts, obj_texts_mask, kps_texts, kps_texts_mask)
        predict_heatmaps_list = outputs[0]  # [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict
        
        # Access openkd_heatmap_fuse through DDP wrapper if needed
        model_module = model.module if hasattr(model, 'module') else model
        heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect = model_module.openkd_heatmap_fuse(
            predict_heatmaps_list[0],  # {'obj': tensor, 'text': tensor, 'image': tensor}
            support_kp_mask[0],
            kps_texts_mask[0]
        )

        # 主预测路径
        output_heatmap = heatmaps_fused.detach()    # B2 x N x h x w

        # 定义翻转对
        flip_pairs = [['1', '2'], ['3', '4'], ['5', '6'], ['7', '8'], ['9', '10'], ['11', '12'], ['13', '14'], ['15', '16']]

        # 将列表中的字符串转换为整数索引
        flip_pairs_indices = [(int(pair[0]), int(pair[1])) for pair in flip_pairs]\

        # 翻转测试增强
        if flip_pairs is not None:  # 执行翻转测试
            # 翻转输入图像
            queries_flipped = queries.flip(4)  # 水平翻转，假设第4个维度是宽度维度
            
            # 使用翻转后的输入进行推理
            outputs_flipped = model(queries_flipped, supports, support_labels, support_kp_mask, 
                                obj_texts, obj_texts_mask, kps_texts, kps_texts_mask)
            predict_heatmaps_list_flipped = outputs_flipped[0]
            
            # 融合翻转后的热图
            heatmaps_fused_flipped, fused_mask_sum_flipped, heatmaps_collect_flipped, masks_collect_flipped = model_module.openkd_heatmap_fuse(
                predict_heatmaps_list_flipped[0],
                support_kp_mask[0],
                kps_texts_mask[0]
            )
            heatmaps_predict_flipped = heatmaps_fused_flipped
            
            # 将翻转后的热图翻转回来，并进行对齐校正
            output_flipped_np = heatmaps_predict_flipped.detach()
            
            # 应用 flip_back
            output_flipped_heatmap = flip_back(output_flipped_np, flip_pairs_indices)
            
            # 融合原始预测和翻转预测
            output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

        # 最终的热图预测结果
        heatmaps_predict = output_heatmap

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
            # 转换为numpy数组
            heatmaps_np = heatmaps_predict.cpu().numpy()  # B2 x N x H x W
            
            # 准备center和scale参数 - 正确的计算方式
            center = np.zeros((B2, 2), dtype=np.float32)
            scale = np.zeros((B2, 2), dtype=np.float32)
            
            # 调用keypoints_from_heatmaps方法（注意：这个函数应该在循环外部调用）
            preds, maxvals = keypoints_from_heatmaps(
                heatmaps=heatmaps_np,
                center=center,
                scale=scale,
                unbiased=False,
                post_process='default',
                kernel=11,
                valid_radius_factor=0.0546875,
                use_udp=True,
                target_type='GaussianHeatmap')
                
            # 转换回tensor
            predictions = torch.from_numpy(preds).float().to(heatmaps_predict.device)
            predictions = (predictions / 96.0) * 2 - 1
            # 转换回tensor
            maxvals = torch.from_numpy(maxvals).float().to(heatmaps_predict.device)
                
        valid_kp_mask = valid_kp_mask.to(heatmaps_predict.device)
        query_labels  = query_labels.to(heatmaps_predict.device)

        predictions = predictions * valid_kp_mask.view(B2, N, 1)
        query_labels = query_labels * valid_kp_mask.view(B2, N, 1)

        predictions = predictions.cpu().detach()
        query_labels = query_labels.cpu().detach()
        valid_kp_mask = valid_kp_mask.cpu().detach()
        scale_trans_q, bbox_origin_q, w_h_origin_q = scale_trans_q.cpu(), bbox_origin_q.cpu(), w_h_origin_q.cpu()

        # square distance diff in original image scale
        predictions_o = recover_kps(predictions, square_image_length, scale_trans_q)
        query_labels_o = recover_kps(query_labels, square_image_length, scale_trans_q)

        #==========================================================
        # TODO: Add coco evaluation codes (For Yuxin). 
        # 使用独立函数生成COCO格式预测结果
        # 添加当前batch的预测结果
        add_coco_predictions(
            pred_data=pred_data,
            gkd_dataset=test_loader.dataset,
            episode_indexes=episode_indexes,
            predictions_o=predictions_o,
            original_gt_dict=original_gt_dict,
            maxvals=maxvals
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
    # Initialize distributed training (supports single GPU, single-node multi-GPU, multi-node multi-GPU)
    is_distributed, rank, world_size, local_rank = misc.init_distributed_mode()
    
    # Set device for current process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)  # Very important! As we may use torch.device('cuda') get current device!
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # Monkey patch print to only output on rank 0 in distributed mode
    misc.setup_print_for_distributed(is_distributed)

    cfg, args = update_config()
    # two ways to control cfg_str, 2023.08.15
    if len(cfg.AUTONAME.KEYS) == 0:  # way 1: determined by filename of cfg_file
        cfg_file_basename = os.path.basename(args.cfg_file)
        cfg_str = os.path.splitext(cfg_file_basename)[0]
    else:  # way 2: determined by autoname dict
        assert len(cfg.AUTONAME.LABELS) == len(cfg.AUTONAME.KEYS)
        cfg_str = ''
        for k, key_tmp in enumerate(cfg.AUTONAME.KEYS):
            label_tmp = cfg.AUTONAME.LABELS[k]
            value_tmp = eval('cfg.'+key_tmp)
            str_value = list2str(value_tmp) if isinstance(value_tmp, (list, tuple)) else str(value_tmp)
            if label_tmp == '':  # if label is an empty str, just continue
                continue
            cfg_str += (label_tmp+str_value)
    output_model_dir, logger, tb_writer = create_loggers(cfg, cfg_str, create_dir=misc.is_main_process(), 
                                        init_logger=misc.is_main_process() and cfg.LOGGER, init_tb=misc.is_main_process() and not cfg.EVAL_ONLY)
    print(cfg)  # or logger.info(cfg)
    print('==>cfg_str: {}'.format(cfg_str))

    manual_seed = cfg.MANUAL_SEED
    if manual_seed is not None:
        # cudnn.benchmark = False  # require to import torch.backends.cudnn as cudnn
        # cudnn.deterministic = True
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
    

    print("==>Preparing model, loss function & optimizer")
    openkd_model = get_gkd_model(cfg)
    if cfg.LOSS.TYPE in ['MSE', 'sigmoid-bce', 'cross-entropy']:  # supervised by GT heatmap
        loss_func = HeatmapLoss(cfg)
    elif cfg.LOSS.TYPE == 'direct_coord':  # supervised by GT keypoints
        loss_func = DirectCoordLoss(cfg)
    else:
        raise NotImplementedError
    
    # Move model and loss to device
    if torch.cuda.is_available():
        openkd_model = openkd_model.to(device)
        loss_func = loss_func.to(device)

    # Wrap model with DDP for distributed training
    if is_distributed:
        openkd_model = DDP(openkd_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = get_optimizer(cfg, openkd_model)

    checkpoint_file_wo_ext = os.path.join(output_model_dir, cfg_str)
    checkpoint_file = checkpoint_file_wo_ext + '.%s'%cfg.LOAD_CHECKPOINT_TYPE  # e.g., /output/openkd.best
    if cfg.RESUME and os.path.exists(checkpoint_file):
        # Load checkpoint to correct device
        dest_device_str = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_file, map_location=dest_device_str)  
        # Handle DDP wrapper when loading
        if hasattr(openkd_model, 'module'):
            openkd_model.module.load_state_dict(checkpoint['model'])
        else:
            openkd_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cfg.START_EPOCH = checkpoint['epoch'] + 1  # set epoch to start training / val
        print("==>Checkpoint loaded: '{}'".format(checkpoint_file))
    else:
        print("==>Checkpoint not found: '{}'".format(checkpoint_file))
        cfg.START_EPOCH = 0  # set epoch to start training / val
    print("==>Model's start epoch:", cfg.START_EPOCH)
    print("==>Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in openkd_model.parameters()]):,}")
    print("==>Learn parameters:", f"{np.sum([int(np.prod(p.shape)) if p.requires_grad else 0 for p in openkd_model.parameters()]):,}") 
    
    print("==>Preparing data")
    episode_type = cfg.DATASET.EPISODE_TYPE  # "one_class", "mix_class"
    k_shot = cfg.TRAIN.NUM_TRAIN_SHOT
    m_query = cfg.TRAIN.NUM_TRAIN_QUERY
    k_shot_test = cfg.TEST.NUM_TEST_SHOT
    m_query_test = cfg.TEST.NUM_TEST_QUERY
    num_episodes = cfg.TRAIN.NUM_EPISODES  # -1 use traverse all episodes; >0 (e.g., 10,000), use fix number of episodes 
    num_episodes_test = cfg.TEST.NUM_EPISODES  # -1 use traverse all episodes; >0 (e.g., 1000), use fix number of episodes 
    # seen or unseen species, base or novel kps
    test_loader = build_episode_loader(cfg, k_shot_test, m_query_test, episode_type, num_episodes_test, phase='test', is_distributed=is_distributed)
    

    print('==>Final testing! Loading %s model'%cfg.LOAD_CHECKPOINT_TYPE)
    if os.path.exists(checkpoint_file):
        dest_device_str = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_file, map_location=dest_device_str)
        # Handle DDP wrapper when loading
        if hasattr(openkd_model, 'module'):
            openkd_model.module.load_state_dict(checkpoint['model'])
        else:
            openkd_model.load_state_dict(checkpoint['model'])
        print("==>Checkpoint loaded: '{}'".format(checkpoint_file))
    else:
        print("==>Checkpoint not found: '{}'".format(checkpoint_file))

    # Access model's functions through DDP wrapper if needed
    model_module = openkd_model.module if hasattr(openkd_model, 'module') else openkd_model
    eval_cost = True
    model_module.set_cost_eval(eval_cost)

    print('==>Test seen/unseen species, base/novel kps')
    eval_results = validate(cfg, openkd_model, test_loader)
    acc, ne = eval_results[0], eval_results[2]

    cost_base = model_module.get_cost_eval()  # record results
    print('cost_base:', cost_base)
    avg_it_base = (cost_base['IT1'] + cost_base['IT2'] + cost_base['IT3'])
    print('Avg IT:  %.6f sec/episode'%(avg_it_base))
    model_module.set_cost_eval(eval_cost)

    # Clean up distributed process group
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()