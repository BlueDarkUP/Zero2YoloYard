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

# import sys
# sys.path.append('.')  # append pwd into system path so that it could find python modules
# print(os.getcwd())

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
from datasets.build_dataset import build_episode_loader, batch_data_to_device, parse_batch_data, mask_for_mix_modal_training, remove_pad_kps
from datasets.transforms import recover_kps
from datasets.dataset_utils import draw_instance, draw_skeletons, draw_markers
from network.gkd_model import get_gkd_model
from core.loss_lw import HeatmapLoss, HeatmapLossV2, DirectCoordLoss
from core.misc import compute_openkd_heatmap_loss, split_main_aux_heatmaps
from evaluation_metric import PCKMetric, NEMetric
from vis import show_save_episode, save_predictions, save_heatmaps

import misc

from network.mae_vit.vit import get_mae_vit_model

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

def train_one_epoch(cfg, model: torch.nn.Module, loss_func, optimizer: torch.optim.Optimizer, train_loader, test_loader_small=None, writer=None, epoch_i=0, **kwargs):
    # training setting related
    T1 = cfg.TRAIN.TEXT_PROMPT_SETTING.OBJ_TEXT  # number of text per object
    T2 = cfg.TRAIN.TEXT_PROMPT_SETTING.NUM_TEXT  # number of text per main keypoint
    B1 = cfg.TRAIN.NUM_TRAIN_SHOT
    B2 = cfg.TRAIN.NUM_TRAIN_QUERY
    num_roll_out = cfg.TRAIN.NUM_ROLL_OUT  # number of episodes per batch per GPU

    world_size = misc.get_world_size()

    # Set epoch for DistributedSampler to ensure different shuffle order each epoch
    if hasattr(train_loader.sampler, 'set_epoch') and cfg.DATASET.SAMPLING_STRATEGY.SHUFFLE:
        train_loader.sampler.set_epoch(epoch_i)
    
    model.train()
    small_scale_eval_interval = max(len(train_loader) // 5, 10)  # small-scale eval N times per epoch

    data_time = AverageMeter()
    iter_time = AverageMeter()
    end_time = time.time()
    print_frequency = cfg.PRINT_FREQUENCY if cfg.get('PRINT_FREQUENCY') is not None else 2
    for batch_i, batch_episodes_data in enumerate(train_loader):
        S = batch_episodes_data['kp_masks'].shape[0]
        data_time.update((time.time()-end_time) / S, n=S)
        
        # =====================================
        # TODO: small-scale evaluation during training phase
        # if (test_loader_small is not None) and (batch_i % small_scale_eval_interval == 0 and batch_i > 0):  # 500
        #     eval_results = validate(cfg, model, test_loader_small, **kwargs)
        #     recall, ne = eval_results[0], eval_results[2]
        #     if writer != None and misc.is_main_process():
        #         # recall is a list which is corresponding to different thresholds
        #         writer.add_scalar('accuracy_smallscale', recall[0], epoch_i*len(train_loader)+batch_i)
        # =====================================

        batch_episodes_data = batch_data_to_device(batch_episodes_data, is_gpu=True)

        (queries, query_labels, query_kp_mask, scale_trans_q, bbox_origin_q, w_h_origin_q), \
        (supports, support_labels, support_kp_mask, scale_trans_s, bbox_origin_s, w_h_origin_s), \
        (obj_texts, obj_texts_mask), \
        (kps_texts, kps_texts_mask), \
        (episode_indexes, num_kp_original) = parse_batch_data(batch_episodes_data, other_info=train_loader.dataset.other_info)

        # =====================================
        # make_grid_images(supports[0], denormalize=True, save_path='grid_image_s.jpg')
        # make_grid_images(queries[0], denormalize=True, save_path='grid_image_q.jpg')

        # show_save_episode(supports[0], support_labels[0], support_kp_mask[0], queries[0], query_labels[0], query_kp_mask[0], data_info=(train_loader, episode_indexes, num_kp_original), episode_num=episode_cnt_local, 
        #                   support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None, is_show=False, is_save=True, delete_old_files=False, save_root='output/episode_images')
        # =====================================        
        
        # Masking for mix-modal training 
        if cfg.TRAIN.MIX_MODAL_TRAINING.TYPE is not None:
            supports, support_kp_mask, kps_texts, kps_texts_mask = mask_for_mix_modal_training(cfg, supports, support_kp_mask, kps_texts, kps_texts_mask)
        
        end_time = time.time()
        N = query_kp_mask.shape[-1]  # S x B2 x N
        outputs = model(queries, supports, support_labels, support_kp_mask, obj_texts, obj_texts_mask, kps_texts, kps_texts_mask,
                        num_main_kps=N,                # used for domain alignment
                        query_kps_=query_labels,       # used for domain alignment
                        query_kp_mask_=query_kp_mask,  # used for domain alignment
                        )
        predict_heatmaps_list = outputs[0]  # [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict
        loss_v_t_align, loss_v_v_align, loss_t_t_align = outputs[1], outputs[2], outputs[3]  # a scalar
        loss_align = None
        loss_main = compute_openkd_heatmap_loss(cfg, model, loss_func, query_labels, query_kp_mask,
                                            predict_heatmaps_list, support_kp_mask, kps_texts_mask)
        loss_weights = cfg.LOSS.WEIGHT_LOSS
        loss = loss_main if (loss_align is None) else (loss_weights[0] * loss_main + loss_weights[2] * loss_align)      

        if np.isnan(loss.cpu().detach().numpy()):
            print('error nan in loss similarity')
            continue
        if np.isinf(loss.cpu().detach().numpy()):
            print('error inf in loss similarity')
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_time.update((time.time()-end_time)/S, n=S)

        if (print_frequency>0) and (batch_i % print_frequency == 0):
            msg = 'epoch: {}, batch: {} / {}, loss: {:.5f}'.format(epoch_i, batch_i, len(train_loader), loss.item())
            if loss_main is not None:
                msg += '/main: {:.5f}'.format(loss_main.item())
            if loss_align is not None:
                msg += ', align: {:.5f}'.format(loss_align.item())
            if loss_v_t_align is not None:
                msg += ', v_t: {:.5f}'.format(loss_v_t_align.item())
            if loss_v_v_align is not None:
                msg += ', v_v: {:.5f}'.format(loss_v_v_align.item())
            if loss_t_t_align is not None:
                msg += ', t_t: {:.5f}'.format(loss_t_t_align.item())
            # msg += ', time: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            msg += ', data time: {:.5f}, iter time: {:.5f}'.format(data_time.val, iter_time.val)
            print(msg)
            if writer != None and misc.is_main_process():
                writer.add_scalar('loss', loss.cpu().detach().numpy(), epoch_i*len(train_loader)+batch_i)

        end_time = time.time()
        
    stats = {
    }
    return stats

def train_episodes(cfg, model: torch.nn.Module, loss_func, optimizer: torch.optim.Optimizer, train_loader, test_loader, test_loader_small=None, writer=None, **kwargs):
    start_epoch = cfg.START_EPOCH
    end_epoch = cfg.TRAIN.NUM_EPOCH
    num_roll_out = cfg.TRAIN.NUM_ROLL_OUT  # number of episodes per batch per GPU
    other_info = train_loader.dataset.other_info  # a dict containing many other info for training, such as k_shot, m_query, etc.

    world_size = misc.get_world_size()
    epoch_time = AverageMeter()

    bs_total = num_roll_out * world_size
    print('==> is_distributed: {}, world_size: {}, num_rollout (train): {}, bs (all GPUs): {}'.format(misc.is_dist_avail_and_initialized(), world_size, num_roll_out, bs_total))

    for epoch_i in range(start_epoch, end_epoch):
        end_time = time.time()
        stats = train_one_epoch(cfg, model, loss_func, optimizer, train_loader, test_loader_small, writer, epoch_i, **kwargs)
        epoch_time.update((time.time()-end_time), n=1)
        msg = 'epoch time: {:.4f}s ({:.4f}), world_size: {}, num_rollout: {}, bs (all GPUs): {}'.format(epoch_time.val, epoch_time.avg, world_size, num_roll_out, bs_total)
        print(msg)

        # =====================================
        # TODO: After each epoch of training, a bit larger scale evaluation can be performed here
        if test_loader is not None:
            eval_results = validate(cfg, model, test_loader, **kwargs)
            recall, ne = eval_results[0], eval_results[2]
            if writer != None and misc.is_main_process():
                # recall is a list which is corresponding to different thresholds
                writer.add_scalar('accuracy', recall[0], epoch_i)
            # save models (only rank 0)
            if misc.is_main_process():
                recall_stack = kwargs['meter']['recall_stack']
                recall_best = kwargs['meter']['recall_best']
                ne_stack = kwargs['meter']['ne_stack']
                ne_best = kwargs['meter']['ne_best']
                recall_stack[0] = recall_stack[1]
                recall_stack[1] = recall[0]
                avg_recall = recall_stack[1]  # np.mean(recall_stack)  
                # Extract model state from DDP wrapper if needed
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                if avg_recall > recall_best:
                    checkpoint_path_best = kwargs['checkpoint_file_wo_ext'] + '.best'
                    misc.save_on_master({'model': model_state, 'optimizer': optimizer.state_dict(), \
                                        'epoch': epoch_i, 'other_info': other_info}, checkpoint_path_best)  # save best model
                    recall_best = avg_recall
                    print('==> Epoch: {}, BEST: {}'.format(epoch_i, recall_best))  
                print('==> Epoch: {}, Curr: {}'.format(epoch_i, avg_recall))   
                checkpoint_path_last = kwargs['checkpoint_file_wo_ext'] + '.last'
                misc.save_on_master({'model': model_state, 'optimizer': optimizer.state_dict(), \
                                    'epoch': epoch_i, 'other_info': other_info}, checkpoint_path_last)  # save last model
                kwargs['meter']['recall_stack'] = recall_stack
                kwargs['meter']['recall_best'] = recall_best        
        # =====================================

def validate(cfg, model, test_loader, **kwargs):
    print('==============testing start==============')
    model.eval()  # affects BN & disable Dropout
    torch.set_grad_enabled(False)  # disable grad computation

    square_image_length = cfg.DATASET.SQUARE_IMAGE_LENGTH  # 384

    # define evaluation metrics
    pck_thresh_type = 'bbox'  # 'bbox' or 'img'
    if pck_thresh_type == 'bbox':
        pck_thresh = cfg.TEST.PCK_THRESH if cfg.TEST.get('PCK_THRESH') is not None else [0.1]  # [0.1, 0.2]
    else:
        pck_thresh = [0.06]  # [0.06, 0.10], 0.06 * 384 = 23.04 pixels (23 pixels)
    pck_metric = PCKMetric(pck_thresholds=pck_thresh)  # Percentage of Correct Keypoints (PCK)
    ne_metric = NEMetric()  # Normalized Error

    episode_i = 0
    num_test_episodes = len(test_loader)  # in testing/val, we set to rollout one test episode per batch (see build_dataset.py)
    for batch_i, batch_episodes_data in enumerate(test_loader):
        batch_episodes_data = batch_data_to_device(batch_episodes_data, is_gpu=True)

        (queries, query_labels, query_kp_mask, scale_trans_q, bbox_origin_q, w_h_origin_q), \
        (supports, support_labels, support_kp_mask, scale_trans_s, bbox_origin_s, w_h_origin_s), \
        (obj_texts, obj_texts_mask), \
        (kps_texts, kps_texts_mask), \
        (episode_indexes, num_kp_original) = parse_batch_data(batch_episodes_data, other_info=test_loader.dataset.other_info)

        # =====================================
        # show_save_episode(supports[0], support_labels[0], support_kp_mask[0], queries[0], query_labels[0], query_kp_mask[0], data_info=(test_loader, episode_indexes, num_kp_original), episode_num=episode_i, 
        #                   support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None, is_show=False, is_save=True, delete_old_files=False, save_root='output/episode_images')
        # =====================================

        outputs = model(queries, supports, support_labels, support_kp_mask, obj_texts, obj_texts_mask, kps_texts, kps_texts_mask)
        predict_heatmaps_list = outputs[0]  # [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict
        
        # Access openkd_heatmap_fuse through DDP wrapper if needed
        model_module = model.module if hasattr(model, 'module') else model
        heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect = model_module.openkd_heatmap_fuse(
            predict_heatmaps_list[0],  # {'obj': tensor, 'text': tensor, 'image': tensor}
            support_kp_mask[0],
            kps_texts_mask[0]
        )
        heatmaps_predict = heatmaps_fused  # B2 x N x h x w

        # if we pad kps during testing/val (see cfg.DATASET.PAD_KPS), we can remove pad kps after inference
        if cfg.DATASET.PAD_KPS.SAME_AT_TEST == True:
            kps_texts, kps_texts_mask, _, data_N_second_dim, data_N_third_dim=remove_pad_kps(
                num_kp_original, kps_texts, kps_texts_mask, \
                data_N_second_dim=[heatmaps_predict, fused_mask_sum],
                data_N_third_dim=[query_labels, query_kp_mask, support_labels, support_kp_mask]
            )
            heatmaps_predict, fused_mask_sum = data_N_second_dim
            query_labels,query_kp_mask, support_labels, support_kp_mask = data_N_third_dim
        
        # multiple episode images into one (S=1)
        query_labels = query_labels[0]                                # B2 x N x 2
        valid_kp_mask = query_kp_mask[0] * (fused_mask_sum>0).long()  # B2 x N
        scale_trans_q = scale_trans_q[0]                      # B2 x 5
        bbox_origin_q = bbox_origin_q[0]                      # B2 x 4
        w_h_origin_q = w_h_origin_q[0]                        # B2 x 2

        # coordinates decoding
        B2, N = heatmaps_predict.shape[:2]
        if cfg.LOSS.TYPE == 'direct_coord':  # no need to decode
            predictions = heatmaps_predict
        else:
            H, W = heatmaps_predict.shape[2:]
            predict_score, predict_grids = torch.max(heatmaps_predict.reshape(B2, N, -1), 2)  # B2 x N
            predict_gridxy = torch.FloatTensor(B2, N, 2).to(heatmaps_predict.device)
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

        # # ----------------------------------------------------------------------------
        # # 1) save predicted keypoints
        # B1 = 0 if (supports is None) or (len(supports) == 0) or (len(supports[0].shape) != 4) else supports[0].shape[0]  # Judge whether it is zero-shot or not
        # supports = None if B1<=0 else supports.flatten(0, 1).cpu().detach()  # (S * B1) x C x H x W
        # support_labels = None if B1<=0 else support_labels.flatten(0, 1).cpu().detach()  # (S * B1) x N x 2
        # support_kp_mask = None if B1<=0 else support_kp_mask.flatten(0, 1).cpu().detach()  # (S * B1) x N
        # kps_texts = kps_texts[0]  # a list of an episode's texts (N_t*T2) or []
        # queries = queries.flatten(0, 1).cpu().detach()  # (S * B2) x C x H x W
        # query_kp_mask = query_kp_mask.flatten(0, 1).cpu().detach()  # (S * B1) x N
        
        # save_predictions(supports, support_labels, support_kp_mask, queries, query_labels, valid_kp_mask, predictions,
        #                  data_info=(test_loader, episode_indexes, num_kp_original), episode_num=episode_i, kp_texts=kps_texts, draw_limbs=False)
        
        # # 2) save multi-group heatmaps / fused heatmaps
        # predict_heatmaps_dict = predict_heatmaps_list[0]  # a dict: {'obj': [], 'text': [], 'image': []}
        # draw_multi_group_heatmaps = []  # model.dict2list(predict_heatmaps_dict)  # a list of G heatmaps, each is B2 x N x h x w
        # draw_fused_heatmaps = heatmaps_predict.cpu().detach()    # B2 x N x h x w
        # draw_multi_group_heatmaps = list(map(lambda x: x.cpu().detach(), draw_multi_group_heatmaps))
        
        # # ``queries" are query images, with size of B2 x C x H x W
        # save_heatmaps(queries, query_labels, valid_kp_mask, predictions, draw_multi_group_heatmaps, draw_fused_heatmaps,
        #               data_info=(test_loader, episode_indexes, num_kp_original), episode_num=episode_i, kp_texts=kps_texts)
        # # ----------------------------------------------------------------------------                

        # square distance diff in original image scale
        predictions_o = recover_kps(predictions, square_image_length, scale_trans_q)
        query_labels_o = recover_kps(query_labels, square_image_length, scale_trans_q)
        
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
                                        init_logger=misc.is_main_process() and cfg.LOGGER, init_tb=misc.is_main_process() and cfg.TB_WRITER and not cfg.EVAL_ONLY)
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
    

    # root = "/project/vonneumann1/cl2025/pretrained_models/mae_pretrained"
    # model = get_mae_vit_model(name='vit-h', root=root)

    # input_tensor = torch.rand(1, 3, 256, 192)

    # if torch.cuda.is_available():
    #     input_tensor = input_tensor.cuda()
    #     model = model.cuda()

    # output = model.forward(input_tensor)
    # print(output.shape)

    print("==>Preparing model, loss function & optimizer")
    openkd_model = get_gkd_model(cfg)
    if cfg.LOSS.TYPE in ['MSE', 'sigmoid-bce', 'cross-entropy']:  # supervised by GT heatmap
        loss_func = HeatmapLossV2(cfg)
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
    if not cfg.EVAL_ONLY:
        # seen species, base kps.
        train_loader = build_episode_loader(cfg, k_shot, m_query, episode_type, num_episodes, phase='train', is_distributed=is_distributed)
        # seen or unseen species, base or novel kps
        val_loader = build_episode_loader(cfg, k_shot_test, m_query_test, episode_type, num_episodes_test, phase='val', is_distributed=is_distributed)
        val_loader_small = build_episode_loader(cfg, k_shot_test, m_query_test, episode_type, 200, phase='val', is_distributed=is_distributed)  # small-scale eval during training
    # seen or unseen species, base or novel kps
    test_loader = build_episode_loader(cfg, k_shot_test, m_query_test, episode_type, num_episodes_test, phase='test', is_distributed=is_distributed)
    
    if not cfg.EVAL_ONLY:
        # train epochs
        meter = {
            'recall_stack': [0, 0],
            'recall_best': 0,
            'ne_stack': [1, 1],
            'ne_best': 1,
        }
        train_episodes(cfg, openkd_model, loss_func, optimizer,
                    train_loader, test_loader=val_loader, test_loader_small=val_loader_small, 
                    writer=tb_writer, checkpoint_file_wo_ext=checkpoint_file_wo_ext, meter=meter)
        print('==>Training finished!')

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
    print('cost:', cost_base)
    avg_it = cost_base['IT1']
    print('Avg IT:  %.6f sec/im'%(avg_it))

    # Clean up distributed process group
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()