import os
import json
import pickle
import numpy as np
import copy
from utils.utils import list_transpose

import torch
import torchvision.transforms as transforms
import datasets.transforms as mytransforms
from datasets.GenericDataset.generic_dataset import GKDDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.dataset_meta_info import get_dataset_splits, get_data_cfg, get_supercategory_to_dataset_inds
from evaluation_related.data_manager_coco_eval import DataManagerCOCOEval

from datasets.pycocotools.coco import COCO
from collections import OrderedDict
import random
import misc


def build_episode_loader(cfg, k_shot, m_query, episode_type='one_class', num_episodes=-1, phase='train', is_distributed=False, **kwargs):
    # phase: 'train', 'val' (seen or unseen species), 'test' (seen or unseen species)
    dataset_meta_info_dict, supercategory_list = get_dataset_splits(cfg, phase)

    sampling_strategy = 'squentially'  # 'squentially' or 'random'

    data_manager = DataManagerCOCOEval(dataset_meta_info_dict, supercategory_list)
    episode_list = data_manager.sampling_episodes(k_shot, m_query, episode_type, sampling_strategy, num_episodes)
    print('==> Total number of sampled episodes: %d (shot: %d, query: %d, phase: %s)' % (len(episode_list), k_shot, m_query, phase))
    
    #===================================================
    # TODO: build image transformations for preprocessing
    square_image_length = cfg.DATASET.SQUARE_IMAGE_LENGTH  # 448, 384, 256, 224, 192
    if phase == 'train':
        # FSOD: support size 320 x 320, query size 1000 x 600, encoded feature is about 1/16
        # Openpose: image size 368 x 368, confidence map size 46 x 46
        # Mask-RCNN:
        preprocess = mytransforms.Compose([
            # color transform
            # mytransforms.RandomApply(mytransforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), p=0.8),
            # mytransforms.RandomGrayscale(p=0.01),
            # geometry transform
            # mytransforms.RandomApply(mytransforms.HFlip(swap=dataset_meta['horizontal_swap_keypoints']), p=0.5),  # 0.5
            mytransforms.RandomApply(mytransforms.RandomRotation(max_rotate_degree=15), p=0.25),  # 0.25
            mytransforms.RelativeResize((0.75, 1.25)),
            mytransforms.RandomCrop(crop_gt_bbox=False),
            # mytransforms.RandomApply(mytransforms.RandomTranslation(), p=0.5),
            mytransforms.Resize(longer_length=square_image_length),  
            mytransforms.CenterPad(target_size=square_image_length),
            mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=False)
        ])
    else:  # 'test' or 'val'
        apply_bbox_crop = True
        if apply_bbox_crop:
            preprocess = mytransforms.Compose([
                mytransforms.RandomCrop(crop_gt_bbox=True),  # crop GT bbox directly without randomness
                mytransforms.Resize(longer_length=square_image_length),  
                mytransforms.CenterPad(target_size=square_image_length),
                mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
            ])
        else:
            preprocess = mytransforms.Compose([
                mytransforms.Resize(longer_length=square_image_length),  
                mytransforms.CenterPad(target_size=square_image_length),
                mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
            ])

    trunk = cfg.MODEL.ENCODER.TRUNK  # pre-trained model. Different pre-trained model has different normalized values.
    if (trunk in ['RESNET', 'DINOv3']) or (trunk=='CLIP' and str(cfg.MODEL.ENCODER.CLIP.VISUAL_ENCODER).startswith('mae-vit')):  # ImageNet's mean pixel values
        image_transform = transforms.Compose([
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomGrayscale(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif (trunk in ['CLIP', 'BLIP']):  # CLIP pre-trained model (OPENAI uses private dataset for training)
        image_transform = transforms.Compose([
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomGrayscale(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    else:
        raise NotImplementedError
    #===================================================
    
    if phase == 'train':
        padded_num_kps = cfg.DATASET.PAD_KPS.NUM_AT_TRAIN  # >0, enable padding kps in training 
        num_text_per_obj = cfg.TRAIN.TEXT_PROMPT_SETTING.OBJ_TEXT
        num_text_per_kp = cfg.TRAIN.TEXT_PROMPT_SETTING.NUM_TEXT
    else:  # 'val' or 'test'
        padded_num_kps = cfg.DATASET.PAD_KPS.NUM_AT_TRAIN if cfg.DATASET.PAD_KPS.SAME_AT_TEST else -1  # -1, no padding in testing/val
        num_text_per_obj = cfg.TEST.TEXT_PROMPT_SETTING.OBJ_TEXT
        num_text_per_kp = cfg.TEST.TEXT_PROMPT_SETTING.NUM_TEXT

    # Most important: 1. episode_list, 2. global_to_local_map, 3. dataset_meta_info_dict, 4. coco_objects
    other_info = {'k_shot': k_shot, 'm_query': m_query, 'phase': phase, 'episode_type': episode_type, 'padded_num_kps': padded_num_kps, \
                  'num_text_per_obj': num_text_per_obj, 'num_text_per_kp': num_text_per_kp}
    gkd_dataset = GKDDataset(
        episode_list,
        data_manager.global_to_local_map,
        dataset_meta_info_dict,
        data_manager.coco_objects,
        preprocess=preprocess,
        input_transform=image_transform,
        target_transform=None,
        other_info=other_info,
        )

    num_roll_out = cfg.TRAIN.NUM_ROLL_OUT if phase == 'train' else 1  # only 1 episode each time in testing/val
    gkd_dataloader = DataLoader(
        gkd_dataset, 
        batch_size=num_roll_out, 
        shuffle=(phase == 'train'),  
        num_workers=4,  # how many is better for H800? 4, 8, 16, 32.
        pin_memory=True, 
        drop_last=True
        )
    
    return gkd_dataloader

def batch_data_to_device(batch_episodes_data: dict, is_gpu=True, device=None):
    if is_gpu:
        if device is None:
            device = torch.device('cuda')  # After setting cuda device via torch.cuda.set_device(local_rank), torch.device(f'cuda:{local_rank}') and torch.device('cuda') and torch.cuda.current_device() point to the same device.
        for k, v in batch_episodes_data.items():
                if torch.is_tensor(v):
                    batch_episodes_data[k] = v.to(device)
    return batch_episodes_data

def parse_batch_data(batch_episodes_data: dict, other_info: dict):
    images = batch_episodes_data['images']  # S x (B1+B2) x 3 x H x W
    labels = batch_episodes_data['labels']  # S x (B1+B2) x N x 2
    kp_masks = batch_episodes_data['kp_masks']  # S x (B1+B2) x N
    scale_trans = batch_episodes_data['scale_trans']  # S x (B1+B2) x 5
    bbox_origin = batch_episodes_data['bbox_origin']  # S x (B1+B2) x 4
    w_h_origin = batch_episodes_data['w_h_origin']  # S x (B1+B2) x 2

    episode_index = batch_episodes_data['episode_index']  # S, a list of episode indexes for GKDDataset.episode_list
    num_kp_original = batch_episodes_data['num_kp_original']  # S x (B1+B2)

    obj_texts = batch_episodes_data['obj_texts']  # T1 x S, a list of T1 lists, each sublist contains S episodes' object texts
    obj_texts_mask = batch_episodes_data['obj_texts_mask']  # S x T1
    kps_texts = batch_episodes_data['kps_texts']  # (N*T2) x S, a list of (N_final*T2) lists, each sublist contains S episodes' kps texts
    kps_texts_mask = batch_episodes_data['kps_texts_mask']  # S x N x T2

    B1 = other_info['k_shot']
    B2 = other_info['m_query']
    T1 = other_info['num_text_per_obj']
    T2 = other_info['num_text_per_kp']

    supports = images[:, :B1]  # S x B1 x 3 x H x W
    queries = images[:, B1:]   # S x B2 x 3 x H x W
    support_labels = (labels[:, :B1]).float()  # S x B1 x N x 2
    query_labels = (labels[:, B1:]).float()    # S x B2 x N x 2
    support_kp_mask = kp_masks[:, :B1]  # S x B1 x N
    query_kp_mask = kp_masks[:, B1:]    # S x B2 x N
    scale_trans_s = scale_trans[:, :B1]  # S x B1 x 5
    scale_trans_q = scale_trans[:, B1:]  # S x B2 x 5
    bbox_origin_s = bbox_origin[:, :B1]  # S x B1 x 4
    bbox_origin_q = bbox_origin[:, B1:]  # S x B2 x 4
    w_h_origin_s = w_h_origin[:, :B1]  # S x B1 x 2
    w_h_origin_q = w_h_origin[:, B1:]  # S x B2 x 2

    _obj_texts = list_transpose(obj_texts)  # S x T1, a list of S lists, each sublist contains an episode's T1 object texts
    _kps_texts = list_transpose(kps_texts)  # S x (N*T2), a list of S lists, each sublist contains an episode's (N*T2) kps texts

    return (queries, query_labels, query_kp_mask, scale_trans_q, bbox_origin_q, w_h_origin_q), \
    (supports, support_labels, support_kp_mask, scale_trans_s, bbox_origin_s, w_h_origin_s), \
    (_obj_texts, obj_texts_mask), \
    (_kps_texts, kps_texts_mask), \
    (episode_index, num_kp_original)

def mask_for_mix_modal_training(cfg, supports, support_kp_mask, kps_texts, kps_texts_mask):
    _type = cfg.TRAIN.MIX_MODAL_TRAINING.TYPE
    _range = cfg.TRAIN.MIX_MODAL_TRAINING.RANGE
    assert cfg.TRAIN.TEXT_PROMPT_SETTING.NUM_TEXT >= 1 and cfg.TRAIN.NUM_TRAIN_SHOT >= 1
    assert len(_range) <= 3  # 'tvm'
    S = support_kp_mask.shape[0]
    device = support_kp_mask.device
    dtype = support_kp_mask.dtype
    # 1) random sampling ids
    if _type == 'batch':
        ids = np.random.randint(0, len(_range), 1)
    else:
        ids = np.random.randint(0, len(_range), S)
    ids = torch.tensor(ids).reshape(-1, 1, 1).to(device=device)  # 1 x 1 x 1 or S x 1 x 1 
    # 2) construct t&v masks and 3) masking
    if 't' in _range:
        t_idx = _range.index('t')
        mask = (ids != t_idx).to(dtype=dtype)
        support_kp_mask = support_kp_mask * mask  # mask visual prompts if 't' is selected
    if 'v' in _range:
        v_idx = _range.index('v')
        mask = (ids != v_idx).to(dtype=dtype)
        kps_texts_mask = kps_texts_mask * mask  # mask text prompts if 'v' is selected
    return supports, support_kp_mask, kps_texts, kps_texts_mask

def remove_pad_kps(num_kp_original, kps_texts, kps_texts_mask, data_N_first_dim=[], data_N_second_dim=[], data_N_third_dim=[]):
    '''
    Docstring for remove_pad_kps. Note S=1 (only process one episode each time)
    
    :param num_kp_original: S x (B1+B2)
    :param kps_texts: S x (N_final*T2), a list of S lists, each sublist contains an episode's (N_final*T2) kps texts
    :param kps_texts_mask:  S x N_final x T2
    :param data_N_first_dim: a list of data with N in the first dim, each data is (N x ...)
    :param data_N_second_dim: a list of data with N in the second dim, each data is (... x N x ...)
    :param data_N_third_dim: a list of data with N in the third dim
    '''
    N_real = num_kp_original[0, -1].item()  # since all samples in the episode have same num_kp_original, we can just take the first one
    _, N_final, T2 = kps_texts_mask.shape
    assert N_real <= N_final
    
    if T2>=1:  # in case no kps_texts, i.e., kps_texts_mask.shape is S x N_final x 0
        _kps_texts = []
        for i in range(N_real):
            for j in range(T2):
                _kps_texts.append(kps_texts[0][i*T2+j])
        kps_texts[0] = _kps_texts
        kps_texts_mask = kps_texts_mask[:, :N_real]

    _data_N_first_dim = []
    for data in data_N_first_dim:
        assert data.shape[0] == N_final, 'The first dim of data should be N_final.'
        _data = data[:N_real]
        _data_N_first_dim.append(_data)

    _data_N_second_dim = []
    for data in data_N_second_dim:
        assert data.shape[1] == N_final, 'The second dim of data should be N_final.'
        _data = data[:, :N_real]
        _data_N_second_dim.append(_data)

    _data_N_third_dim = []
    for data in data_N_third_dim:
        assert data.shape[2] == N_final, 'The first dim of data should be N_final.'
        _data = data[:, :, :N_real]
        _data_N_third_dim.append(_data)

    return kps_texts, kps_texts_mask, _data_N_first_dim, _data_N_second_dim, _data_N_third_dim
    
def get_info_from_batch_data(data_info: list):
    data_loader, episode_indexes, num_kp_original = data_info[0], data_info[1], data_info[2]
    gkd_dataset = data_loader.dataset
    gkd_other_info = data_loader.dataset.other_info

    annos_list = []
    ims_list = []
    kp_types_list = []
    full_kp_types_list = []
    skeleton_list=[]
    for batch_i, per_episode_index in enumerate(episode_indexes):  # a batch of episode indices
        one_episode_global_ids = gkd_dataset.episode_list[per_episode_index]
        annos_list.append([])
        ims_list.append([])
        for one_global_id in one_episode_global_ids:
            id_map_entry = gkd_dataset.global_to_local_id_map[one_global_id]
            dataset_id, local_anno_id = id_map_entry['dataset_id'], id_map_entry['local_id']
            
            dataset_meta_info = gkd_dataset.dataset_meta_info[dataset_id]
            cocoGT = gkd_dataset.coco_objects[dataset_id]
            anno_sample = cocoGT.anns[local_anno_id]

            category_id = anno_sample['category_id']
            category_entry = cocoGT.loadCats(category_id)[0]
            FULL_SET_KEYPOINT_TYPES = category_entry['keypoints']

            kps_classes_input_tmp = dataset_meta_info['kps_classes']  # if None, use all kps in FULL_SET_KEYPOINT_TYPES
            if kps_classes_input_tmp in [None, []]:
                support_kp_categories = FULL_SET_KEYPOINT_TYPES
            else:
                support_kp_categories = kps_classes_input_tmp  
            num_kp_categories = len(support_kp_categories)  # N_original

            image_id = anno_sample['image_id']
            image_entry = cocoGT.loadImgs(image_id)[0]

            annos_list[batch_i].append(anno_sample)
            ims_list[batch_i].append(image_entry)
            skeleton_list[batch_i].append(category_entry['skeleton'])
        kp_types_list.append(support_kp_categories)  # since each episode has same support_kp_categories, we record once
        full_kp_types_list.append(FULL_SET_KEYPOINT_TYPES)

    infos = (ims_list, annos_list, kp_types_list, full_kp_types_list, gkd_other_info, skeleton_list)
    return infos

    

if __name__ == '__main__':
    f=json.load(open('/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json', 'r'))
    f=json.load(open('/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json', 'r'))
        





