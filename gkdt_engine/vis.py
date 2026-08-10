import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os
import cv2
import copy
import logging
from collections import OrderedDict
from utils.utils import image_normalize, show_cam_on_image, list2str
import textwrap

from datasets.dataset_utils import draw_skeletons, draw_markers, draw_ellipses
from datasets.build_dataset import get_info_from_batch_data

FILENAME_MAX_LENGTH=200

# data_info=(test_loader, episode_global_ids, num_kp_original)

# show the images in one episode
def  show_save_episode(supports, support_labels, support_kp_mask, queries, query_labels, query_kp_mask, data_info=[], episode_num=0,
                      support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None, is_show=False, is_save=True, delete_old_files=False, draw_main_kps=True, save_root='output/episode_images'):
    '''
    show the supervised keypoints in the support images and query images, as well as optionally drawing interpolated keypoints

    if support_aux_kps is not none, the image will draw interpolated keypoints.
    '''
    infos = get_info_from_batch_data(data_info)  
    ims_list, annos_list, support_kp_categories, KEYPOINT_TYPES = infos[0][0], infos[1][0], infos[2][0], infos[3][0]  # Only 1 episode per batch, so we index 0

    import copy
    if (supports is None) or (len(supports) == 0) or (len(supports.shape) !=4 ):
        supports = []
    else:
        supports, support_labels, support_kp_mask = copy.deepcopy(supports.detach().cpu()), copy.deepcopy(support_labels.detach().cpu()), copy.deepcopy(support_kp_mask.detach().cpu())
    queries, query_labels, query_kp_mask = copy.deepcopy(queries.detach().cpu()), copy.deepcopy(query_labels.detach().cpu()), copy.deepcopy(query_kp_mask.detach().cpu())

    # whether draw interpolated keypoints
    draw_interpolated_kps = False if (support_aux_kps is None) else True
    if draw_interpolated_kps == True:
        support_aux_kps, support_aux_kp_mask = copy.deepcopy(support_aux_kps.detach().cpu()), copy.deepcopy(support_aux_kp_mask.detach().cpu())
        query_aux_kps, query_aux_kp_mask = copy.deepcopy(query_aux_kps.detach().cpu()), copy.deepcopy(query_aux_kp_mask.detach().cpu())

    # grid_image = torchvision.utils.make_grid(supports, nrow=2, padding=2, pad_value=1)
    # grid_image = grid_image.permute(1,2,0)
    # plt.imshow(grid_image)
    # plt.show()

    if is_save:
        # save_image_root = './episode_images/preprocessed'
        save_image_root = os.path.join(save_root, 'preprocessed')

        if os.path.exists(save_image_root) == False:
            os.makedirs(save_image_root)
        if os.path.exists(save_image_root + "/" + 'support') == False:
            os.mkdir(save_image_root + "/" + 'support')
        if os.path.exists(save_image_root + "/" + 'query') == False:
            os.mkdir(save_image_root + "/" + 'query')

        # remove old episode images
        if delete_old_files == True:
            for each_file in os.listdir(os.path.join(save_image_root, 'support')):
                os.remove(os.path.join(save_image_root, 'support', each_file))
            for each_file in os.listdir(os.path.join(save_image_root, 'query')):
                os.remove(os.path.join(save_image_root, 'query', each_file))

    B1 = len(supports)
    B2 = queries.shape[0]
    width = queries.shape[-1]

    # ImageNet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # CLIP
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    r_pixels = 7  # 7 for most datasets; 2 for human_face_68 landmarks, w300
    for batch_i in range(B1):
        # support_image = supports.squeeze().permute(1, 2, 0)
        # query_image = queries.squeeze().permute(1, 2, 0)
        single_support_image, single_support_label = supports[batch_i, :, :, :], support_labels[batch_i, :, :]
        single_support_image = single_support_image.permute(1, 2, 0)

        # single_support_label = (single_support_label * (width-1)).long()
        single_support_label = ((single_support_label / 2 + 0.5) * (width - 1)).long()
        keypoints = {kp_type: single_support_label[i, :] for i, kp_type in enumerate(support_kp_categories) \
                     if support_kp_mask[batch_i, i] != 0 }
        print(ims_list[batch_i], annos_list[batch_i])
        print(keypoints)

        # support_image ranges 0~1
        pad_mask = torch.prod(torch.abs(single_support_image) < 0.01, dim=2).numpy().astype(np.uint8)
        for c in range(3):
            single_support_image[:, :, c] = (single_support_image[:, :, c] * std[c] + mean[c])
        im_uint8 = single_support_image.mul(255).numpy().astype(np.uint8)
        im_uint8 *= (1 - pad_mask).reshape(width, width, 1)  # set zero for padding area
        im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
        im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr
        # cv2.circle(im_uint8_bgr, (160, 169), 15, [255,255,255])
        # plt.imshow(im_uint8_bgr[:,:,::-1])
        # plt.show()

        if draw_main_kps:
            labeled_support_image = draw_markers(im_uint8_bgr, keypoints, marker='circle', color=[255,255,255], circle_radius=(r_pixels+3), thickness=-1)
            labeled_support_image = draw_skeletons(labeled_support_image, [keypoints], KEYPOINT_TYPES, circle_radius=r_pixels, limbs=[])
        else:
            labeled_support_image = im_uint8_bgr
        if draw_interpolated_kps == True:
            npimg_cur = np.copy(labeled_support_image)
            for j, is_visible in enumerate(support_aux_kp_mask[batch_i]):
                if is_visible == 0:
                    continue
                body_part = ((support_aux_kps[batch_i, j, :] / 2 + 0.5) * (width - 1)).long()
                center = (int(body_part[0]), int(body_part[1]))
                cv2.circle(npimg_cur, center, (int)(r_pixels/2), [0, 0, 255], thickness=-1)
                labeled_support_image = cv2.addWeighted(labeled_support_image, 0.3, npimg_cur, 0.7, 0)
        labeled_support_image = labeled_support_image[:, :, ::-1]  # bgr to rgb

        if is_show:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(single_support_image)
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(labeled_support_image)
            plt.show()

        if is_save:
            # write new episode images
            cv2.imwrite(os.path.join(save_image_root, 'support/eps{}_s_{}.jpg'.format(episode_num, batch_i)), labeled_support_image[:,:,::-1])

    for batch_i in range(B2):
        single_query_image, single_query_label = queries[batch_i, :, :, :], query_labels[batch_i, :, :]
        single_query_image = single_query_image.permute(1, 2, 0)

        single_query_label = ((single_query_label / 2 + 0.5) * (width - 1)).long()
        keypoints = {kp_type: single_query_label[i, :] for i, kp_type in
                     enumerate(support_kp_categories) if query_kp_mask[batch_i, i] != 0}
        print(ims_list[B1+batch_i], annos_list[B1+batch_i])
        # keypoints = {i: single_query_label[i] for i in range(len(single_query_label))}
        print(keypoints)

        pad_mask = torch.prod(torch.abs(single_query_image) < 0.01, dim=2).numpy().astype(np.uint8)
        for c in range(3):
            single_query_image[:, :, c] = (single_query_image[:, :, c] * std[c] + mean[c])
        im_uint8 = single_query_image.mul(255).numpy().astype(np.uint8)
        im_uint8 *= (1 - pad_mask).reshape(width, width, 1)  # set zero for padding area
        im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
        im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr
        if draw_main_kps:
            labeled_query_image = draw_markers(im_uint8_bgr, keypoints, marker='circle', color=[255,255,255], circle_radius=(r_pixels+3), thickness=-1)
            labeled_query_image = draw_skeletons(labeled_query_image, [keypoints], KEYPOINT_TYPES, circle_radius=r_pixels, limbs=[])
        else:
            labeled_query_image = im_uint8_bgr
        if draw_interpolated_kps == True:
            npimg_cur = np.copy(labeled_query_image)
            for j, is_visible in enumerate(query_aux_kp_mask[batch_i]):
                if is_visible == 0:
                    continue
                body_part = ((query_aux_kps[batch_i, j, :] / 2 + 0.5) * (width - 1)).long()
                center = (int(body_part[0]), int(body_part[1]))
                cv2.circle(npimg_cur, center, (int)(r_pixels / 2), [0, 0, 255], thickness=-1)
                labeled_query_image = cv2.addWeighted(labeled_query_image, 0.3, npimg_cur, 0.7, 0)
        labeled_query_image = labeled_query_image[:, :, ::-1]  # bgr to rgb

        if is_show:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(single_query_image)
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(labeled_query_image)

            plt.show()

        if is_save:
            # write new episode images
            cv2.imwrite(os.path.join(save_image_root, 'query/eps{}_q_{}.jpg'.format(episode_num, batch_i)), labeled_query_image[:,:,::-1])

def draw_line(im, kp_dict1, kp_dict2, color=[255, 255, 255], thickness=2):
    im = np.copy(im)
    for k, kp_type in enumerate(kp_dict1):
        kp1 = kp_dict1[kp_type]
        kp2 = kp_dict2[kp_type]
        x1, y1 = int(kp1[0]), int(kp1[1])
        x2, y2 = int(kp2[0]), int(kp2[1])
        cv2.line(im, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return im

def does_det_success(kp_list1, kp_list2, thresh=384*0.05, correct_criterion=1.0):  # 384*0.01
    thresh_sq = thresh ** 2
    flag = True
    incorrect_count = 0
    total_num = len(kp_list1)
    for i in range(total_num):
        kp1 = kp_list1[i]
        kp2 = kp_list2[i]
        d = sum((kp1 - kp2) ** 2)
        if d > thresh_sq:
            incorrect_count += 1
    if total_num == 0:
        return False
    if (total_num - incorrect_count) / total_num < correct_criterion:
        flag = False
    return flag

def extract_texts(texts, mask):
    '''
    :param texts: a list of N keypoint texts
    :param mask: N
    :return:
    '''
    kp_texts_str = []
    num = len(mask)
    for n in range(num):
        if mask[n] > 0:
            kp_texts_str.append(texts[n])
    return kp_texts_str

def im_denormalize_and_padding_area_masking(normalized_im, thresh=0.018, padding_color='black', param_type='CLIP'):
    # normalized_im: H x W x 3
    # denormalized image: H x W x 3
    L = normalized_im.shape[0]
    pad_mask = torch.prod(torch.abs(normalized_im) < thresh, dim=2).numpy().astype(np.uint8)
    im_tmp = image_normalize(normalized_im, denormalize=True, copy=True, param_type=param_type)
    im_uint8 = im_tmp.mul(255).numpy().astype(np.uint8)
    if padding_color == 'black':
        im_uint8 *= (1 - pad_mask).reshape(L, L, 1)  # set zero for padding area
    elif padding_color == 'white':
        im_uint8 *= (1 - pad_mask).reshape(L, L, 1)  # set zero for padding area
        im_uint8 += pad_mask.reshape(L, L, 1) * 255  # set 255
    else:
        raise NotImplementedError
    return im_uint8

# show support and query images; show predictions
def save_predictions(supports, support_labels, support_kp_mask, queries, query_labels, query_kp_mask, predictions, data_info=[], episode_num=0,
                     delete_old_files=False, save_root='output/episode_images/predictions', param_type='ImageNet', kp_texts=(), draw_limbs=False, **kwargs):
    infos = get_info_from_batch_data(data_info)
    ims_list, annos_list, support_kp_categories, KEYPOINT_TYPES, skeleton_list = infos[0][0], infos[1][0], infos[2][0], infos[3][0], infos[4][0]  # Only 1 episode per batch, so we index 0

    if draw_limbs == False:
        limbs = []
    else:
        # TODO later
        limbs = skeleton_list
    
    if supports is not None:
        supports, support_labels, support_kp_mask = copy.deepcopy(supports.detach().cpu()), copy.deepcopy(support_labels.detach().cpu()), copy.deepcopy(support_kp_mask.detach().cpu())
    queries, query_labels, query_kp_mask = copy.deepcopy(queries.detach().cpu()), copy.deepcopy(query_labels.detach().cpu()), copy.deepcopy(query_kp_mask.detach().cpu())
    predictions = copy.deepcopy(predictions.detach().cpu())

    support_root = os.path.join(save_root, 'support')
    query_root = os.path.join(save_root, 'query')
    query_im_only_root = os.path.join(save_root, 'query_im_only')
    query_prediction_root = os.path.join(save_root, 'query_prediction')
    query_prediction2_root = os.path.join(save_root, 'query_prediction2')  # diff to GT
    root_list = [support_root, query_root, query_im_only_root, query_prediction_root, query_prediction2_root]
    for per_root in root_list:
        if os.path.exists(per_root) == False:
            os.makedirs(per_root)
        else:
            # remove old episode images
            if delete_old_files == True:
                for each_file in os.listdir(per_root):
                    os.remove(os.path.join(per_root, each_file))

    B1 = supports.shape[0] if supports is not None else 0  # B1 x C x H x W or None
    B2 = queries.shape[0]   # B2 x C x H x W
    L = queries.shape[-1]  # W

    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])

    color_red, color_blue, color_white, color_black,color_gray = (0, 0, 255), (255, 0, 0), (255, 255, 255), (0,0,0), (100,100,100)
    color_pink = (255, 0, 255)
    dark_red = (0,0,232)
    light_green, cyan, orange = (128, 255,128), (255,255,0), (64,128,255)

    r_pixels = 2  # 7 for most datasets; 2 for human_face_68 landmarks, w300
    has_successful_det = False  # used for cherry-pick
    for b in range(B2):
        anno = annos_list[B1+b]
        im_id = anno['image_id']
        im_entry = ims_list[B1+b]
        im_path = im_entry['file_name']
        _, filename = os.path.split(im_path)
        filename_wo_ext, ext = os.path.splitext(filename)

        if len(kp_texts) == len(query_kp_mask[b]):  # prepend the texts to filename
            sub_kp_texts = extract_texts(kp_texts, query_kp_mask[b])
            kp_texts_str =  list2str(sub_kp_texts, link_str=', ')
            filename_wo_ext = f'({kp_texts_str})_{filename_wo_ext}'
        filename_wo_ext = textwrap.shorten(filename_wo_ext, width=FILENAME_MAX_LENGTH, placeholder='...')

        im_tmp, kp_tmp = queries[b], query_labels[b]
        im_tmp = im_tmp.permute(1, 2, 0)  # H x W x 3

        kp_tmp = ((kp_tmp / 2 + 0.5) * (L - 1)).long()
        kps_gt_dict = {kp_type: kp_tmp[n, :] for n, kp_type in enumerate(support_kp_categories) \
                     if query_kp_mask[b, n] > 0 }
        if len(kps_gt_dict) == 0:
            continue

        predict_kp_tmp = predictions[b]
        predict_kp_tmp = ((predict_kp_tmp / 2 + 0.5) * (L - 1)).long()
        kps_pred_dict = {kp_type: predict_kp_tmp[n, :] for n, kp_type in enumerate(support_kp_categories) \
                     if query_kp_mask[b, n] > 0 }

        success_det_flag = does_det_success(list(kps_pred_dict.values()), list(kps_gt_dict.values()))
        if success_det_flag == False:  # used for cherry-pick
            continue
        else:
            has_successful_det = True

        im_uint8 = im_denormalize_and_padding_area_masking(im_tmp, param_type=param_type)
        im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
        im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr

        p = os.path.join(query_im_only_root, 'e{}_q{}_{}.jpg'.format(episode_num, b, filename_wo_ext))
        cv2.imwrite(p, im_uint8_bgr)  # write im to query_im_only_root

        labeled_im = draw_markers(im_uint8_bgr, kps_gt_dict, marker='circle', color=[255,255,255], circle_radius=(r_pixels+3), thickness=-1)
        labeled_im = draw_skeletons(labeled_im, [kps_gt_dict], KEYPOINT_TYPES, circle_radius=r_pixels, limbs=limbs)
        p = os.path.join(query_root, 'e{}_q{}_{}.jpg'.format(episode_num, b, filename_wo_ext))
        cv2.imwrite(p, labeled_im)  # write im to query_root

        labeled_im = draw_markers(im_uint8_bgr, kps_pred_dict, marker='circle', color=[255,255,255], circle_radius=(r_pixels+3), thickness=-1)
        labeled_im = draw_skeletons(labeled_im, [kps_pred_dict], KEYPOINT_TYPES, circle_radius=r_pixels, limbs=limbs)
        p = os.path.join(query_prediction_root, 'e{}_q{}_{}.jpg'.format(episode_num, b, filename_wo_ext))
        cv2.imwrite(p, labeled_im)  # write im to query_prediction_root

        kps_gt_filtered = list(kps_gt_dict.values())
        kps_gt_filtered = list(map(lambda x: [float(x[0]), float(x[1])], kps_gt_filtered))
        kps_gt_filtered = np.array(kps_gt_filtered)
        ellipses_list = np.zeros((kps_gt_filtered.shape[0], 5))
        ellipses_list[:, :2] = kps_gt_filtered[:, :2]  # set centers
        ellipses_list[:, 2:4] = 384 * 0.1  # set semi-major/semi-minor axes
        ellipses_list[:, 4] = 0  # set angle
        labeled_im = draw_ellipses(im_uint8_bgr, ellipses_list, color=color_white, thickness=-1, alpha=0.45)
        # labeled_im = draw_ellipses(labeled_im, ellipses_list, color=color_white, thickness=2, alpha=1)
        labeled_im = draw_skeletons(labeled_im, [kps_gt_dict], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels+2, limbs=limbs)
        labeled_im = draw_skeletons(labeled_im, [kps_pred_dict], KEYPOINT_TYPES, circle_radius=r_pixels, limbs=limbs)
        labeled_im = draw_line(labeled_im, kps_pred_dict, kps_gt_dict)

        p = os.path.join(query_prediction2_root, 'e{}_q{}_{}.jpg'.format(episode_num, b, filename_wo_ext))
        cv2.imwrite(p, labeled_im)  # write im to query_prediction2_root

    if has_successful_det == False:
        return
    for b in range(B1):
        anno = annos_list[b]
        im_id = anno['image_id']
        im_entry = ims_list[b]
        im_path = im_entry['file_name']
        _, filename = os.path.split(im_path)
        filename_wo_ext, ext = os.path.splitext(filename)

        if len(kp_texts) == len(support_kp_mask[b]):  # prepend the texts to filename
            sub_kp_texts = extract_texts(kp_texts, support_kp_mask[b])
            kp_texts_str =  list2str(sub_kp_texts, link_str=', ')
            filename_wo_ext = f'({kp_texts_str})_{filename_wo_ext}'
        filename_wo_ext = textwrap.shorten(filename_wo_ext, width=FILENAME_MAX_LENGTH, placeholder='...')

        im_tmp, kp_tmp = supports[b], support_labels[b]
        im_tmp = im_tmp.permute(1, 2, 0)  # H x W x 3

        kp_tmp = ((kp_tmp / 2 + 0.5) * (L - 1)).long()
        keypoints = {kp_type: kp_tmp[n, :] for n, kp_type in enumerate(support_kp_categories) \
                     if support_kp_mask[b, n] > 0 }
        # print(episode_generator.supports[b])
        # print(keypoints)

        im_uint8 = im_denormalize_and_padding_area_masking(im_tmp, param_type=param_type)
        im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
        im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr

        labeled_im = draw_markers(im_uint8_bgr, keypoints, marker='circle', color=[255,255,255], circle_radius=(r_pixels+3), thickness=-1)
        labeled_im = draw_skeletons(labeled_im, [keypoints], KEYPOINT_TYPES, circle_radius=r_pixels, limbs=limbs)
        p = os.path.join(support_root, 'e{}_s{}_{}.jpg'.format(episode_num, b, filename_wo_ext))
        cv2.imwrite(p, labeled_im)  # write im to support_root


# save multi-group heatmaps & fused heatmaps
def save_heatmaps(queries, query_labels, query_kp_mask, predictions, multi_group_heatmaps, fused_heatmaps,
                  data_info=[], episode_num=0, save_root='output/episode_images/predictions',
                  param_type='ImageNet', kp_texts=(), **kwargs):
    infos = get_info_from_batch_data(data_info)
    ims_list, annos_list, support_kp_categories, KEYPOINT_TYPES = infos[0][0], infos[1][0], infos[2][0], infos[3][0]  # Only 1 episode per batch, so we index 0
    gkd_other_info = infos[4]  # {'k_shot': k_shot, 'm_query': m_query, 'phase': phase, ...}
    B1 = gkd_other_info['k_shot']
    

    # Note multi_group_heatmaps is a list.
    queries, query_labels, query_kp_mask = copy.deepcopy(queries.detach().cpu()), copy.deepcopy(query_labels.detach().cpu()), copy.deepcopy(query_kp_mask.detach().cpu())
    predictions = copy.deepcopy(predictions.detach().cpu())
    multi_group_heatmaps = copy.deepcopy(multi_group_heatmaps)  # a list of G heatmaps, each is (S * B2) x N x h x w
    fused_heatmaps = copy.deepcopy(fused_heatmaps)
    if fused_heatmaps is not None:
        multi_group_heatmaps.append(fused_heatmaps)  # appending fused_heatmaps
    G = len(multi_group_heatmaps)

    heatmaps_root = os.path.join(save_root, 'heatmaps')
    heatmaps_sep_root = os.path.join(save_root, 'heatmaps_sep')
    root_list = [heatmaps_root, heatmaps_sep_root]
    for per_root in root_list:
        if os.path.exists(per_root) == False:
            os.makedirs(per_root)

    B2 = queries.shape[0]  # B2 x C x H x W
    L = queries.shape[-1]  # W
    for b in range(B2):
        anno = annos_list[B1+b]
        im_id = anno['image_id']
        im_entry = ims_list[B1+b]
        im_path = im_entry['file_name']
        _, filename = os.path.split(im_path)
        filename_wo_ext, ext = os.path.splitext(filename)

        if len(kp_texts) == len(query_kp_mask[b]):  # prepend the texts to filename
            sub_kp_texts = extract_texts(kp_texts, query_kp_mask[b])
            kp_texts_str =  list2str(sub_kp_texts, link_str=', ')
            filename_wo_ext = f'({kp_texts_str})_{filename_wo_ext}'
        filename_wo_ext = textwrap.shorten(filename_wo_ext, width=FILENAME_MAX_LENGTH, placeholder='...')

        im_tmp, kp_tmp = queries[b], query_labels[b]
        im_tmp = im_tmp.permute(1, 2, 0)  # H x W x 3

        kp_tmp = ((kp_tmp / 2 + 0.5) * (L - 1)).long()
        kps_gt_dict = OrderedDict()
        for n, kp_type in enumerate(support_kp_categories):
            if query_kp_mask[b, n] > 0:
                kps_gt_dict[kp_type] = kp_tmp[n, :]
        if len(kps_gt_dict) == 0:
            continue

        predict_kp_tmp = predictions[b]
        predict_kp_tmp = ((predict_kp_tmp / 2 + 0.5) * (L - 1)).long()
        kps_pred_dict = OrderedDict()
        for n, kp_type in enumerate(support_kp_categories):
            if query_kp_mask[b, n] > 0:
                kps_pred_dict[kp_type] = predict_kp_tmp[n, :]

        valid_kps_types = list(kps_pred_dict.keys())

        success_det_flag = does_det_success(list(kps_pred_dict.values()), list(kps_gt_dict.values()))
        if success_det_flag == False:  # used for cherry-pick
            continue
        else:
            has_successful_det = True

        im_uint8 = im_denormalize_and_padding_area_masking(im_tmp, param_type=param_type)
        im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
        im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr

        padding_between_grid = 8
        padding_between_grid_color = 'white'  # 'black' or 'white'
        N_kps = len(kps_gt_dict)  # number of valid kps
        # image_grids = np.zeros((L * N_kps + (N_kps-1)*padding_between_grid, L * G + (G-1)*padding_between_grid, 3))
        image_grids = np.zeros((L * G + (G-1)*padding_between_grid, L * N_kps + (N_kps-1)*padding_between_grid, 3))
        if padding_between_grid_color != 'black':
            image_grids += 255
        for n in range(N_kps):
            kp_type = valid_kps_types[n]  # used to retrieve valid keypoint's heatmap
            kp_index = support_kp_categories.index(kp_type)
            for g in range(G):
                heatmap = multi_group_heatmaps[g][b, kp_index]  # h x w

                # TODO: for the modulated map we found: the more negative, the more correlated;
                # TODO: or, the more positive, the more correlated. This is depending on the learning (blackbox)
                if g != G-1:  # before G-1: modulated maps; G-1: fused heatmap
                    heatmap = heatmap  # heatmap or -heatmap
                heatmap = heatmap / heatmap.max()

                heatmap = heatmap.mul(255).clamp(0, 255).numpy()  # .byte()
                heatmap_resized = cv2.resize(heatmap, (L, L), interpolation=cv2.INTER_LINEAR)
                # cam = show_cam_on_image(im_uint8_bgr, heatmap_resized, save_path=None, mode='color')
                colormap = cv2.applyColorMap(np.uint8(heatmap_resized), cv2.COLORMAP_JET)
                alpha = 0.5
                cam = np.float32(colormap) * alpha + np.float32(im_uint8_bgr) * (1-alpha)
                cam = cam / np.max(cam)
                cam = np.uint8(255 * cam)

                h_begin = 0 if n==0 else (L+padding_between_grid) * n
                h_end   = L if n==0 else (L+padding_between_grid) * n + L
                w_begin = 0 if g==0 else (L+padding_between_grid) * g
                w_end   = L if g==0 else (L+padding_between_grid) * g + L
                # image_grids[h_begin:h_end, w_begin:w_end, :] = cam
                image_grids[w_begin:w_end, h_begin:h_end, :] = cam

                # save individual heatmap
                p = os.path.join(heatmaps_sep_root, 'e{}_q{}_n{}_{}.jpg'.format(episode_num, b, n, filename_wo_ext))
                cv2.imwrite(p, cam)  # write im to heatmaps_root

        # save heatmaps combined in image_grids
        p = os.path.join(heatmaps_root, 'e{}_q{}_{}.jpg'.format(episode_num, b, filename_wo_ext))
        cv2.imwrite(p, image_grids)  # write im to heatmaps_root
