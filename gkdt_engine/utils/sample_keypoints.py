import os
import numpy as np
import scipy.stats
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import json
import random
from einops import rearrange
import torch
import torch.nn as nn
from collections import namedtuple
from utils.utils import image_normalize
from datasets.dataset_utils import draw_markers

def sample_even_points(sampling_freq=12, is_cuda=False):
    '''
    sampling (sampling_freq*sampling_freq) points with each point (x, y) in range -1~1
    '''
    W = sampling_freq
    yy, xx = torch.meshgrid(torch.arange(0, W), torch.arange(0, W))
    grids = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1)  # (H*W) x 2
    grids = ((grids + 0.5) / W - 0.5) * 2  # -1 < x,y < 1
    if is_cuda:
        grids = grids.cuda()
    return grids  # (H*W) x 2

def sample_keypoints_within_rect(nkps=20, im_h=384, im_w=384, bbx=None, sample_method='random'):
    '''
    nkps: the keypoints to sample
    im_h, im_w: image height and width
    bbx: if not None, sample nkps within bbx (xmin, ymin, w, h)
    sample_method: 'random' or 'regular_grid'

    return grids: (nkps) x 2, each row is a point (x, y)
    '''
    if bbx is None:
        bbx = (0, 0, im_w, im_h)  # (xmin, ymin, w, h)
    xmin, ymin = bbx[0], bbx[1]
    w, h = bbx[2], bbx[3]
    if sample_method == 'random':
        x = np.random.randint(xmin, xmin+w, nkps)
        y = np.random.randint(ymin, ymin+h, nkps)
    elif sample_method == 'regular_grid':
        # from skimage.util import regular_grid
        yx_slices = regular_grid((h, w), nkps)
        y = np.array(range(h)[yx_slices[0]])
        x = np.array(range(w)[yx_slices[1]])
        y = y + ymin
        x = x + xmin

    x, y = x.reshape(nkps, 1), y.reshape(nkps, 1)
    grids = np.concatenate((x, y), axis=1)

    return grids

def sample_points_within_rect_but_distant_to_anchors(nkps=20, bbx=(0, 0, 384, 384), sample_method='random', anchors=None, anchor_mask=None, dist_thresh=10):
    '''
    Efficient point sampling approach based on stack.

    idea: firstly sample points within rect, secondly filter and maintain valid points, thirdly store the valid points
    into a stack until full.
    bbx (xmin, ymin, w, h)
    anchors: M x 2 anchor points, np.array, each row is a point (x, y)
    anchor_mask: M, np.array
    return: success flag & sampled keypoints
    '''
    N = 100 if nkps <= 20 else 5*nkps  # our strategy is to sample N pts first and then fileter nkps valid.
    # long_edge = bbx[2] if bbx[2] >= bbx[3] else bbx[3]
    # T = float(long_edge) * dist_thresh
    T = dist_thresh ** 2
    sample_success = False
    num_valid_anchors = sum(anchor_mask)
    if num_valid_anchors <= 0:  # no valid anchor points, simply sample points and return
        sampled_kps = sample_keypoints_within_rect(nkps, 0, 0, bbx, sample_method)
        sample_success = True
    else:
        # filter invalid anchor points
        anchor_mask = anchor_mask.astype(bool)
        anchors = anchors[anchor_mask]  # Assume M x 2
        M = num_valid_anchors
        sampled_kps = np.zeros((nkps, 2))  # define a stack
        stored_count = 0
        for i in range(50):  # try 50 times
            pts = sample_keypoints_within_rect(N, 0, 0, bbx, sample_method)
            dist_square = ((pts.reshape(-1, 1, 2) - anchors.reshape(1, -1, 2)) ** 2).sum(axis=2)  # N x M
            indicator = (dist_square >= T)  # N x M
            indicator = indicator.sum(axis=1) == M  # N
            num_valid_sampled_points = sum(indicator)  # 0~N
            if num_valid_sampled_points == 0:
                continue
            # num_valid_sampled_points > 0
            pts_filtered = pts[indicator]
            num_demand = nkps - stored_count
            if num_valid_sampled_points >= num_demand:  # if we have many valid sampled points, surpassing demand
                sampled_kps[stored_count:] = pts_filtered[0:num_demand]
                sample_success = True
                break
            else:  # no enough valid sampled points
                # store valid sampled points into stack, then go next sampling iteration
                sampled_kps[stored_count:stored_count+num_valid_sampled_points] = pts_filtered
                stored_count += num_valid_sampled_points

    return sample_success, sampled_kps

def sample_points_via_interpolate_but_distant_to_anchors(nkps=20, anchors=None, anchor_mask=None, dist_thresh=10):
    '''
    idea: Firstly extract valid anchor points given anchor mask. Then construct body part paths given valid points.
    Finally sample body part path and interpolate keypoint.
    anchors: M x 2 anchor points, np.array, each row is a point (x, y)
    anchor_mask: M
    return: success flag & sampled keypoints
    '''
    T = dist_thresh ** 2
    sample_success = False
    sampled_kps = np.zeros((nkps, 2))
    N_valid = int(sum(anchor_mask))
    if N_valid <= 1:
        return sample_success, sampled_kps
    # copy valid kps
    # valid_kps = np.zeros((N_valid, 2))
    # count = 0
    # for i in range(len(anchor_mask)):
    #     if anchor_mask[i] == 1:
    #         valid_kps[count] = anchors[i]
    #         count += 1
    anchor_mask_bool = anchor_mask.astype(bool)
    valid_kps = anchors[anchor_mask_bool]
    # construct paths
    body_part_paths = []
    for i in range(0, N_valid - 1, 1):
        for j in range(i + 1, N_valid, 1):
            body_part_paths.append([i, j])
    # sample path and interpolate
    sample_success = True
    for i in range(nkps):
        sample_each_pt_success = False
        for j in range(30):  # try 30 times
            path_ind = np.random.randint(0, len(body_part_paths))
            ind1, ind2 = body_part_paths[path_ind]
            pt1, pt2 = valid_kps[ind1], valid_kps[ind2]
            t = 0.5
            pt = t * pt1 + (1-t) * pt2
            dist1 = sum((pt - pt1) ** 2)
            dist2 = sum((pt - pt2) ** 2)
            if dist1 >= T and dist2 >= T:
                sampled_kps[i] = pt
                sample_each_pt_success = True
                break
            else:
                continue
        if sample_each_pt_success == False:
            sample_success = False
            break

    return sample_success, sampled_kps

# pytorch version
def get_bbx_from_kps_pytorch(kps, kps_mask, image_width=384, image_height=384):
    '''
    kps: N x 2, each row is (x, y) in image space
    kps_mask: N
    return a bbx (xmin, ymin, w, h)
    '''
    num_valid_kps = kps_mask.sum()
    if num_valid_kps <= 1:
        return (0, 0, image_width, image_height)

    kps_mask_bool = kps_mask.bool()
    valid_kps = kps[kps_mask_bool]
    kp_min = valid_kps.min(0)
    kp_max = valid_kps.max(0)
    x_min, y_min = max(kp_min[0], 0), max(kp_min[1], 0)
    x_max, y_max = min(kp_max[0], image_width-1), min(kp_max[1], image_height-1)
    w = (x_max - x_min) + 1
    h = (y_max - y_min) + 1
    if w <= 5 or h <= 5:  # bbx too small
        return (0, 0, image_width, image_height)

    # extending bbx by 15% in length
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    w *= 1.15
    h *= 1.15
    half_w, half_h = w / 2.0, h / 2.0
    x_min_new = max((center_x - half_w), 0)
    y_min_new = max((center_y - half_h), 0)
    x_max_new = min((center_x + half_w), image_width-1)
    y_max_new = min((center_y + half_h), image_height-1)
    w_new = x_max_new - x_min_new + 1
    h_new = y_max_new - y_min_new + 1

    return (int(x_min_new), int(y_min_new), int(w_new), int(h_new))

# numpy version
def get_bbx_from_kps_numpy(kps, kps_mask, image_width=384, image_height=384, bbx_extend_ratio=1.15):
    '''
    kps: N x 2, each row is (x, y) in image space
    kps_mask: N
    return a bbx (xmin, ymin, w, h)
    '''
    num_valid_kps = kps_mask.sum()
    if num_valid_kps <= 1:
        return (0, 0, image_width, image_height)

    kps_mask_bool = kps_mask.astype(bool)
    valid_kps = kps[kps_mask_bool]
    kp_min = valid_kps.min(0)  # 2
    kp_max = valid_kps.max(0)  # 2
    x_min, y_min = max(kp_min[0], 0), max(kp_min[1], 0)
    x_max, y_max = min(kp_max[0], image_width-1), min(kp_max[1], image_height-1)
    w = (x_max - x_min) + 1
    h = (y_max - y_min) + 1
    if w <= 30 or h <= 30:  # bbx too small
        return (0, 0, image_width, image_height)

    # extending bbx by 15% in length
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    w *= bbx_extend_ratio
    h *= bbx_extend_ratio
    half_w, half_h = w / 2.0, h / 2.0
    x_min_new = max((center_x - half_w), 0)
    y_min_new = max((center_y - half_h), 0)
    x_max_new = min((center_x + half_w), image_width-1)
    y_max_new = min((center_y + half_h), image_height-1)
    w_new = x_max_new - x_min_new + 1
    h_new = y_max_new - y_min_new + 1

    return (int(x_min_new), int(y_min_new), int(w_new), int(h_new))

# class version
class PointSampler(object):
    def __init__(self, type='use_bbx', num_kps=3, dist_thresh=10, image_width=384, image_height=384, bbx_extend_ratio=1.15):
        self.type = type
        self.num_kps = num_kps  # sampled kps per image
        self.dist_thresh = dist_thresh  # dist_thresh to anchors
        self.image_width = image_width
        self.image_height = image_height
        self.bbx_extend_ratio = bbx_extend_ratio

    def __call__(self, main_kps=None, main_kps_mask=None, **kwargs):
        '''
        main_kps (numpy): S x (B1+B2) x N x 2 in range -1~1
        main_kps_mask (numpy): S x (B1+B2) x N
        return sampled_kps: S x (B1+B2) x N_sampled x 2 in range -1~1
        '''
        main_kps = ((main_kps/2+0.5) * self.image_width).clamp(0, self.image_width-1).cpu().numpy()  # 0~image_width-1
        main_kps_mask = main_kps_mask.cpu().numpy()

        S, B_total, N = main_kps_mask.shape
        sampled_kps_list = []
        for s in range(S):
            for b in range(B_total):
                kps = main_kps[s, b]
                kps_mask = main_kps_mask[s, b]
                bbx = get_bbx_from_kps_numpy(kps, kps_mask, self.image_width, self.image_height, self.bbx_extend_ratio)
                if self.type == 'use_bbx':
                    sample_success_flag, sampled_kps = sample_points_within_rect_but_distant_to_anchors(
                        self.num_kps, bbx, 'random', kps, kps_mask, dist_thresh=self.dist_thresh
                    )
                elif self.type == 'use_itpl':
                    sample_success_flag, sampled_kps = sample_points_via_interpolate_but_distant_to_anchors(
                        self.num_kps, kps, kps_mask, dist_thresh=self.dist_thresh
                    )
                    if sample_success_flag == False:  # if fail, we try another method
                        sample_success_flag, sampled_kps = sample_points_within_rect_but_distant_to_anchors(
                            self.num_kps, bbx, 'random', kps, kps_mask, dist_thresh=self.dist_thresh
                        )
                else:
                    raise NotImplementedError
                if sample_success_flag == False:  # sample kps failed
                    # we can directly sample from whole image as last solution
                    bbx = (0, 0, self.image_width, self.image_height)
                    sample_success_flag, sampled_kps = sample_points_within_rect_but_distant_to_anchors(
                        self.num_kps, bbx, 'random', kps, kps_mask, dist_thresh=self.dist_thresh
                    )
                    # raise ValueError
                # np.random.shuffle(sampled_kps)
                sampled_kps_list.append(sampled_kps)

                # #-----------------------------------------------------
                # # show image below
                # episode_num = kwargs['episode_num']
                # origin_ims = (kwargs['ims']).permute(0, 1, 3, 4, 2).cpu().detach()  # S x B_total x C x H x W
                # image = origin_ims[s, b]  # H x W x C
                # pad_mask = torch.prod(torch.abs(image) < 0.01, dim=2).numpy().astype(np.uint8)
                # image = image_normalize(image, denormalize=True, copy=True)  # pixel value 0~1
                # image = image.mul(255).numpy().astype(np.uint8)  # pixel value 0~255
                # image *= (1 - pad_mask)[:,:,np.newaxis]  # set zero for padding area
                # image_bgr = image[:, :, [2,1,0]]
                # image_tmp = np.zeros(image_bgr.shape, dtype=np.uint8)
                # image_tmp[:,:,:] = image_bgr[:,:,:]
                # keypoint_dict = {i:sampled_kps[i] for i in range(len(sampled_kps))}
                # # image_tmp = draw_markers(image_tmp, keypoint_dict, marker='circle', color=[255, 255, 255], circle_radius=10, thickness=9)
                # new_im = draw_markers(image_tmp, keypoint_dict, marker='circle', color=[255,255,255], circle_radius=7, thickness=3)
                # keypoint_dict2 = {i:kps[i] for i in range(len(kps)) if kps_mask[i] == 1}
                # # new_im = draw_markers(new_im, keypoint_dict2, marker='circle', color=[255, 255, 255], circle_radius=10, thickness=9)
                # new_im = draw_markers(new_im, keypoint_dict2, marker='circle', color=[0,0,255], circle_radius=7, thickness=3)
                # # bbx has shape (xmin, ymin, w, h)
                # cv2.rectangle(new_im, (int(bbx[0]), int(bbx[1])),
                #       (int(bbx[0] + bbx[2]), int(bbx[1] + bbx[3])), (0, 255, 0), thickness=3)
                #
                # p_folder = './output/episode_images/preprocessed/sampled_neg_kps'
                # if os.path.exists(p_folder) == False:
                #     os.makedirs(p_folder)
                # if kwargs.get('episode_generator') == None:
                #     save_p = p_folder + '/e{}_s{}_{}.png'.format(episode_num, s, b)
                # else:
                #     episode_generator = kwargs['episode_generator']
                #     annos = episode_generator.supports + episode_generator.queries
                #     im_path = annos[b]  # assume S=1
                #     _, filename = os.path.split(im_path)
                #     filename_wo_ext, ext = os.path.splitext(filename)
                #     save_p = p_folder + '/e{}_s{}_{}_{}.png'.format(episode_num, s, b, filename_wo_ext)
                # cv2.imwrite(save_p, new_im)
                # # -----------------------------------------------------

        sampled_kps_all = np.stack(sampled_kps_list, axis=0)
        sampled_kps_all = sampled_kps_all.reshape((S, B_total, self.num_kps, 2))

        sampled_kps_all = (sampled_kps_all / self.image_width - 0.5) * 2
        sampled_kps_all = torch.tensor(sampled_kps_all).cuda()  # range -1~1

        return sampled_kps_all