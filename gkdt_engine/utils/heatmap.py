# coding=utf-8
# ------------------------------------------------------------------------------
# Copyright (c) Changsheng Lu
# Licensed under the MIT License.
# Written by Changsheng Lu (ChangshengLuu@gmail.com)
# ------------------------------------------------------------------------------

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""Implement the generation of every channel of ground truth heatmap.
:param centerA: int with shape (2,), every coordinate of person's keypoint (x, y) in image space. the sigma is also in image space.
:param grid_y & grid_x are heatmap's height and width (heatmap space)
:param stride = (image_height / grid_y) = (image_width / grid_x)
:param accumulate_confid_map: one channel of heatmap, which is accumulated, 
       1/100 is the smallest value of heatmap (namely log(1/100)=-d2/(2sigma^2)=-4.6052).   
       
Pay attention. For this function it has risk. The image length should be divisible by (upscaled) feature map length, i.e. grid_x * stride = image_length.
Otherwise the performance may be influenced.            
"""

def putGaussianMaps(center, sigma, grid_y, grid_x, stride, normalization=False, accumulate_confid_map=None, use_cuda=True):
    # center (x, y)
    # sigma
    start = stride / 2.0 - 0.5
    y_range = torch.Tensor([i for i in range(int(grid_y))])
    x_range = torch.Tensor([i for i in range(int(grid_x))])
    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
    # xx, yy = torch.meshgrid(x_range, y_range, indexing='xy')  # xy exchange
    if use_cuda and torch.cuda.is_available():
        xx, yy = xx.cuda(), yy.cuda()
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / (2.0 * sigma * sigma)
    mask = exponent <= 4.6052
    confid_map = torch.exp(-exponent)
    confid_map = torch.mul(mask, confid_map)

    norm_succ_flag = True
    if normalization == True:
        # print(torch.sum(confid_map).cpu().detach())
        if torch.sum(confid_map).cpu().detach().numpy() == 0:
            print(center.cpu().detach().numpy())  # this case may happen when center is out of window
            norm_succ_flag = False
            # exit(0)
        # print(torch.sum(confid_map).cpu().detach().numpy())
        confid_map = confid_map / torch.sum(confid_map).item()
    if accumulate_confid_map != None:
        accumulate_confid_map += confid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

        return accumulate_confid_map

    # map = confid_map.cpu().detach().numpy()
    # plt.imshow(map)
    # plt.show()

    return confid_map, norm_succ_flag

def putGaussianMaps2(center, sigma, h, w, prob_clip=True, normalization=False, accumulate_confid_map=None, use_cuda=True):
    '''
    generate gaussian map in heatmap space (center and sigma are subject to heatmap space instead of image space)
    center: [x, y]
    sigma : single-valued
    h     : height of gaussian map
    w     : width  of gaussian map
    prob_clip: clip if p<=0.01
    '''
    y_range = torch.arange(0, int(h), step=1)
    x_range = torch.arange(0, int(w), step=1)
    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
    # xx, yy = torch.meshgrid(x_range, y_range, indexing='xy')  # xy exchange
    if use_cuda and torch.cuda.is_available():
        xx, yy = xx.cuda(), yy.cuda()
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / (2.0 * sigma * sigma)
    confid_map = torch.exp(-exponent)
    if prob_clip:
        mask = exponent <= 4.6052
        confid_map = torch.mul(mask, confid_map)

    norm_succ_flag = True
    if normalization == True:
        sum_ = torch.sum(confid_map).item()
        if sum_ == 0:
            print(center.cpu().detach().numpy())  # this case may happen when center is out of window
            norm_succ_flag = False
            # exit(0)
        confid_map = confid_map / sum_
    if accumulate_confid_map != None:
        accumulate_confid_map += confid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

        return accumulate_confid_map

    # map = confid_map.cpu().detach().numpy()
    # plt.imshow(map)
    # plt.show()

    return confid_map, norm_succ_flag

def putAnisotropicGaussianMaps(center, Sigma, h, w, prob_clip=True, normalization=False, accumulate_confid_map=None, use_cuda=True):
    '''
    generate gaussian map in heatmap space (center and sigma are subject to heatmap space instead of image space)
    center: [x, y]
    Sigma : 2 x 2 covariance matrix
    h     : height of gaussian map
    w     : width  of gaussian map
    prob_clip: clip if p<=0.01
    '''
    y_range = torch.arange(0, int(h), step=1)
    x_range = torch.arange(0, int(w), step=1)
    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
    # xx, yy = torch.meshgrid(x_range, y_range, indexing='xy')  # xy exchange
    if use_cuda and torch.cuda.is_available():
        xx, yy = xx.cuda(), yy.cuda()

    det = (Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] ** 2) + 1e-6
    assert det > 0, 'det should be larger than 0, in putAnisotropicGaussianMaps'

    diff_x = xx - center[0]
    diff_y = yy - center[1]
    exponent = 0.5 * (Sigma[1, 1] * diff_x ** 2 - 2*Sigma[0, 1] * diff_x * diff_y + Sigma[0, 0] * diff_y ** 2) / det
    confid_map = torch.exp(-exponent)
    if prob_clip:
        mask = exponent <= 4.6052
        confid_map = torch.mul(mask, confid_map)

    norm_succ_flag = True
    if normalization == True:
        sum_ = torch.sum(confid_map).item()
        if sum_ == 0:
            print(center.cpu().detach().numpy())  # this case may happen when center is out of window
            norm_succ_flag = False
            # exit(0)
        confid_map = confid_map / sum_
    if accumulate_confid_map != None:
        accumulate_confid_map += confid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

        return accumulate_confid_map

    # map = confid_map.cpu().detach().numpy()
    # plt.imshow(map)
    # plt.show()

    return confid_map, norm_succ_flag



if __name__ == '__main__':
    # center = (64, 64)
    # sigma = 15
    # h = 128
    # w = 128
    # heatmap, _ = putGaussianMaps2(center, sigma, h, w, prob_clip=True, normalization=False, use_cuda=False)

    center = (64, 64)
    Sigma = torch.tensor([[50, -10], [-10, 10]])
    h = 128
    w = 128
    heatmap, _ = putAnisotropicGaussianMaps(center, Sigma, h, w, prob_clip=True, normalization=False, use_cuda=False)

    plt.imshow(heatmap.cpu().numpy())
    plt.show()
