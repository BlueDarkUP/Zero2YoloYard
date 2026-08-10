import os
import argparse
from yacs.config import CfgNode
import numpy as np

import torch
import torchvision.transforms as transforms
from utils.utils import make_grid_images

import copy
import cv2
from PIL import Image
import test_real_world.gkd_inference_lib.transforms as mytransforms
from test_real_world.gkd_inference_lib.gkd_model import get_gkd_model

#======================================================================
# Single-object General Keypoint Detection for Real-world Applications
# Paper: https://arxiv.org/abs/2607.00752
# Date: 2026.07.18
#======================================================================

def pad_kps(max_kps, support_kps, support_kp_mask, kps_texts, kps_texts_mask):
    '''
    Pad keypoints to max_kps
    support_kps: S x B1 x N_v x 2, ranges -1~1 (continuous) or None
    support_kp_mask: S x B1 x N_v or None
    kps_texts: a list with S lists where each is an episode's N_t kps texts (i.e. S x (N_t*T2)) or []
    kps_texts_mask: tensor S x N_t x T2, or None    

    output arguments:
    support_kps: S x B1 x max_kps x 2, ranges -1~1 (continuous) or None
    support_kp_mask: S x B1 x max_kps or None
    kps_texts: S x (max_kps*T2)) or []
    kps_texts_mask: S x max_kps x T2
    '''
    N_v = support_kp_mask.shape[2] if support_kp_mask is not None else 0
    N_t = kps_texts_mask.shape[1] if kps_texts_mask is not None else 0
    assert (N_v <= max_kps) and (N_t <= max_kps), 'Number of visual/text prompted keypoints should be <= max_kps'
    if (N_v > 0) and (N_v < max_kps):
        S, B, _ = support_kp_mask.shape
        support_kps = torch.cat([support_kps, torch.zeros((S, B, max_kps - N_v, 2), dtype=support_kps.dtype).to(support_kps.device)], dim=2)
        support_kp_mask = torch.cat([support_kp_mask, torch.zeros((S, B, max_kps - N_v), dtype=support_kp_mask.dtype).to(support_kp_mask.device)], dim=2)
    if (N_t > 0) and (N_t < max_kps):
        S, _, T2 = kps_texts_mask.shape
        kps_texts_mask = torch.cat([kps_texts_mask, torch.zeros((S, max_kps - N_t, T2), dtype=kps_texts_mask.dtype).to(kps_texts_mask.device)], dim=1)
        for s in range(S):
            kps_texts[s] += [''] * (max_kps - N_t)*T2  # pad with empty strings
    return support_kps, support_kp_mask, kps_texts, kps_texts_mask

def remove_pad_kps(num_kp_origin, max_kps, kps_texts=None, kps_texts_mask=None, data_N_second_dim=[], data_N_third_dim=[]):
    '''
    Remove padded keypoints after inference
    num_kp_origin: the original number of keypoints before padding
    max_kps: the maximum number of keypoints after padding
    kps_texts: a list with S lists where each is an episode's N_t kps texts (i.e. S x (N_t*T2)) or []
    kps_texts_mask: tensor S x N_t x T2, or None
    data_N_second_dim: a list of data with N in the second dim, each data is (... x N x ...)
    data_N_third_dim: a list of data with N in the third dim
    '''
    if kps_texts_mask is not None:
        T2 = kps_texts_mask.shape[2]
        kps_texts[0] = kps_texts[0][:num_kp_origin*T2]  # remove pad kps texts
        kps_texts_mask = kps_texts_mask[:, :num_kp_origin, :]  # remove pad kps texts mask

    _data_N_second_dim = []
    for data in data_N_second_dim:
        assert data.shape[1] == max_kps, 'The second dim of data should be N_final.'
        _data = data[:, :num_kp_origin]
        _data_N_second_dim.append(_data)

    _data_N_third_dim = []
    for data in data_N_third_dim:
        assert data.shape[2] == max_kps, 'The third dim of data should be N_final.'
        _data = data[:, :, :num_kp_origin]
        _data_N_third_dim.append(_data)

    return kps_texts, kps_texts_mask, _data_N_second_dim, _data_N_third_dim

class GKDInference(object):
    def __init__(self, cfg_file, checkpoint_path, opts=None):
        self.cfg_file = cfg_file
        self.checkpoint_path = checkpoint_path
        self.opts = opts

        # Config
        self.cfg = CfgNode.load_cfg(open(cfg_file))
        if opts is not None:
            self.cfg.merge_from_list(opts)
        print(self.cfg)

        # Load model
        self.gkd_model = get_gkd_model(self.cfg)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')  
            self.gkd_model.load_state_dict(checkpoint['model'])
            del checkpoint
            print("==>Checkpoint loaded: '{}'".format(checkpoint_path))
        else:
            print("==>Checkpoint not found: '{}'".format(checkpoint_path))
            exit(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gkd_model = self.gkd_model.to(self.device)

        self.square_image_length = self.cfg.DATASET.SQUARE_IMAGE_LENGTH  # 384
        
    def vanilla_detect(self, queries, supports=None, support_kps=None, support_kp_mask=None, kps_texts=[], kps_texts_mask=None): 
        '''
        Vanilla yet advanced GKD detection interface for multiple episodes (S>=1)!

        ***input arguments***
        supports (image): S x B1 x 3 x H x W (S episodes' support images) or None
        queries (image):  S x B2 x 3 x H x W (S episodes' query images)
        support_kps: S x B1 x N_v x 2, ranges -1~1 (continuous) or None
        support_kp_mask: S x B1 x N_v or None

        kps_texts: a list with S lists where each is an episode's N_t kps texts (i.e. S x (N_t*T2)) or []
        kps_texts_mask: tensor S x N_t x T2, or None

        ***output arguments***
        predictions: B2 x N x 2 (each row is a point (x, y) in -1~1)
        predict_score: B2 x N (0~about 1)   
        '''
        N_v = support_kp_mask.shape[2] if support_kp_mask is not None else 0
        N_t = kps_texts_mask.shape[1] if kps_texts_mask is not None else 0
        assert (N_t > 0) or (N_v > 0), 'At least text prompt or visual prompt is provided.'
        if (N_t > 0) and (N_v > 0):
            assert N_v == N_t, 'The number of visual prompts and textual prompts should be the same.'
        N_origin = N_v if N_v > 0 else N_t

        # if we pad kps during testing/val (see cfg.DATASET.PAD_KPS)
        if self.cfg.DATASET.PAD_KPS.SAME_AT_TEST == True:  
            max_kps = self.cfg.DATASET.PAD_KPS.NUM_AT_TRAIN
            support_kps, support_kp_mask, kps_texts, kps_texts_mask = pad_kps(max_kps, support_kps, support_kp_mask, kps_texts, kps_texts_mask)

        # GKD inference
        outputs = self.gkd_model(queries, supports, support_kps, support_kp_mask, kps_texts_=kps_texts, kps_texts_mask_=kps_texts_mask)
        predict_heatmaps_list = outputs[0]  # [{'text': tensor, 'image': tensor}, ...], a list of dict
        heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect = self.gkd_model.openkd_heatmap_fuse(
            predict_heatmaps_list[0],  # {'text': tensor, 'image': tensor}
            support_kp_mask=support_kp_mask[0] if (support_kp_mask is not None) else None,
            kps_texts_mask=kps_texts_mask[0] if (kps_texts_mask is not None) else None
        )
        heatmaps_predict = heatmaps_fused  # B2 x N x h x w      

        # if we pad kps during testing/val (see cfg.DATASET.PAD_KPS), we can remove pad kps after inference
        if self.cfg.DATASET.PAD_KPS.SAME_AT_TEST == True:
            # do remove pad kps here
            kps_texts, kps_texts_mask, data_N_second_dim, _=remove_pad_kps(
                N_origin, max_kps, kps_texts, kps_texts_mask, \
                data_N_second_dim=[heatmaps_predict, fused_mask_sum]
                )
            heatmaps_predict, fused_mask_sum = data_N_second_dim
        
        valid_prompt_mask = (fused_mask_sum > 0).long()  # 1 x N, mask for valid fused V & T prompts

        # coordinates decoding
        B2, _, H, W = heatmaps_predict.shape
        predict_score, predict_grids = torch.max(heatmaps_predict.reshape(B2, N_origin, -1), 2)  # B2 x N
        predict_gridxy = torch.FloatTensor(B2, N_origin, 2).to(heatmaps_predict.device)
        predict_gridxy[:, :, 0] = predict_grids % W  # grid x
        predict_gridxy[:, :, 1] = predict_grids // H  # grid y
        # 'MSE', 'cross-entropy'
        predictions = ((predict_gridxy + 0.5) / H - 0.5) * 2  # -1~1

        # set detected kps and scores induced by invalid prompts to 0
        predictions = predictions * valid_prompt_mask.view(1, N_origin, 1)
        predict_score = predict_score * valid_prompt_mask

        return predictions, predict_score

    def supplement_mask_for_support_kps(self, support_kps=None):
        '''
        Assume the given visual keypoints are valid. We construct the valid kp mask for them.
        support_kps: B1 x N_v x 2 or None
        '''
        if support_kps is not None:
            support_kp_mask = torch.ones(support_kps.shape[:-1], dtype=torch.long)  # B1 x N_v
        else:
            support_kp_mask = None
        return support_kp_mask  # B1 x N_v or None

    def supplement_mask_for_kps_texts(self, kps_texts=[]):
        '''
        Assume the given textual keypoints are valid. We construct the valid kp mask for them.
        kps_texts: a list of N_t kps texts, e.g., ['left_eye', 'right_eye', 'nose'] or []
        '''
        N_t = len(kps_texts)  # T2=1
        if N_t > 0:
            kps_texts_mask = torch.ones((N_t, 1), dtype=torch.long)  # N_t x 1
        else:
            kps_texts_mask = None
        return kps_texts_mask  # N_t x 1 or None

    def detect(self, queries, supports=None, support_kps=None, support_kps_mask=None, kps_texts=[], kps_texts_mask=None):
        '''
        Simple GKD detection interface!

        ***input arguments***
        queries: B2 x C x H x W (H=W=384)

        supports: B1 x C x H x W (H=W=384) or None (if no visual prompt)
        support_kps: B1 x N_v x 2, ranges -1~1 (continuous) or None (if no visual prompt)
        support_kps_mask: B1 x N_v or None (if no visual prompt)

        kps_texts: a list of N_t kps texts, e.g., ['left_eye', 'right_eye', 'nose'] or [] (if no text prompt)
        kps_texts_mask: N_t x 1 or None (if no text prompt)

        ***output arguments***
        predictions: B2 x N x 2 (each row is a point (x, y) in -1~1)
        predict_score: B2 x N (0~about 1)   
        '''
        queries = queries.to(self.device)  # B2 x C x H x W

        if supports is not None:
            supports = supports.to(self.device)  # B1 x C x H x W
            support_kps = support_kps.to(self.device)  # B1 x N_v x 2
            support_kps_mask = support_kps_mask.to(self.device)  # B1 x N_v
        else:
            supports, support_kps, support_kps_mask = None, None, None

        if len(kps_texts) > 0:
            kps_texts_mask = kps_texts_mask.to(self.device)  # N_t x 1
        else:
            kps_texts_mask = None

        # add episode axis (episode S=1)
        queries = queries.unsqueeze(0)  # 1 x B2 x C x H x W
        if supports is not None:
            supports, support_kps, support_kps_mask = supports.unsqueeze(0), support_kps.unsqueeze(0), support_kps_mask.unsqueeze(0)  # 1 x B1 x C x H x W, 1 x B1 x N_v x 2, 1 x B1 x N_v
        if len(kps_texts) > 0:
            kps_texts = [kps_texts]  # a list with S lists where each is an episode's N_t kps texts (i.e. S x (N_t*1))
            kps_texts_mask = kps_texts_mask.unsqueeze(0)  # 1 x N_t x 1

        predictions, predict_score = self.vanilla_detect(queries, supports, support_kps, support_kps_mask, kps_texts, kps_texts_mask)

        return predictions, predict_score

def bbox_check(xmin_bbox, ymin_bbox, w_bbox, h_bbox, w, h):
    '''
    Safty check to ensure bounding box being within an image.
    w, h: image width and height

    return a bbox with xmin_bbox, ymin_bbox, w_bbox, h_bbox
    '''
    xmin_bbox, ymin_bbox = min(max(xmin_bbox, 0), w-1), min(max(ymin_bbox, 0), h-1)  # (xmin, ymin) should be within image
    w_bbox, h_bbox = max(w_bbox, 20), max(h_bbox, 20)  # set bbox_w, bbox_h >= 20
    w_bbox, h_bbox = min(w_bbox, w-xmin_bbox), min(h_bbox, h-ymin_bbox)  # (xmax, ymax) should be within image
    if w_bbox < 20 or h_bbox < 20:  # pay attention to the case: w_bbox = 20 & h_bbox = 20
        bbox = np.array([0, 0, w-1, h-1], np.float64)  # reset bbox. For safe, set [0, 0, w-1, h-1] instead of [0, 0, w, h]
    else:
        bbox = np.array([xmin_bbox, ymin_bbox, w_bbox, h_bbox], np.float64)
    return bbox

def data_preprocess(preprocess, image_transform, image, bboxes, keypoints=[]):
    '''
    input
    image: PIL.Image
    bboxes: numpy.ndarray with shape N_bbox x 4, each row is (x1,y1,x2,y2)
    keypoints: [...], a list of N_bbox sublists/None (None if no kps for that object box). Each sublist includes N_kps keypoints like [x1,y1,is_visible,...].

    output
    roi_images: torch.Tensor, N_bbox x 3 x H x W (ROI edges H and W are determined by preprocess function)
    scale_trans: torch.Tensor, N_bbox x 5
    transformed_kps_list: a list of N_bbox sublists/None (None if no kps for that object box)
    '''
    w, h = image.size  # PIL image
    N_bbox = bboxes.shape[0]

    object_images_list = []  # ROI images
    scale_trans_list = []
    transformed_kps_list = []  
    for i in range(N_bbox):
        xmin_bbox, ymin_bbox, xmax_bbox, ymax_bbox = bboxes[i]
        w_bbox, h_bbox = xmax_bbox-xmin_bbox+1, ymax_bbox-ymin_bbox+1
        bbox = bbox_check(xmin_bbox, ymin_bbox, w_bbox, h_bbox, w, h)

        kps_per_instance = keypoints[i]  # None or a list 
        if kps_per_instance is not None:
            kps_per_instance = np.array(keypoints, np.float64).reshape(-1, 3)  # N_kps x 3
        anno = {'keypoints': kps_per_instance, 'bbox': bbox}
        meta = {
            'scale': 1.0,
            'offset': np.array([0, 0], np.float64),
            'pad_offset': np.array([0, 0], np.float64),
        }
        image_new, anno, meta = preprocess(image, anno, meta)

        transformed_kps_per_instance = anno['keypoints']
        if kps_per_instance is not None:
            # invisible keypoints set to be (0,0,0)
            visible_kps_mask = (transformed_kps_per_instance[:, 2] > 0).astype(np.float64)
            transformed_kps_per_instance = transformed_kps_per_instance * visible_kps_mask[:, np.newaxis]  
            transformed_kps_per_instance = transformed_kps_per_instance.reshape(-1).tolist()

        # scale, xoffset, yoffset, pad_xoffset, pad_yoffset
        scale_trans = torch.tensor([meta['scale'], meta['offset'][0], meta['offset'][1], meta['pad_offset'][0], meta['pad_offset'][1]])
        image_new = image_transform(image_new)

        # gather data
        object_images_list.append(image_new)
        scale_trans_list.append(scale_trans)
        transformed_kps_list.append(transformed_kps_per_instance)

    roi_images = torch.stack(object_images_list, dim=0) # N_bbox x 3 x H x W
    scale_trans = torch.stack(scale_trans_list, dim=0)  # N_bbox x 5
    
    return roi_images, scale_trans, transformed_kps_list  

def demo(model_infer: GKDInference, input_im_path: str, bbox_on_input_im: list=[], support_im_path: str='', support_kps: list=[], kps_texts: list=[]):
    '''
    ***input arguments***
    input_im_path: path to an image waiting to detect keypoints
    bbox_on_input_im: a list of bboxes, N_bbox*4 numbers like [x1,y1,x2,y2,...]; every four numbers (x1,y1,x2,y2) represent one instance box (top-left point, bottom-right point). If empty, the entire image (0,0,w-1,h-1) is region of interest.

    VISUAL PROMPT
    support_im_path: path of support image (It is '' if no visual prompt)
    support_kps: a list of points, N_v*2 numbers like [x1,y1,...]; every two numbers (x1,y1) represent a point. (It is [] if no visual prompt)
    
    TEXT PROMPT
    kps_texts: a list of N_t keypoint texts (It is [] if no text prompt). 

    MULTIMODAL PROMPT
    In this demo, if both visual and text prompts are provided, N_v should equal N_t.

    ***output arguments***
    predictions_o: N_bbox x N x 2 (each row is a point (x, y) in [0, image width]x[0, image height])
    predict_score: N_bbox x N (0~about 1)  
    '''
    # Set status to avoid being out of GPU memory
    model_infer.gkd_model.eval()  # affects BN & disable Dropout
    torch.set_grad_enabled(False)  # disable grad computation

    # 1) Define preprocess functions to get ROI images with standard sizes (H x W = square_image_length x square_image_length)
    square_image_length = model_infer.square_image_length
    preprocess = mytransforms.Compose([
        mytransforms.RandomCrop(crop_gt_bbox=True),  # crop GT bbox directly without randomness
        mytransforms.Resize(longer_length=square_image_length),
        mytransforms.CenterPad(target_size=square_image_length),
        mytransforms.CoordinateNormalize(normalize_bbox=True)
    ])
    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet's mean pixel values
        ])
    
    # 2. Load input image (original image space --> input space)
    image = cv2.imread(input_im_path)  # opencv can handle corrupted image
    image = image[:, :, [2,1,0]]  # BGR to RGB
    image = Image.fromarray(image)
    w, h = image.size  # PIL image
    w_h_origin = [w, h]

    if len(bbox_on_input_im) // 4 == 0:
        bbox_on_input_im = [0, 0, w-1, h-1]
    bbox_on_input_im = np.array(bbox_on_input_im, np.float64).reshape(-1, 4)  # N_bbox x 4
    N_bbox = bbox_on_input_im.shape[0]
    
    # N_bbox x 3 x H x W, N_bbox x 5, _
    queries, q_scale_trans, _ = data_preprocess(preprocess, image_transform, image, bbox_on_input_im, keypoints=[None]*N_bbox)
    
    # 3. If given, load 1-shot support image (original image space --> input space)
    if support_im_path != '':
        support_image = cv2.imread(support_im_path)  # opencv can handle corrupted image
        support_image = support_image[:, :, [2,1,0]]  # BGR to RGB
        support_image = Image.fromarray(support_image)
        s_w, s_h = support_image.size  # PIL image
        bbox_on_support_im = np.array([0, 0, s_w-1, s_h-1], np.float64).reshape(1, 4)  # 1 x 4
        support_kps = np.array(support_kps, np.float64).reshape(-1, 2)
        s_kps_with_visibility = np.ones((support_kps.shape[0], 3), dtype=np.float64)  # assume all provided kps in demo are valid
        s_kps_with_visibility[:, :2] = support_kps
        s_kps_list_origin = [s_kps_with_visibility.reshape(-1).tolist()]  # keypoints for one object
        # 1 x 3 x H x W, _, [[x1, y1, v1, ...]]
        supports, _, support_kps_list = data_preprocess(preprocess, image_transform, support_image, bbox_on_support_im, keypoints=s_kps_list_origin)
        support_kps_full = torch.tensor(support_kps_list).reshape(1, -1, 3)
        support_kps, support_kps_mask = support_kps_full[:, :, :2], (support_kps_full[:, :, 2]).long()
    else:
        supports, support_kps, support_kps_mask = None, None, None
    
    # make_grid_images(supports, denormalize=True, save_path='./test_real_world/grid_image_s.jpg')
    # make_grid_images(queries, denormalize=True, save_path='./test_real_world/grid_image_q.jpg')

    # 4. Detection
    kps_texts = copy.deepcopy(kps_texts)  # Do not affect original inputs as we may pad/unpad kps_texts later
    kps_texts_mask = model_infer.supplement_mask_for_kps_texts(kps_texts)  # assume all kps_texts are valid keypoint texts in this demo
    predictions, predict_score = model_infer.detect(queries, supports, support_kps, support_kps_mask, kps_texts, kps_texts_mask)  # N_bbox x N_v/N_t x 2, N_bbox x N_v/N_t

    predictions = predictions.cpu().detach()
    predict_score = predict_score.cpu().detach()

    # 5. Recover coordinates (output space --> original image space)
    predictions_o = mytransforms.recover_kps(predictions, square_image_length, q_scale_trans)
    
    return predictions_o, predict_score, w_h_origin

