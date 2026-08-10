# -*- coding: utf-8 -*-
import os
import json
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from collections import OrderedDict
import random

import copy
import cv2
import math
import time
import h5py

# from datasets.pycocotools.coco import COCO

from datasets.TextGeneration.text_prompts_input import generate_input_text_prompts
from datasets.TextGeneration.kp_names_mapping import get_mapped_kps_names
from datasets.TextGeneration.names_misc import get_obj_class_name_for_query_image, obj_class_name_preprocess

class GKDDataset(Dataset):
    def __init__(self, episode_list: list, global_to_local_id_map: dict, dataset_meta_info: dict, coco_objects: dict, preprocess=None, input_transform=None, target_transform=None, **kwargs):
        """
        GKDDataset for episodic training or testing
        Keyword Arguments:
            episode_list {list} -- a list of episode, where each episode is a tuple of support and query global ids, 
                                    i.e., (id1, ..., id_B1, id_(B1+1), ..., id_(B1+B2))
            global_to_local_id_map {dict} -- a map from global image id to (dataset_index, local_image_id)
            dataset_meta_info {dict} -- key is dataset_index, value is dataset_meta_info
            coco_objects {dict} -- key is dataset_index, value is coco object

            preprocess {Callable} -- preprocessing for images and labels (default: {None})
            input_transform {Callable} -- preprocessing for images (default: {None})
            target_transform {Callable} -- preprocessing for labels (default: {None})
        """
        self.episode_list = episode_list
        self.global_to_local_id_map = global_to_local_id_map
        self.dataset_meta_info = dataset_meta_info  # key is dataset_index, value is dataset_meta_info
        self.coco_objects = coco_objects  # key is dataset_index, value is coco object

        self.preprocess = preprocess  # callable object
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        self.other_info = kwargs.get('other_info', {'padded_num_kps': -1})  # used for recording some other info

        self.padded_num_kps = self.other_info['padded_num_kps']  # -1 means disable padding

    def __getitem__(self, index):
        # Get one episode of global indices, i.e., a tuple of (id1, ..., id_B1, id_(B1+1), ..., id_(B1+B2))
        one_episode_global_ids = self.episode_list[index]  # index is episode index
        
        images_list = []
        labels_list = []
        kp_masks_list = []

        scale_trans_list = []
        bbox_origin_list = []
        w_h_origin_list = []

        episode_index = index
        num_kp_categories_list = []

        for global_id in one_episode_global_ids:
            id_map_entry = self.global_to_local_id_map[global_id]
            dataset_id, local_anno_id = id_map_entry['dataset_id'], id_map_entry['local_id']
            
            dataset_meta_info = self.dataset_meta_info[dataset_id]
            cocoGT = self.coco_objects[dataset_id]

            each_sample = cocoGT.anns[local_anno_id]
            keypoints = each_sample['keypoints']  # 1D list of [x, y, is_visible, ...]
            visible_bounds = each_sample['bbox']  # [xmin, ymin, w, h]

            image_id = each_sample['image_id']
            image_entry = cocoGT.loadImgs(image_id)[0]
            filename = image_entry['file_name']

            category_id = each_sample['category_id']
            category_entry = cocoGT.loadCats(category_id)[0]
            # category = category_entry['name']
            FULL_SET_KEYPOINT_TYPES = category_entry['keypoints']

            kps_classes_input_tmp = dataset_meta_info['kps_classes']  # if None, use all kps in FULL_SET_KEYPOINT_TYPES
            if kps_classes_input_tmp in [None, []]:
                support_kp_categories = FULL_SET_KEYPOINT_TYPES
            else:
                support_kp_categories = kps_classes_input_tmp   

            # try:
            use_hdf5 = dataset_meta_info['image_root']['hdf5']
            if use_hdf5 is False:  # read from raw files
                image_path = os.path.join(dataset_meta_info['image_root']['path'], filename)
                #-----------------------------
                # image = Image.open(image_path).convert('RGB')  # method1: may load wrong image after considering shooting angle
                # image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')  # method2: safe image loading (but cannot handle corrupted image)
                image = cv2.imread(image_path)  # method3: seems opencv can handle corrupted image
                image = image[:, :, [2,1,0]]  # BGR to RGB
                image = Image.fromarray(image)
                #-----------------------------
                w, h = image.size  # PIL image
            else:  # read from hdf5
                hdf5_images_fin = h5py.File(dataset_meta_info['image_root']['path']+'.hdf5', 'r')
                key_for_image = filename
                jpeg_stream = hdf5_images_fin[key_for_image]
                image = cv2.imdecode(jpeg_stream[()], cv2.IMREAD_COLOR)  # cv2.IMREAD_UNCHANGED
                image = image[:, :, [2,1,0]]  # rgb
                image = Image.fromarray(image)
                w, h = image.size
                hdf5_images_fin.close()  # close the hdf5 file
            # except:
            #     print(image_path)
            #     print(each_sample, image_entry, dataset_meta_info)
            
            w_h_origin = torch.tensor([w, h])
            # if w <= 50 or h <= 50:
            #     print(w, h)
            #     print(each_sample, image_entry, dataset_meta_info)
            assert w > 30 and h > 30, 'Input image should be not too small!'  # check image size

            all_labels = keypoints  # 1D list of [x, y, is_visible, ...]
            all_labels = np.array(all_labels, np.float64).reshape(-1, 3)  # N x 3
            xmin_bbox, ymin_bbox, w_bbox, h_bbox = visible_bounds[0], visible_bounds[1], visible_bounds[2], visible_bounds[3]  # [xmin, ymin, w, h]
            # secure our bbox is within the image (some bboxes may be out of boundary)
            xmin_bbox, ymin_bbox = min(max(xmin_bbox, 0), w-1), min(max(ymin_bbox, 0), h-1)  # (xmin, ymin) should be within image
            w_bbox, h_bbox = max(w_bbox, 20), max(h_bbox, 20)  # set bbox_w, bbox_h >= 20
            w_bbox, h_bbox = min(w_bbox, w-xmin_bbox), min(h_bbox, h-ymin_bbox)  # (xmax, ymax) should be within image
            if w_bbox < 20 or h_bbox < 20:  # pay attention to the case: w_bbox = 20 & h_bbox = 20
                bbox = np.array([0, 0, w-1, h-1], np.float64)  # reset bbox. For safe, set [0, 0, w-1, h-1] instead of [0, 0, w, h]
            else:
                bbox = np.array([xmin_bbox, ymin_bbox, w_bbox, h_bbox], np.float64)
            bbox_origin = torch.tensor(copy.deepcopy(bbox))  # [xmin, ymin, w, h]

            anno = {
                'keypoints': all_labels,
                'bbox': bbox
            }
            meta = {
                'scale': 1.0,
                'offset': np.array([0, 0], np.float64),
                'pad_offset': np.array([0, 0], np.float64),
                'valid_area': np.array([0, 0, w, h], np.float64),
                'hflip': False,  # may randomly flip and set it to be True in preprocess
                # 'anno_entry': copy.deepcopy(each_sample),
                # 'image_entry': copy.deepcopy(image_entry),
                # 'bbox': copy.deepcopy(bbox),
                # 'bbox_origin': copy.deepcopy(bbox_origin),
                # 'dataset_meta_info': copy.deepcopy(dataset_meta_info)
            }
            if self.preprocess != None:
                image, anno, meta = self.preprocess(image, anno, meta)
            
            all_labels_transformed = anno['keypoints']
            # re-set those invalid keypoint coordinates since they may be out of boundary after transformation
            # invisible keypoints set to be (0,0,0)
            all_labels_visible_mask = (all_labels_transformed[:, 2] > 0).astype(np.float64)
            all_labels_transformed = all_labels_transformed * all_labels_visible_mask[:, np.newaxis]  

            # scale, xoffset, yoffset, pad_xoffset, pad_yoffset
            scale_trans = torch.tensor([meta['scale'], meta['offset'][0], meta['offset'][1], meta['pad_offset'][0], meta['pad_offset'][1]])

            # extract the transformed keypoint labels relevant to our support keypoints
            num_kp_categories = len(support_kp_categories)  # N_original
            num_kp_final = self.compute_num_kp_final(num_kp_categories)  # N_final, after kp padding if needed
            label = np.zeros((num_kp_final, 2))  # N x 2
            kp_mask = torch.zeros(num_kp_final)  # N
            for i, kp_type in enumerate(support_kp_categories):
                kp_id = FULL_SET_KEYPOINT_TYPES.index(kp_type)
                label[i, :] = all_labels_transformed[kp_id, :2]
                if all_labels_transformed[kp_id, 2] <= 0:  # invisible
                    kp_mask[i] = 0
                else:
                    kp_mask[i] = 1
            label = torch.tensor(label, dtype=torch.float32)
            
            if self.input_transform is not None:
                image = self.input_transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)    

            # gather data
            images_list.append(image)
            labels_list.append(label)
            kp_masks_list.append(kp_mask)
            scale_trans_list.append(scale_trans)
            bbox_origin_list.append(bbox_origin)
            w_h_origin_list.append(w_h_origin)
            num_kp_categories_list.append(num_kp_categories)

        #=====================================================
        # TODO: text prompt generation (Note that object text depends on object class name of query image)
        obj_texts, obj_texts_mask, kps_texts, kps_texts_mask = self.text_prompt_generation(one_episode_global_ids)
        #=====================================================

        # return image, label, kp_mask, scale_trans, bbox_origin, w_h_origin 
        one_episode_data = {
            'images': torch.stack(images_list, dim=0),  # (B1+B2) x 3 x H x W
            'labels': torch.stack(labels_list, dim=0),  # (B1+B2) x N x 2
            'kp_masks': torch.stack(kp_masks_list, dim=0),  # (B1+B2) x N
            'scale_trans': torch.stack(scale_trans_list, dim=0),  # (B1+B2) x 5
            'bbox_origin': torch.stack(bbox_origin_list, dim=0),  # (B1+B2) x 4
            'w_h_origin': torch.stack(w_h_origin_list, dim=0),    # (B1+B2) x 2
            'episode_index': torch.tensor(episode_index),  # 1
            'num_kp_original': torch.tensor(num_kp_categories_list),  # (B1+B2)
            #----------------------------------------
            # below are text prompts for this episode
            'obj_texts': obj_texts,  # a list of T1 object texts
            'obj_texts_mask': obj_texts_mask,  # T1
            'kps_texts': kps_texts,  # a list of (N_final * T2) kps texts
            'kps_texts_mask': kps_texts_mask,  # N_final x T2
        }       
        return one_episode_data
    
    def __len__(self):
        return len(self.episode_list)
    
    def compute_num_kp_final(self, num_kp_categories):
        if self.padded_num_kps == -1:
            return num_kp_categories
        else:
            num_kp_final = self.padded_num_kps
            assert num_kp_final >= num_kp_categories, 'padded_num_kps should be >= num_kp_categories'
            return num_kp_final
        
    def text_prompt_generation(self, one_episode_global_ids):
        #----------------------------------------
        # code copy from self.__getitem__(); just for preparation.
        one_global_id = one_episode_global_ids[-1]  # the last one is the query image
        id_map_entry = self.global_to_local_id_map[one_global_id]
        dataset_id, local_anno_id = id_map_entry['dataset_id'], id_map_entry['local_id']
        
        dataset_meta_info = self.dataset_meta_info[dataset_id]
        cocoGT = self.coco_objects[dataset_id]
        one_query_sample = cocoGT.anns[local_anno_id]

        category_id = one_query_sample['category_id']
        category_entry = cocoGT.loadCats(category_id)[0]
        FULL_SET_KEYPOINT_TYPES = category_entry['keypoints']

        kps_classes_input_tmp = dataset_meta_info['kps_classes']  # if None, use all kps in FULL_SET_KEYPOINT_TYPES
        if kps_classes_input_tmp in [None, []]:
            support_kp_categories = FULL_SET_KEYPOINT_TYPES
        else:
            support_kp_categories = kps_classes_input_tmp  
        
        num_kp_categories = len(support_kp_categories)  # N_original
        num_kp_final = self.compute_num_kp_final(num_kp_categories)  # N_final, after kp padding if needed
        #----------------------------------------

        #----------------------------------------
        # below code is for text prompt generation
        obj_name = category_entry['name']  # Use the name of last query image as the name for whole episode
        
        dataset_type = dataset_meta_info['dataset_type']
        obj_name = obj_class_name_preprocess(dataset_type, obj_name)  # minor pre-processing to obj_name
        kps_names = get_mapped_kps_names(dataset_type, support_kp_categories, FULL_SET_KEYPOINT_TYPES)  # a list, N_kps texts
        
        # a list, T1 object texts; a list, (N_kps * T2) kps texts
        num_text_per_obj = self.other_info['num_text_per_obj']
        num_text_per_kp = self.other_info['num_text_per_kp']
        obj_texts, kps_texts = generate_input_text_prompts(obj_name, kps_names, num_text_per_obj, num_text_per_kp)  
        obj_texts_mask = torch.ones(num_text_per_obj) * (num_text_per_obj > 0) # T1
        kps_texts_mask = torch.ones(len(support_kp_categories), num_text_per_kp) * (num_text_per_kp>0) # N_kps x T2
        if num_kp_final > num_kp_categories:  # need text padding
            kps_texts = kps_texts + [''] * ((num_kp_final - num_kp_categories) * num_text_per_kp)  # N_final * T2, or [] if num_text_per_kp=0
            kps_texts_mask = torch.cat([kps_texts_mask, torch.zeros(num_kp_final - num_kp_categories, num_text_per_kp)], dim=0)  # N_final x T2
        
        return obj_texts, obj_texts_mask, kps_texts, kps_texts_mask
        #----------------------------------------


if __name__=='__main__':
    #================================
    from datasets.dataset_utils import draw_instance, draw_skeletons, filter_keypoints, draw_markers
    #================================
    pass

