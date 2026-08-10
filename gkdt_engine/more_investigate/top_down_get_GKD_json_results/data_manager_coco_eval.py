import os
import json
import numpy as np
import copy

import torch
from datasets.pycocotools.coco import COCO
from collections import OrderedDict
import random
from datasets.data_manager import DataManager

class DataManagerCOCOEval(DataManager):
    '''
        This class inherites DataManger and override function self.sampling_episodes(), 
        in order to sample all possible data to form episodes for fair evaluation
    '''
    def __init__(self, dataset_dict, supercategory_list):
        super().__init__(dataset_dict, supercategory_list)
    
    def sampling_episodes(self, k_shot, m_query, episode_type='one_class', sampling_strategy='squentially', num_episodes=-1, **kwargs):
        """
        k_shot: number of support samples 
        m_query: number of query samples (Set k_shot=0, m_query=1 can guarrantee all data are sampled)
        episode_type: 'one_class' or 'multi_class'
        sampling_strategy: 'squentially' or 'random'
        num_episodes: number of episodes to sample, -1 means sample until no more data can be sampled
        """
        assert k_shot >= 0 and m_query > 0
        assert episode_type in ['one_class', 'mix_class']
        self.reset_sampling_pool()

        episodes_list = []  # each entry is a tuple of support and query global ids, i.e., (id1, ..., id_B1, id_(B1+1), ..., id_(B1+B2))
        while True:
            if (num_episodes >= 0) and (len(episodes_list) >= num_episodes):  # reached the desired number of episodes
                print('==> Sampled the desired number of episodes.')
                break
            
            # 1. sample a supercategory
            # 2. sample a dataset from the supercategory
            # 3. sample k+m annotations from the dataset
            # 4. update the dynamic sampling pool
            # 5. if the supercategory or dataset has no more annotations, remove it from the dynamic pool
            # 6. if no more supercategories or datasets, stop sampling

            # Both 'importance' and 'uniform' are ok as we only have 1 dataset to eval during testing phase!
            chosen_sc = self.sampling_a_supercategory(method='importance')
            chosen_dataset_id = self.sampling_a_dataset_from_a_supercategory(chosen_sc, method='importance')  

            coco = self.coco_objects[chosen_dataset_id]
            each_dataset_meta_info = self.dataset_dict[chosen_dataset_id]
            each_dataset_partition = self.dataset_partitions_dynamic[chosen_dataset_id]  # {'cat_to_anno_ids': {cat_id1: [], ...}, 'total_anno_num': 0}

            if episode_type == 'one_class':
                obj_class_ids = list(each_dataset_partition['cat_to_anno_ids'].keys())
                sampled_cat_id = np.random.choice(obj_class_ids)  # sample a class id
                candidates_ids = each_dataset_partition['cat_to_anno_ids'][sampled_cat_id]  # local annotation ids

                category_entry = coco.loadCats([sampled_cat_id])[0]
                FULL_SET_KEYPOINT_TYPES = category_entry['keypoints'] 
                kps_classes_input = each_dataset_meta_info['kps_classes']  # if None, use all kps in FULL_SET_KEYPOINT_TYPES
                if kps_classes_input in [None, []]:
                    support_kp_categories = FULL_SET_KEYPOINT_TYPES
                else:
                    support_kp_categories = kps_classes_input

                num_kp_categories = len(support_kp_categories)
                assert num_kp_categories > 0
                NUM_INSTANCE = k_shot + m_query
                kp_mask = torch.ones(NUM_INSTANCE, num_kp_categories)   # (K_shot+M_query) x N_way
                
                num_candidate = len(candidates_ids)
                if num_candidate < NUM_INSTANCE:  # not enough samples in this class
                    each_dataset_partition['total_anno_num'] -= len(candidates_ids)
                    each_dataset_partition['cat_to_anno_ids'].pop(sampled_cat_id)  # remove this class from dynamic pool
                    if each_dataset_partition['total_anno_num'] < NUM_INSTANCE:  # no more annotations in this dataset
                        self.sc_to_dataset_inds_dynamic[chosen_sc].remove(chosen_dataset_id)
                        if len(self.sc_to_dataset_inds_dynamic[chosen_sc]) == 0:  # no more datasets in this supercategory
                            self.sc_to_dataset_inds_dynamic.pop(chosen_sc)
                            if len(self.sc_to_dataset_inds_dynamic) == 0:  # no more supercategories
                                print('==> No more supercategories to sample from. Stopping.')
                                break
                    continue  # go to the next episode
                
                # TODO: sample annotation ids and remove them from candidates_ids
                if sampling_strategy == 'squentially':
                    indices = list(range(NUM_INSTANCE))  # Select first NUM_INSTANCE annotations every time.
                    sampled_anno_ids = [candidates_ids[i] for i in indices]
                else:  # 'random'
                    indices = random.sample(range(num_candidate), NUM_INSTANCE)  # to efficiently sample from a large list, we sample indexes first and then retrieve
                    sampled_anno_ids = [candidates_ids[i] for i in indices]
                sampled_global_ids = []
                sample_flag = True

                # --------------------------------------------------------------------------
                # TODO: Judge if the sampled annotations contain the required keypoints.
                annos = coco.loadAnns(sampled_anno_ids)
                for i, each_anno in enumerate(annos):
                    # gather global ids
                    global_id = each_anno['global_id']
                    sampled_global_ids.append(global_id)

                    # compute kp_mask, torch.Tensor, (K_shot+M_query) x N_way
                    v = each_anno['keypoints'][2::3]  # x, y, v
                    for j, kp_type in enumerate(support_kp_categories):
                        kp_id = FULL_SET_KEYPOINT_TYPES.index(kp_type)
                        if v[kp_id] <= 0:  # invisible
                            kp_mask[i, j] = 0
                        else:
                            kp_mask[i, j] = 1

                if k_shot > 0:  # few-shot visual prompt
                    support_kp_mask = kp_mask[:k_shot, :]  # K_shot x N_way
                else:
                    support_kp_mask = torch.ones(1, num_kp_categories)  # 1 x N_way
                query_kp_mask = kp_mask[k_shot:, :]    # M_query x N_way

                # compute the union of keypoint types in sampled images, N_least <= N(union) <= N_way
                union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0  # tensor([True, False, True, ...])
                least_num_tmp = min(1, num_kp_categories)  # in case of N_way is very small
                if torch.sum(union_support_kp_mask) < least_num_tmp:
                    # sample_flag = False
                    pass  # still sample anyway
                
                # compute the valid query keypoints, using broadcast
                valid_kp_mask = query_kp_mask * union_support_kp_mask.reshape(1, num_kp_categories)
                least_query_kps_num_tmp = min(1, num_kp_categories)  # in case of N_way is very small
                valid_list = torch.sum(valid_kp_mask, dim=1) >= least_query_kps_num_tmp  # check how many queries have least number of keypoints
                if torch.sum(valid_list) < valid_list.size(0) * 0.1:  # valid query samples
                    # sample_flag = False
                    pass  # still sample anyway
                # --------------------------------------------------------------------------

                if sample_flag == True:
                    episodes_list.append(tuple(sampled_global_ids))  # each entry is a tuple of (global ids, kps_classes, dataset_meta_info)
                else:  # re-sample and DO NOT add episode
                    pass
                
                # TODO: remove sampled ids from candidates_ids
                for i in sorted(indices, reverse=True):  
                    del candidates_ids[i]
                each_dataset_partition['total_anno_num'] -= NUM_INSTANCE
                
                if len(candidates_ids) < NUM_INSTANCE:  # no enough annotations in this class
                    each_dataset_partition['total_anno_num'] -= len(candidates_ids)
                    each_dataset_partition['cat_to_anno_ids'].pop(sampled_cat_id)  # remove this class from dynamic pool
                    if each_dataset_partition['total_anno_num'] < NUM_INSTANCE:  # no more annotations in this dataset
                        self.sc_to_dataset_inds_dynamic[chosen_sc].remove(chosen_dataset_id)
                        if len(self.sc_to_dataset_inds_dynamic[chosen_sc]) == 0:  # no more datasets in this supercategory
                            self.sc_to_dataset_inds_dynamic.pop(chosen_sc)
                            if len(self.sc_to_dataset_inds_dynamic) == 0:  # no more supercategories
                                print('==> No more supercategories to sample from. Stopping.')
                                break
                    continue  # go to the next episode
            else:  # 'mix_class'
                raise NotImplementedError
            
        return episodes_list            