import os
import json
import numpy as np
import copy

import torch
from datasets.dataset_meta_info import get_dataset_splits, get_data_cfg, get_supercategory_to_dataset_inds

from datasets.pycocotools.coco import COCO
from collections import OrderedDict
import random

class DataManager(object):
    def __init__(self, dataset_dict: dict, supercategory_list: list):
        '''
        dataset_dict, where each entry has the format as follows:
            {
                'dataset_type': 'human_face_300w',
                'anno_path': 'dataset_root/human_face/300w/annotations/face_landmarks_300w_train.json',
                'image_root': {'hdf5': False, 'path': 'dataset_root/human_face/300w/images/'},
                'obj_classes': [],
                'kps_classes': [],
                'supercategory': 'human_face',
            }
        supercategory_list: e.g., ['human_pose', 'human_face', 'human_limbs', 'animal_pose', ...]
        '''
        self.dataset_dict = dataset_dict
        self.supercategory_list = supercategory_list

        print('==> Loading COCO annotations for datasets ...')
        self.coco_objects = {}
        for per_dataset_index, v in dataset_dict.items():
            cocoGT = COCO(v['anno_path'])
            self.coco_objects[per_dataset_index] = cocoGT
        print('==> Finished loading COCO annotations for datasets.')

        # Build global-local annotation index mapping
        self.global_to_local_map, self.dataset_partitions =  self.build_global_annotation_index_for_all_datasets()
        
        # get a dict for mapping supercategory name to dataset_indexes
        self.sc_to_dataset_inds = get_supercategory_to_dataset_inds(supercategory_list, dataset_dict)

        # below two dict will be updated during sampling
        self.dataset_partitions_dynamic = copy.deepcopy(self.dataset_partitions)  
        self.sc_to_dataset_inds_dynamic = copy.deepcopy(self.sc_to_dataset_inds)

        # a = self.compute_anno_num_for_a_supercategory('animal_pose')
        # b = self.compute_anno_num_for_all_supercategories()


    def build_global_annotation_index_for_all_datasets(self):
        """
        Creates a global index for each annotation in each COCO object.

        Args:
            coco_objects (dict): Key is per_dataset_index, value is a COCO object.

        Returns:
            dict: global_to_local_map, where key is global index and value is a dict with
                'dataset_id' and 'local_anno_id'.
            dict: data_partitions, where key is dataset_index and value is a dict with
                'cat_to_anno_ids' (mapping from category id to list of local annotation ids) and 'total_anno_num'.
        """
        global_to_local_map = {}  # global index to local annotation index mapping
        global_index = 0
        dataset_partitions = {}  # a dict to store dataset partitions across categories
        
        for dataset_index, coco in self.coco_objects.items():
            # TODO: 1. Build global annotation id for each instance
            annotations = coco.dataset.get('annotations', [])
            for i in range(len(annotations)):
                per_anno = annotations[i]
                local_anno_id = per_anno['id']
                global_to_local_map[global_index] = {
                    'dataset_id': dataset_index,
                    'local_id': local_anno_id
                }
                # dataset_to_global_map[dataset_index].append(global_index)

                # Write global index to the annotation for future reference
                annotations[i]['global_id'] = global_index
                coco.anns[local_anno_id]['global_id'] = global_index  # also update in coco.anns    

                global_index += 1

            # TODO: 2. Build dataset partitions across categories
            if dataset_partitions.get(dataset_index) is None:
                dataset_partitions[dataset_index] = {
                    'cat_to_anno_ids': {},  # local annotation ids
                    'total_anno_num': 0
                    }
                
                per_dataset_meta_info = self.dataset_dict[dataset_index]
                obj_classes = per_dataset_meta_info['obj_classes']  # object class names
                obj_class_ids = []
                if obj_classes in [None, []]:  # retrieve all class ids from dataset
                    obj_class_ids = coco.getCatIds()
                else:  # retrieve some class ids from dataset specified by class names
                    for each_class_name in obj_classes:
                        cat_ids_tmp = coco.getCatIds(catNms=each_class_name)
                        obj_class_ids.extend(cat_ids_tmp)  # a list containing object class ids
                obj_class_ids.sort()

                for cat_id in obj_class_ids:
                    anno_ids = coco.getAnnIds(catIds=cat_id)
                    dataset_partitions[dataset_index]['cat_to_anno_ids'][cat_id] = anno_ids
                    dataset_partitions[dataset_index]['total_anno_num'] += len(anno_ids)

        return global_to_local_map, dataset_partitions

    def compute_anno_num_for_a_supercategory(self, supercategory: str, mode='dynamic'):
        # mode: 'dynamic' or 'orignal'
        if mode == 'dynamic':
            dataset_partitions = self.dataset_partitions_dynamic
        else:  # orignal
            dataset_partitions = self.dataset_partitions
        dataset_inds_list = self.sc_to_dataset_inds[supercategory]
        count = 0
        for per_dataset_index in dataset_inds_list:
            count += dataset_partitions[per_dataset_index]['total_anno_num']
        return count
        
    def compute_anno_num_for_all_supercategories(self, mode='dynamic'):
        # mode: 'dynamic' or 'orignal'
        if mode == 'dynamic':
            dataset_partitions = self.dataset_partitions_dynamic
            sc_to_dataset_inds = self.sc_to_dataset_inds_dynamic
        else:  # orignal
            dataset_partitions = self.dataset_partitions
            sc_to_dataset_inds = self.sc_to_dataset_inds

        sc_to_anno_num = {}
        for sc_name, dataset_inds_list in sc_to_dataset_inds.items():
            count = 0
            for per_dataset_index in dataset_inds_list:
                count += dataset_partitions[per_dataset_index]['total_anno_num']
            sc_to_anno_num[sc_name] = count
        return sc_to_anno_num
    
    def is_a_supercategory_topk(self, chosen_sc, topk=0.5, mode='dynamic'):
        sc_to_anno_num = self.compute_anno_num_for_all_supercategories(mode)  # a dict, e.g., {'human_pose': n1, 'animal_pose': n2, ...}
        sorted_supercategories = sorted(sc_to_anno_num, key=sc_to_anno_num.get, reverse=True)  # sort a dict's keys
        index = sorted_supercategories.index(chosen_sc)
        # if (index+1) <= topk:  # topk here is an integer
        if (index+1) <= (int(len(sorted_supercategories) * topk) + 1):  # topk here is a ratio; ceiling 
            within_topk = True
        else:
            within_topk = False
        return within_topk

    def reset_sampling_pool(self):
        self.dataset_partitions_dynamic = copy.deepcopy(self.dataset_partitions)  
        self.sc_to_dataset_inds_dynamic = copy.deepcopy(self.sc_to_dataset_inds)

    def sampling_a_supercategory(self, method='importance'):
        # sampling method: 'uniform', 'importance'
        if method == 'uniform':
            supercategories = list(self.sc_to_dataset_inds_dynamic.keys())
            # chosen_sc_id = np.random.randint(0, len(supercategories), 1)[0]  # choose a supercategory
            # chosen_sc = supercategories[chosen_sc_id]
            chosen_sc = np.random.choice(supercategories)
        else:  # importance
            sc_to_anno_num = self.compute_anno_num_for_all_supercategories(mode='dynamic')
            total_anno_num = sum(sc_to_anno_num.values())
            sc_to_prob = {k: v / total_anno_num for k, v in sc_to_anno_num.items()}

            supercategories = list(sc_to_prob.keys())
            probs = list(sc_to_prob.values())
            chosen_sc = np.random.choice(supercategories, p=probs)  # choose a supercategory
        return chosen_sc
    
    def sampling_a_dataset_from_a_supercategory(self, supercategory: str, method='importance'):
        # sampling method: 'uniform', 'importance'
        dataset_inds_list = self.sc_to_dataset_inds_dynamic[supercategory]
        if method == 'uniform':
            chosen_dataset_id = np.random.choice(dataset_inds_list)  # choose a dataset
        else:  # 'importance'
            nums = np.zeros(len(dataset_inds_list))
            for i, per_dataset_index in enumerate(dataset_inds_list):
                num_per_dataset = self.dataset_partitions_dynamic[per_dataset_index]['total_anno_num']
                nums[i] = num_per_dataset
            total_num = np.sum(nums)
            prob = nums / total_num
            chosen_dataset_id = np.random.choice(dataset_inds_list, p=prob)  # choose a dataset

        return chosen_dataset_id
    
    def sampling_episodes(self, k_shot, m_query, episode_type='one_class', sampling_strategy='importance', num_episodes=-1, 
                          least_support_kps_num=1, least_query_kps_num=1, **kwargs):
        """
        k_shot: number of support samples 
        m_query: number of query samples 
        episode_type: 'one_class' or 'multi_class'
        sampling_strategy: 'importance', 'uniform', 'dynamic_importance'
        num_episodes: number of episodes to sample.
                      -1 means sample until no more data can be sampled in a dataset;
                      >=0 means we will sample the desired number of episodes. If the data is not enough in one epoch, 
                          we will reset the data pool and continue sampling
        """
        assert k_shot >= 0 and m_query > 0
        assert episode_type in ['one_class', 'mix_class']
        self.reset_sampling_pool()

        episodes_list = []  # each entry is a tuple of support and query global ids, i.e., (id1, ..., id_B1, id_(B1+1), ..., id_(B1+B2))
        
        #------------
        # # for recording distributions over supercategories and datasets
        # # _r_N=84411  # dynamic_importance
        # # _r_N=78511  # importance
        # _r_N=78389  # uniform
        # _r_steps = [1, 0.5, 0.1, 0.02]
        # _r_nums = [int(_r_N * (1-_r_steps[_i])) for _i in range(len(_r_steps))]
        # _r_count = 0
        # _r_dists = []
        #------------
        while True:
            #------------
            # # for recording distributions over supercategories and datasets
            # if (_r_count<len(_r_steps)) and (len(episodes_list) == _r_nums[_r_count]):
            #     sc_to_anno_num = self.compute_anno_num_for_all_supercategories(mode='dynamic')
            #     _r_dists.append(copy.deepcopy(sc_to_anno_num))
            #     _r_count += 1
            #     if _r_count >= len(_r_steps):
            #         # np.save('dynamic_importance_sampling.npy', _r_dists)
            #         # np.save('importance_sampling.npy', _r_dists)
            #         np.save('uniform_sampling.npy', _r_dists)
            #         exit(0)
            #------------

            if (num_episodes >= 0) and (len(episodes_list) >= num_episodes):  # reached the desired number of episodes
                print('==> Sampled the desired number of episodes.')
                break
            
            # 1. sample a supercategory
            # 2. sample a dataset from the supercategory
            # 3. sample k+m annotations from the dataset
            # 4. update the dynamic sampling pool
            # 5. if the supercategory or dataset has no more annotations, remove it from the dynamic pool
            # 6. if no more supercategories or datasets, stop sampling
            if sampling_strategy in ['importance', 'dynamic_importance']:
                chosen_sc = self.sampling_a_supercategory(method='importance')
                chosen_dataset_id = self.sampling_a_dataset_from_a_supercategory(chosen_sc, method='importance')
            elif sampling_strategy == 'uniform':
                chosen_sc = self.sampling_a_supercategory(method='uniform')
                chosen_dataset_id = self.sampling_a_dataset_from_a_supercategory(chosen_sc, method='uniform')
            else:
                raise NotImplementedError
            
            coco = self.coco_objects[chosen_dataset_id]
            each_dataset_meta_info = self.dataset_dict[chosen_dataset_id]
            each_dataset_partition = self.dataset_partitions_dynamic[chosen_dataset_id]  # {'cat_to_anno_ids': {cat_id1: [], ...}, 'total_anno_num': 0}
            
            if episode_type == 'one_class':
                obj_class_ids = list(each_dataset_partition['cat_to_anno_ids'].keys())
                sampled_cat_id = np.random.choice(obj_class_ids)  # sample a class id
                candidates_ids = each_dataset_partition['cat_to_anno_ids'][sampled_cat_id]  # local annotation ids

                category_entry = coco.loadCats([sampled_cat_id])[0]
                FULL_SET_KEYPOINT_TYPES = category_entry['keypoints'] 
                kps_classes_input = each_dataset_meta_info['kps_classes']  
                if kps_classes_input in [None, []]:  # if None, use all kps in FULL_SET_KEYPOINT_TYPES
                    support_kp_categories = FULL_SET_KEYPOINT_TYPES
                else:  # use kp types specified in each dataset
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
                                if (num_episodes >= 0) and (len(episodes_list) < num_episodes):  # not reached the desired number of episodes
                                    self.reset_sampling_pool()
                                    continue
                                else:  # num_episode=-1, sample one epoch data
                                    print('==> No more supercategories to sample from. Stopping.')
                                    break
                    continue  # go to the next episode
                
                # TODO: sample annotation ids and remove them from candidates_ids
                indices = random.sample(range(num_candidate), NUM_INSTANCE)  # to efficiently sample from a large list, we sample indexes first and then retrieve
                sampled_anno_ids = [candidates_ids[i] for i in indices]
                sampled_global_ids = []
                sample_flag = True

                # --------------------------------------------------------------------------
                # TODO: Judge if the sampled annotations contain the required keypoints. If not, re-sample
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
                least_num_tmp = min(least_support_kps_num, num_kp_categories)  # in case of N_way is very small
                if torch.sum(union_support_kp_mask) < least_num_tmp:
                    sample_flag = False
                
                # compute the valid query keypoints, using broadcast
                valid_kp_mask = query_kp_mask * union_support_kp_mask.reshape(1, num_kp_categories)
                least_query_kps_num_tmp = min(least_query_kps_num, num_kp_categories)  # in case of N_way is very small
                valid_list = torch.sum(valid_kp_mask, dim=1) >= least_query_kps_num_tmp  # check how many queries have least number of keypoints
                if torch.sum(valid_list) < valid_list.size(0) * 0.1:  # valid query samples
                    sample_flag = False
                # --------------------------------------------------------------------------

                if sample_flag == True:
                    episodes_list.append(tuple(sampled_global_ids))  # each entry is a tuple of (global ids, kps_classes, dataset_meta_info)
                else:  # re-sample and DO NOT add episode
                    pass

                delete_flag = True  # by default is True
                if sampling_strategy in ['importance', 'uniform']:
                    # remove sampled ids from candidates_ids
                    delete_flag = True
                elif sampling_strategy == 'dynamic_importance':
                    # conditionally remove sampled ids from candidates_ids
                    within_topk = self.is_a_supercategory_topk(chosen_sc, kwargs['topk'], mode='dynamic')
                    delete_flag = True if within_topk == True else False
                else:
                    raise NotImplementedError
                if delete_flag == True:
                    # TODO: remove sampled ids from candidates_ids
                    for i in sorted(indices, reverse=True):  
                        del candidates_ids[i]
                    each_dataset_partition['total_anno_num'] -= NUM_INSTANCE
                else:  
                    pass  # do not remove
                
                if len(candidates_ids) < NUM_INSTANCE:  # no enough annotations in this class
                    each_dataset_partition['total_anno_num'] -= len(candidates_ids)
                    each_dataset_partition['cat_to_anno_ids'].pop(sampled_cat_id)  # remove this class from dynamic pool
                    if each_dataset_partition['total_anno_num'] < NUM_INSTANCE:  # no more annotations in this dataset
                        self.sc_to_dataset_inds_dynamic[chosen_sc].remove(chosen_dataset_id)
                        if len(self.sc_to_dataset_inds_dynamic[chosen_sc]) == 0:  # no more datasets in this supercategory
                            self.sc_to_dataset_inds_dynamic.pop(chosen_sc)
                            if len(self.sc_to_dataset_inds_dynamic) == 0:  # no more supercategories
                                if (num_episodes >= 0) and (len(episodes_list) < num_episodes):  # not reached the desired number of episodes
                                    self.reset_sampling_pool()
                                    continue
                                else:  # num_episode=-1, sample one epoch data
                                    print('==> No more supercategories to sample from. Stopping.')
                                    break
                    continue  # go to the next episode
            else:  # 'mix_class'
                raise NotImplementedError
            
        return episodes_list


if __name__ == '__main__':
    f=json.load(open('/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json', 'r'))
    f=json.load(open('/project/vonneumann1/cl2025/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json', 'r'))