import os
from collections import OrderedDict

# SUPER_CATEGORIES = [
#     'human_pose',
#     'human_face',
#     'human_limbs',
#     'animal_pose',
#     'insect_pose',
#     'animal_face',
#     'furniture',
#     'vehicle',
#     'clothes',
#     'medical',
# ]

DATASET_ROOTS = {
    # Super-category: human_pose
    'coco_train': {
        'image_root': 'human_pose/coco/images/train2017',
        'anno_root': 'human_pose/coco/annotations',
    },
    'coco_val': {
        'image_root': 'human_pose/coco/images/val2017',
        'anno_root': 'human_pose/coco/annotations',
    },
    'human_art': {
        'image_root': 'human_pose/human_art/images',
        'anno_root': 'human_pose/human_art/annotations',
    },

    # Super-category: human_face
    'human_face_300w': {
        'image_root': 'human_face/300w/images',
        'anno_root': 'human_face/300w/annotations',
    },
    'human_face_aflw': {
        'image_root': 'human_face/aflw/images',
        'anno_root': 'human_face/aflw/annotations',
    },

    # Super-category: human_limbs
    'onehand10k': {
        'image_root': 'human_limbs/onehand10k/images',
        'anno_root': 'human_limbs/onehand10k/annotations',
    },
    'hint': {
        'image_root': 'human_limbs/hint/images',
        'anno_root': 'human_limbs/hint/annotations',
    },

    # Super-category: animal_pose
    'animal_pose_dataset': {
        'image_root': 'animal_pose/animal_pose_dataset/images',
        'anno_root': 'animal_pose/animal_pose_dataset/annotations',
    },
    'awa_pose': {
        'image_root': 'animal_pose/awa_pose/JPEGImages',
        'anno_root': 'animal_pose/awa_pose/annotations',
    },
    'cub': {
        'image_root': 'animal_pose/cub/images',
        'anno_root': 'animal_pose/cub/annotations'
    },
    'nabird': {
        'image_root': 'animal_pose/nabird/images',
        'anno_root': 'animal_pose/nabird/annotations'
    },
    'ap10k': {
        'image_root': 'animal_pose/ap10k/images',
        'anno_root': 'animal_pose/ap10k/annotations_fskd'
    },
    'apt36k': {
        'image_root': 'animal_pose/apt36k/images',
        'anno_root': 'animal_pose/apt36k/annotations'
    },
    'macaque_pose': {
        'image_root': 'animal_pose/macaque_pose/images',
        'anno_root': 'animal_pose/macaque_pose/annotations'
    },
    'atrw_tiger': {
        'image_root': 'animal_pose/atrw_tiger/images',
        'anno_root': 'animal_pose/atrw_tiger/annotations'
    },
    'acinoset_cheetah': {
        'image_root': 'animal_pose/acinoset_cheetah/images',
        'anno_root': 'animal_pose/acinoset_cheetah/annotations'
    },
    'animal_kingdom': {
        'image_root': 'animal_pose/animal_kingdom/images',
        'anno_root': 'animal_pose/animal_kingdom/annotations/ak_combined'
    },
    'topviewmouse5k': {
        'image_root': 'animal_pose/topviewmouse5k/images',
        'anno_root': 'animal_pose/topviewmouse5k/annotations'
    },

    # Super-category: insect_pose
    'vinegar_fly': {
        'image_root': 'insect_pose/vinegar_fly/images',
        'anno_root': 'insect_pose/vinegar_fly/annotations'
    },
    'desert_locust': {
        'image_root': 'insect_pose/desert_locust/images',
        'anno_root': 'insect_pose/desert_locust/annotations'
    },

    # Super-category: animal_face
    'animalweb': {
        'image_root': 'animal_face/animalweb/images',
        'anno_root': 'animal_face/animalweb/annotations'
    },

    # Super-category: furniture
    'keypoint5': {
        'image_root': 'furniture/keypoint-5/images',
        'anno_root': 'furniture/keypoint-5/annotations'
    },

    # Super-category: vehicle
    'carfusion': {
        'image_root': 'vehicle/carfusion/images',
        'anno_root': 'vehicle/carfusion/annotations'
    },

    # Super-category: clothes (Pay attention: many no-name keypoints)
    'deepfashion2_train': {
        'image_root': 'clothes/deepfashion2/train/image',
        'anno_root': 'clothes/deepfashion2/annotations'
    },
    'deepfashion2_val': {
        'image_root': 'clothes/deepfashion2/validation/image',
        'anno_root': 'clothes/deepfashion2/annotations'
    },

    # Super-category: medical
    'cephalometric_landmark': {
        'image_root': 'medical/cephalometric_landmark/images',
        'anno_root': 'medical/cephalometric_landmark/annotations'
    },
    'hand_xray': {
        'image_root': 'medical/hand_xray/images',
        'anno_root': 'medical/hand_xray/annotations'
    },
}


def get_image_root(dataset_type: str):
    return DATASET_ROOTS[dataset_type]['image_root']

def get_anno_root(dataset_type: str):
    return DATASET_ROOTS[dataset_type]['anno_root']

def get_data_cfg(cfg, phase='train'):
    if phase == 'train':
        DATA_CFG = cfg.DATASET.TRAIN_DATA
    elif phase == 'val':
        DATA_CFG = cfg.DATASET.VAL_DATA
    elif phase == 'test':
        DATA_CFG = cfg.DATASET.TEST_DATA
    else:
        raise NotImplementedError
    return DATA_CFG

def get_dataset_splits(cfg, phase='train'):
    '''
    phase: 'train', 'val' (seen or unseen species), 'test' (seen or unseen species)

    return a dataset_dict, where each entry has the format as follows:
        {
            'dataset_type': 'human_face_300w',
            'anno_path': 'dataset_root/human_face/300w/annotations/face_landmarks_300w_train.json',
            'image_root': {'hdf5': False, 'path': 'dataset_root/human_face/300w/images/'},
            'obj_classes': [],
            'kps_classes': [],
            'supercategory': 'human_face',
        }
    
    return a supercategory_list, which is a unique set
    '''
    DATA_CFG = get_data_cfg(cfg, phase)
    
    dataset_dict = {}
    supercategory_list = []  # we will use this supercategory_list to unique indexing  

    dataset_root = cfg.DATASET.ROOT
    for i in range(len(DATA_CFG)):
        new_entry = {}

        data_entry = DATA_CFG[i]  # ['awa_pose', 'AwAPose_split_train.json', [], []]
        dataset_type = data_entry[0]
        new_entry['dataset_type'] = dataset_type

        relative_anno_root = get_anno_root(dataset_type)
        relative_anno_path = os.path.join(relative_anno_root, data_entry[1])
        new_entry['anno_path'] = os.path.join(dataset_root, relative_anno_path)

        relative_image_root = get_image_root(dataset_type)
        image_root = os.path.join(dataset_root, relative_image_root)
        new_entry['image_root'] = {'hdf5': False, 'path': image_root}

        obj_classes = data_entry[2]  # if i-th element is None, it means i-th json's all classes can be sampled.
        if obj_classes in [[], None, 'None', 'null']:  # rectify
            obj_classes = None
        new_entry['obj_classes'] = obj_classes

        kps_classes = data_entry[3]  # if keypoint_set==None, all kps in json used
        if kps_classes in [[], None, 'None', 'null']:  # rectify
            kps_classes = None
        new_entry['kps_classes'] = kps_classes

        # e.g., 'human_pose', 'human_face', 'human_limbs', 'animal_pose', ...
        supercategory = relative_anno_root.split('/')[0]  
        new_entry['supercategory'] = supercategory
        supercategory_list.append(supercategory)

        dataset_dict[i] = new_entry

    supercategory_list = list(set(supercategory_list))
    supercategory_list.sort()
        
    return dataset_dict, supercategory_list

def get_supercategory_to_dataset_inds(supercategory_list: list, dataset_dict: dict):
    '''
    Try to return a dict with the keys being supercategory name and the values being a list of dataset index

    supercategory_list: a list of supercategory names
    dataset_dict: a dataset_dict, where each entry has the format as follows:
        {
            'dataset_type': 'human_face_300w',
            'anno_path': 'dataset_root/human_face/300w/annotations/face_landmarks_300w_train.json',
            'image_root': {'hdf5': False, 'path': 'dataset_root/human_face/300w/images/'},
            'obj_classes': [],
            'kps_classes': [],
            'supercategory': 'human_face',
        }
    '''
    sc_to_dataset_inds = OrderedDict()
    for sc_tmp in supercategory_list:
        sc_to_dataset_inds[sc_tmp] = []

    for per_dataset_index, v in dataset_dict.items():
        sc_tmp = v['supercategory']
        sc_to_dataset_inds[sc_tmp].append(per_dataset_index)

    return sc_to_dataset_inds