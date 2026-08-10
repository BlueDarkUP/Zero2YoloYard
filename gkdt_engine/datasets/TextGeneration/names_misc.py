import re

def get_obj_class_name_for_query_image(query_annos, cocoGT):
    '''
    get the object class name for each query image
    '''
    query_cat_ids = [q['category_id'] for q in query_annos]
    query_cat_names = [cocoGT.cats[id]['name'] for id in query_cat_ids]
    return query_cat_names

def obj_class_name_preprocess(dataset_type, class_name):
    '''
    Perform minor pre-processing to the category string
    '''
    if dataset_type == 'animal_pose_dataset':
        # ANIMAL_POSE dataset requires no extra rules
        category_processed = class_name
    elif dataset_type == 'awa_pose':
        # example: 'polar+bear' -> 'polar bear'
        category_processed = class_name.replace("+", " ")
    elif dataset_type == 'cub':
        # for CUB-200-2011, remove 00X. prepend, replace underscore to space, lower capital words
        category_processed = class_name[4:].replace("_"," ").lower()
    elif dataset_type == 'nabird':
        # NABird experimentally removes the additional information in parenthesis
        category_processed = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", class_name).replace("()", "")
    elif 'deepfashion2' in dataset_type:  # 'deepfashion2_train', 'deepfashion2_val'
        category_processed = class_name.replace("_"," ").lower()
    else:  # coco, onehand10k, ap10k, deepfashion2, etc.
        category_processed = class_name

    return category_processed