import os
import numpy as np
import random
import json

def random_sample_kps(kps_num, sample_num):
    a = [int(v) for v in range(0, kps_num, 1)]  # 0~kps_num-1
    # print(a)
    return random.sample(a, sample_num)


if __name__=='__main__':
    n_animal = (17, 6)
    n_cub = (15, 5)
    n_nab = (11, 4)

    path = '../annotation_prepare/kp_types_random_split.json'
    splits_dict = {'animal':[[],[],[]], 'cub':[[],[],[]], 'nab':[[],[],[]],}
    if os.path.exists(path) == True:
        # read
        with open(path, 'r') as fin:
            splits = json.load(fin)
            splits_dict = splits
            fin.close()
    print('read dict:')
    print(splits_dict)

    # round 0
    # kps_animal  = random_sample_kps(n_animal[0], n_animal[1])
    # splits_dict['animal'][0] = kps_animal
    # round 1
    # kps_animal1 = random_sample_kps(n_animal[0], n_animal[1])
    # splits_dict['animal'][1] = kps_animal1
    # round 2
    # kps_animal2 = random_sample_kps(n_animal[0], n_animal[1])
    # splits_dict['animal'][2] = kps_animal2
    #
    # round 0
    # kps_cub = random_sample_kps(n_cub[0], n_cub[1])
    # splits_dict['cub'][0] = kps_cub
    # round 1
    # kps_cub1 = random_sample_kps(n_cub[0], n_cub[1])
    # splits_dict['cub'][1] = kps_cub1
    # round 2
    # kps_cub2 = random_sample_kps(n_cub[0], n_cub[1])
    # splits_dict['cub'][2] = kps_cub2
    #
    # round 0
    # kps_nab = random_sample_kps(n_nab[0], n_nab[1])
    # splits_dict['nab'][0] = kps_nab
    # round 1
    # kps_nab1 = random_sample_kps(n_nab[0], n_nab[1])
    # splits_dict['nab'][1] = kps_nab1
    # round 2
    # kps_nab2 = random_sample_kps(n_nab[0], n_nab[1])
    # splits_dict['nab'][2] = kps_nab2

    print('write dict:')
    print(splits_dict)
    with open(path, 'w') as fout:
        json.dump(splits_dict, fout)
        fout.close()
