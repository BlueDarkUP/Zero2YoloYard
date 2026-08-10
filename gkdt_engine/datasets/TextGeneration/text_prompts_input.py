import numpy as np

def kp_text_templates_backup(kp, obj):
    texts = [
        f'{kp}',
        f'the {kp}',
        f'the {kp} of {obj}',
        f'{obj}\'s {kp}',
    ]
    return texts

# Usage example: per_text.format(obj)
OBJ_TEXT_TEMPLATES = [
    '{0}',
    '{0}.',
    'the {0}',
    'a photo of {0}',
    'a photo of {0}.',
]

# Usage example: per_text.format(kp, obj), where {0} and {1} give the positions.
KP_TEXT_TEMPLATES = [
    '{0}',
    'the {0}',
    'the {0} of {1}',
    '{1}\'s {0}'
]

def generate_input_text_prompts(obj_name: str, kps_names: list, num_text_per_obj: int = 1, num_text_per_kp: int = 1):
    '''
    Generate object text prompts and keypoint text prompts. We can specify the number of prompts for each object or kp.

    :param obj_names: a single obj
    :param kps_names: [kp1, kp2, ...]
    :param num_text_per_obj: 1
    :param num_text_per_kp: 1
    :return:
    '''
    inds = np.random.randint(0, len(OBJ_TEXT_TEMPLATES), num_text_per_obj)
    obj_texts = [OBJ_TEXT_TEMPLATES[i].format(obj_name) for i in inds]

    kp_texts = []
    inds2 = np.random.randint(0, len(KP_TEXT_TEMPLATES), num_text_per_kp)
    for i in range(len(kps_names)):
        for j in inds2:
            kp_texts.append(KP_TEXT_TEMPLATES[j].format(kps_names[i], obj_name))

    return obj_texts, kp_texts


if __name__ == '__main__':
    obj_texts, kps_texts = generate_input_text_prompts('cat', ['ears', 'eyes'], 2, 3)
    print(obj_texts)
    print(kps_texts)
