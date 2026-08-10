# human's 17 keypoints
coco = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']
# human_face_68_landmarks
human_face_300w = ['right_cheekbone_1', 'right_cheekbone_2', 'right_cheek_1', 'right_cheek_2', 'right_cheek_3', 'right_cheek_4', 'right_cheek_5', 'right_chin', 'chin_center', 'left_chin', 'left_cheek_5', 'left_cheek_4', 'left_cheek_3', 'left_cheek_2', 'left_cheek_1', 'left_cheekbone_2', 'left_cheekbone_1', 'right_eyebrow_1', 'right_eyebrow_2', 'right_eyebrow_3', 'right_eyebrow_4', 'right_eyebrow_5', 'left_eyebrow_1', 'left_eyebrow_2', 'left_eyebrow_3', 'left_eyebrow_4', 'left_eyebrow_5', 'nasal_bridge_1', 'nasal_bridge_2', 'nasal_bridge_3', 'nasal_bridge_4', 'right_nasal_wing_1', 'right_nasal_wing_2', 'nasal_wing_center', 'left_nasal_wing_1', 'left_nasal_wing_2', 'right_eye_eye_corner_1', 'right_eye_upper_eyelid_1', 'right_eye_upper_eyelid_2', 'right_eye_eye_corner_2', 'right_eye_lower_eyelid_2', 'right_eye_lower_eyelid_1', 'left_eye_eye_corner_1', 'left_eye_upper_eyelid_1', 'left_eye_upper_eyelid_2', 'left_eye_eye_corner_2', 'left_eye_lower_eyelid_2', 'left_eye_lower_eyelid_1', 'right_mouth_corner', 'upper_lip_outer_edge_1', 'upper_lip_outer_edge_2', 'upper_lip_outer_edge_3', 'upper_lip_outer_edge_4', 'upper_lip_outer_edge_5', 'left_mouth_corner', 'lower_lip_outer_edge_5', 'lower_lip_outer_edge_4', 'lower_lip_outer_edge_3', 'lower_lip_outer_edge_2', 'lower_lip_outer_edge_1', 'upper_lip_inter_edge_1', 'upper_lip_inter_edge_2', 'upper_lip_inter_edge_3', 'upper_lip_inter_edge_4', 'upper_lip_inter_edge_5', 'lower_lip_inter_edge_3', 'lower_lip_inter_edge_2', 'lower_lip_inter_edge_1']
onehand10k = ['wrist', 'thumb_root', "thumb's first knuckle", "thumb's second knuckle", "thumb's tip", \
        "forefinger's first knuckle", "forefinger's second knuckle", "forefinger's third knuckle", "forefinger's tip", \
        "middle_finger's first knuckle", "middle_finger's second knuckle", "middle_finger's third knuckle", "middle_finger's tip", \
        "ring_finger's first knuckle", "ring_finger's second knuckle", "ring_finger's third knuckle", "ring_finger's tip", \
        "pinky_finger's first knuckle", "pinky_finger's second knuckle", "pinky_finger's third knuckle", "pinky_finger's tip"]
hint = ['wrist', 'thumb_root', "thumb's first knuckle", "thumb's second knuckle", "thumb's tip", \
        "forefinger's first knuckle", "forefinger's second knuckle", "forefinger's third knuckle", "forefinger's tip", \
        "middle_finger's first knuckle", "middle_finger's second knuckle", "middle_finger's third knuckle", "middle_finger's tip", \
        "ring_finger's first knuckle", "ring_finger's second knuckle", "ring_finger's third knuckle", "ring_finger's tip", \
        "pinky_finger's first knuckle", "pinky_finger's second knuckle", "pinky_finger's third knuckle", "pinky_finger's tip"]
hand_xray = ['styloid_process_of_ulna', 'Pisiform_bone', 'Distal_radio_ulnar_joint', 'Distal_radio_radius_joint', 'styloid_process_of_radius', 'Capitate_bone', 'Pisiform_bone', 'left_Capitate_bone', 'right_Capitate_bone', 'Hamate_bone', 'Scaphoid_bone', 'Trapezoid_bone', 'Trapezium_bone', \
             "pinky_finger's root", "ring_finger's root", "middle_finger's root", "forefinger's root", \
             'thumb_root', "thumb's first knuckle", "thumb's second knuckle", "thumb's tip", \
            "forefinger's first knuckle", "forefinger's second knuckle", "forefinger's third knuckle", "forefinger's tip", \
            "middle_finger's first knuckle", "middle_finger's second knuckle", "middle_finger's third knuckle", "middle_finger's tip", \
            "ring_finger's first knuckle", "ring_finger's second knuckle", "ring_finger's third knuckle", "ring_finger's tip", \
            "pinky_finger's first knuckle", "pinky_finger's second knuckle", "pinky_finger's third knuckle", "pinky_finger's tip"]

map_dict = {
    'coco_train': coco,
    'coco_val': coco,
    'animal_pose_dataset': [
        'left eye',
        'right eye',
        'left ear',
        'right ear',
        'nose',
        'throat',
        'withers',
        'tail',
        'left-front leg',
        'right-front leg',
        'left-back leg',
        'right-back leg',
        'left-front knee',
        'right-front knee',
        'left-back knee',
        'right-back knee',
        'left-front paw',
        'right-front paw',
        'left-back paw',
        'right-back paw'
    ],
    'awa_pose': [
        # '_background_',
        'nose',
        'upper jaw',
        'lower jaw',
        'mouth_end_right',
        'mouth_end_left',
        'right eye',
        'right earbase',
        'right earend',
        'right_antler_base',
        'right_antler_end',
        'left eye',
        'left earbase',
        'left earend',
        'left_antler_base',
        'left_antler_end',
        'neck base',
        'neck end',
        'throat base',
        'throat end',
        'back base',
        'back end',
        'back middle',
        'tail base',
        'tail end',
        'front-left thai',
        'front-left knee',
        'front-left paw',
        'front-right thai',
        'front-right paw',
        'front-right knee',
        'back-left knee',
        'back-left paw',
        'back-left thai',
        'back-right thai',
        'back-right paw',
        'back-right knee',
        'belly bottom',
        'body_middle_right',
        'body_middle_left',
    ],
    'cub': [
        'back',
        'beak',
        'belly',
        'breast',
        'crown',
        'forehead',
        'left eye',
        'left leg',
        'left wing',
        'nape',
        'right eye',
        'right leg',
        'right wing',
        'tail',
        'throat',
    ],
    'nabird': [
        'bill',  # namely beak
        'crown',
        'nape',
        'left eye',
        'right eye',
        'belly',
        'breast',
        'back',
        'tail',
        'left wing',
        'right wing',
    ],
    'ap10k': [
        'left eye',
        'right eye',
        'nose',
        'neck',
        'root of tail',
        'left shoulder',
        'left elbow',
        'left-front paw',
        'right shoulder',
        'right elbow',
        'right-front paw',
        'left hip',
        'left knee',
        'left-back paw',
        'right hip',
        'right knee',
        'right-back paw'
    ],
}

def get_mapped_kps_names(dataset_type, origin_kp_types, ORIGIN_FULL_KEYPOINT_TYPES):
    mapped_keypoints = map_dict.get(dataset_type)
    if mapped_keypoints == None:  # if not found, directly return
        return origin_kp_types
    
    kps_names = []
    for each_kp_type in origin_kp_types:
        id = ORIGIN_FULL_KEYPOINT_TYPES.index(each_kp_type)
        kps_names.append(mapped_keypoints[id])

    return kps_names

def get_original_kps_names(dataset_type, mapped_kp_types, ORIGIN_FULL_KEYPOINT_TYPES):
    mapped_keypoints = map_dict.get(dataset_type)
    if mapped_keypoints == None:  # if not found, directly return
        return mapped_kp_types
    
    kps_names = []
    for each_kp_type in mapped_kp_types:
        id = mapped_keypoints.index(each_kp_type)
        kps_names.append(ORIGIN_FULL_KEYPOINT_TYPES[id])

    return kps_names
