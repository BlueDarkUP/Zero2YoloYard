cd '/your/path/to/General-Keypoint-Detection'  # modify this to your path of General-Keypoint-Detection

export CUDA_VISIBLE_DEVICES=0
CONFIG_FILE=experiments/expert_benchmark/configs/gkd_coco_only.yaml
OUTPUT_DIR=output/expert_benchmark/coco/GKDT-L_ft_backup_16ep_70.5-ft-from-GKD-68.9
original_gt_file=/your/path/to/keypoint_datasets/human_pose/coco/annotations/person_keypoints_val2017.json
output_pred_file=experiments/expert_benchmark/pred_file/result_coco_gkdt_l_ft.json

# python3 -m debugpy --listen 10.22.4.80:4567 --wait-for-client \
python3 evaluation_related/main_gkd_flip_test_gaussian.py \
    --cfg_file ${CONFIG_FILE} \
    --output_pred_file ${output_pred_file} \
    --eval_only  \
    OUTPUT_DIR ${OUTPUT_DIR} \
    MODEL.ENCODER.DINOv3.VISUAL_ENCODER 'dinov3_vitl16' \
    MODEL.DETECTION_HEAD.IM_FEAT_UPSAMPLER.TYPE 'bilinear' \
    TEST.NUM_EPISODES -1 \
    TEST.TEXT_PROMPT_SETTING.NUM_TEXT 1 \
    TEST.NUM_TEST_SHOT 0 \
    TEST.NUM_TEST_QUERY 12 \
    LOAD_CHECKPOINT_TYPE 'best' \
    DATASET.TEST_DATA [['coco_val','person_keypoints_val2017.json',[],["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]]] 


python evaluation_metric/coco_eval_ap.py  \
    --pred_file  ${output_pred_file} \
    --original_gt_file  ${original_gt_file}