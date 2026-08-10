export CUDA_VISIBLE_DEVICES=0
CONFIG_FILE=/project/vonneumann1/cl2025/GKD/experiments/gkd_ablation/configs/gkd.yaml
OUTPUT_DIR=/scratch/vonneumann1/cl2025/GKD/gkd_ablation/main4_neg_samples/bilinear_tvm-backup-before-tune-study
original_gt_file=/project/vonneumann1/cl2025/keypoint_datasets/human_pose/human_art/annotations/validation_humanart.json
output_pred_file=/project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/result_json/validation_humanart_x_pose_bbox_our_result.json

#对于coco和humanart应该使用main_gkd_flip_test_gaussian.py这个文件来eval，因为是human_pose
python3 more_investigate/top_down_get_GKD_json_results/main_gkd_flip_test_gaussian.py \
    --cfg_file ${CONFIG_FILE} \
    --eval_only  \
    --output_pred_file ${output_pred_file} \
    OUTPUT_DIR ${OUTPUT_DIR} \
    MODEL.ENCODER.DINOv3.VISUAL_ENCODER 'dinov3_vitl16' \
    MODEL.DETECTION_HEAD.IM_FEAT_UPSAMPLER.TYPE 'bilinear' \
    TEST.NUM_EPISODES -1 \
    TEST.TEXT_PROMPT_SETTING.NUM_TEXT 1 \
    TEST.NUM_TEST_SHOT 0 \
    TEST.NUM_TEST_QUERY 1 \
    LOAD_CHECKPOINT_TYPE 'best' \
    DATASET.PAD_KPS.SAME_AT_TEST True \
    DATASET.TRAIN_DATA [['vinegar_fly','fly0.20.json',[],[]]]   \
    DATASET.VAL_DATA [['vinegar_fly','fly0.20.json',[],[]]]   \
    DATASET.TEST_DATA [['human_art','validation_humanart_x_pose.json',[],[]]]

# python evaluation_metric/coco_eval_pck.py \
# --pred_file  ${output_pred_file} \
# --original_gt_file  ${original_gt_file} 

python evaluation_metric/coco_eval_ap.py  \
--pred_file  ${output_pred_file} \
--original_gt_file  ${original_gt_file}


python evaluation_metric/coco_eval_ap.py  \
--pred_file  /project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/humanart_our_score/validation_humanart_x_pose_bbox_our_result.json \
--original_gt_file  /project/vonneumann1/cl2025/keypoint_datasets/human_pose/human_art/annotations/validation_humanart.json


