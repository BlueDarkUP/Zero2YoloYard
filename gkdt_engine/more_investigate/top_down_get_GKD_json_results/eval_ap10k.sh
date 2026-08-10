export CUDA_VISIBLE_DEVICES=0
CONFIG_FILE=/project/vonneumann1/cl2025/GKD/experiments/gkd_ablation/configs/gkd.yaml
OUTPUT_DIR=/scratch/vonneumann1/cl2025/GKD/gkd_ablation/main4_neg_samples/bilinear_tvm-backup-before-tune-study
original_gt_file=/project/vonneumann1/cl2025/keypoint_datasets/animal_pose/ap10k/annotations_fskd/ap10k_split_test.json
output_pred_file=/project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/result_json_F_measure/ap10k_grounddino_our_results.json

python3 more_investigate/top_down_get_GKD_json_results/main_gkd.py \
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
    DATASET.TEST_DATA [['ap10k','ap10k_grounddino_results_F_measure_select.json',[],[]]]

python evaluation_metric/coco_eval_pck.py \
--pred_file  ${output_pred_file} \
--original_gt_file  ${original_gt_file} 



# 原始GT文件统计:
#   - 包含图像: 2883 张
#   - 包含标注: 3431 条
# 原始grounding dino文件统计:
#   - 涉及图像: 2860 张
#   - 涉及标注: 3804 条



# python3 -m debugpy --listen 10.22.4.157:4567 --wait-for-client \
# more_investigate/top_down_get_GKD_json_results/main_gkd.py \
#     --cfg_file ${CONFIG_FILE} \
#     --eval_only  \
#     --output_pred_file ${output_pred_file} \
#     OUTPUT_DIR ${OUTPUT_DIR} \
#     MODEL.ENCODER.DINOv3.VISUAL_ENCODER 'dinov3_vitl16' \
#     MODEL.DETECTION_HEAD.IM_FEAT_UPSAMPLER.TYPE 'bilinear' \
#     TEST.NUM_EPISODES -1 \
#     TEST.TEXT_PROMPT_SETTING.NUM_TEXT 1 \
#     TEST.NUM_TEST_SHOT 0 \
#     TEST.NUM_TEST_QUERY 1 \
#     LOAD_CHECKPOINT_TYPE 'best' \
#     DATASET.TEST_DATA [['ap10k','ap10k_grounddino_results_v2.json',[],[]]]



