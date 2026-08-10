export CUDA_VISIBLE_DEVICES=2
CONFIG_FILE=/project/vonneumann1/cl2025/GKD/experiments/gkd_ablation/configs/gkd.yaml
OUTPUT_DIR=/scratch/vonneumann1/cl2025/GKD/gkd_ablation/main4_neg_samples/bilinear_tvm-backup-before-tune-study
original_gt_file=/project/vonneumann1/cl2025/keypoint_datasets/vehicle/carfusion/annotations/car_keypoints_test.json
output_pred_file=/project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/result_json_F_measure/carfusion_grounddino_our_results.json

python3 more_investigate/top_down_get_GKD_json_results/main_gkd.py \
    --cfg_file ${CONFIG_FILE} \
    --eval_only  \
    --output_pred_file ${output_pred_file} \
    OUTPUT_DIR ${OUTPUT_DIR} \
    MODEL.DETECTION_HEAD.IM_FEAT_UPSAMPLER.TYPE 'bilinear' \
    TEST.NUM_EPISODES -1 \
    TEST.TEXT_PROMPT_SETTING.NUM_TEXT 1 \
    TEST.NUM_TEST_SHOT 0 \
    TEST.NUM_TEST_QUERY 1 \
    LOAD_CHECKPOINT_TYPE 'best' \
    DATASET.TEST_DATA [['carfusion','carfusion_grounddino_results_F_measure_select.json',[],[]]]

python evaluation_metric/coco_eval_pck.py \
--pred_file  ${output_pred_file} \
--original_gt_file  ${original_gt_file} \


# 原始GT文件统计:
#   - 包含图像: 12761 张
#   - 包含标注: 22531 条
# 从预测文件中提取到 13917 个预测结果
# 预测结果涉及 12752 个不同的图像
# v2:100050 个预测结果
