export CUDA_VISIBLE_DEVICES=2
CONFIG_FILE=/project/vonneumann1/cl2025/GKD/experiments/gkd_ablation/configs/gkd.yaml
OUTPUT_DIR=/scratch/vonneumann1/cl2025/GKD/gkd_ablation/main4_neg_samples/bilinear_tvm-backup-before-tune-study
original_gt_file=/project/vonneumann1/cl2025/keypoint_datasets/animal_pose/macaque_pose/annotations/macaque_pose_coco0.20.json
output_pred_file=/project/vonneumann1/cl2025/GKD/more_investigate/top_down_get_GKD_json_results/result_json_F_measure/macaque_pose_grounddino_our_results.json

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
    DATASET.TEST_DATA [['macaque_pose','macaque_pose_grounddino_results_F_measure_select.json',[],[]]]

python evaluation_metric/coco_eval_pck.py \
--pred_file  ${output_pred_file} \
--original_gt_file  ${original_gt_file} \