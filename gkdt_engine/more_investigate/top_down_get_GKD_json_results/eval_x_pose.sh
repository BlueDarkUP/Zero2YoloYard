python evaluation_metric/coco_eval_pck.py \
--pred_file  /project/vonneumann1/cyx/X-pose/output_json_file/filter_json_file/ap10k_x_pose_results.json \
--original_gt_file  /project/vonneumann1/cl2025/keypoint_datasets/animal_pose/ap10k/annotations_fskd/ap10k_split_test.json



python evaluation_metric/coco_eval_pck.py \
--pred_file  /project/vonneumann1/cyx/X-pose/output_json_file/filter_json_file/macaque_pose_x_pose_results.json \
--original_gt_file  /project/vonneumann1/cl2025/keypoint_datasets/animal_pose/macaque_pose/annotations/macaque_pose_coco0.20.json

python evaluation_metric/coco_eval_pck.py \
--pred_file  /project/vonneumann1/cyx/X-pose/output_json_file/filter_json_file/carfusion_x_pose_results.json \
--original_gt_file  /project/vonneumann1/cl2025/keypoint_datasets/vehicle/carfusion/annotations/car_keypoints_test.json
