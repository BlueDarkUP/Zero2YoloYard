# Examples for single-object general keypoint detection (GKD) on real-world images

cd '/your/path/to/General-Keypoint-Detection'  # modify this to your path of General-Keypoint-Detection
echo $(pwd)

python3 test_real_world/single_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/2007_007524.jpg \
    --bbox_on_input_im \
    --obj_type "" \
    --kps_texts 'left eye' 'right eye' 'nose' \
    --support_im test_real_world/ims1/2007_003778.jpg \
    --support_kps 343 166 281 158 311 197 \
    --skeleton 1 2 1 3 2 3

python3 test_real_world/single_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/2007_007524.jpg \
    --bbox_on_input_im 33 38 241 310 \
    --obj_type '' \
    --kps_texts 'nose' 'left eye' 'right eye' 'left ear' 'right ear' \
    --skeleton 1 2 1 3 2 3 2 4 3 5

python3 test_real_world/single_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/2007_007524.jpg \
    --bbox_on_input_im \
    --obj_type '' \
    --kps_texts 'left eye' 'right eye' 'nose' \
    --support_im test_real_world/ims1/alpaca_150.jpg \
    --support_kps 615 495 483 493 521 549 \
    --skeleton 1 2 1 3 2 3
    # MODEL.ENCODER.DINOv3.VISUAL_ENCODER dinov3_vith16plus 




