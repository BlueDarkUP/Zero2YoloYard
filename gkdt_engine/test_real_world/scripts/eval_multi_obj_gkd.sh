# Examples for multi-object general keypoint detection (GKD) on real-world images

cd '/your/path/to/General-Keypoint-Detection'  # modify this to your path of General-Keypoint-Detection
echo $(pwd)


python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/alpaca_150.jpg \
    --obj_type 'alpaca' \
    --kps_texts 'left eye' 'right eye' 'left ear' 'right ear' 'nose' 'throat' 'withers' 'tail' 'left-front leg' 'right-front leg' 'left-back leg' 'right-back leg' 'left-front knee' 'right-front knee' 'left-back knee' 'right-back knee' 'left-front paw' 'right-front paw' 'left-back paw' 'right-back paw' \
    --skeleton 5 1 5 2 1 3 2 4 7 8 7 9 9 13 13 17 7 10 10 14 14 18 8 11 11 15 15 19 8 12 12 16 16 20 6 7 6 5 \
    --object_detector locateanything

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/alpaca_150.jpg \
    --obj_type 'alpaca' \
    --kps_texts 'left eye' 'right eye' 'left ear' 'right ear' 'nose' 'throat' 'withers' 'tail' 'left-front leg' 'right-front leg' 'left-back leg' 'right-back leg' 'left-front knee' 'right-front knee' 'left-back knee' 'right-back knee' 'left-front paw' 'right-front paw' 'left-back paw' 'right-back paw' \
    --skeleton 5 1 5 2 1 3 2 4 7 8 7 9 9 13 13 17 7 10 10 14 14 18 8 11 11 15 15 19 8 12 12 16 16 20 6 7 6 5 \
    --object_detector groundingdino

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/cat_dog.jpg \
    --obj_type 'cat, dog' \
    --object_detector locateanything

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/000000011511.jpg \
    --obj_type 'human' \
    --object_detector locateanything

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/wash_dishes_egocentric.jpg \
    --obj_type 'human_hand' \
    --object_detector locateanything

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/car_penn2_0_1931.jpg \
    --obj_type 'car, bus, truck' \
    --object_detector locateanything

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/pigs_stock_farming.jpg \
    --obj_type 'pig' \
    --object_detector locateanything

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/pet_birds.jpg \
    --obj_type 'bird' \
    --object_detector locateanything

python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/fish_swim.jpg \
    --obj_type 'fish' \
    --object_detector locateanything





python3 test_real_world/multi_obj_gkd_inference.py \
    --cfg_file test_real_world/configs/gkd.yaml \
    --checkpoint output/GKDT-L_for_app/model/gkd_fullset.best \
    --input_im test_real_world/ims1/alpaca_150.jpg \
    --obj_type 'alpaca' \
    --kps_texts 'left eye' 'right eye' 'nose' \
    --support_im test_real_world/ims1/alpaca_150.jpg \
    --support_kps 615 495 483 493 521 549 \
    --skeleton 1 2 1 3 2 3  \
    --object_detector locateanything
