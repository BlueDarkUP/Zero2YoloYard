import os
import argparse
from yacs.config import CfgNode
import numpy as np
import torch

import sys
sys.path.append('.')  # append pwd into system path so that it could find python modules
print(os.getcwd())

from test_real_world.predefined_keypoints import get_prompt_info
from test_real_world.gkd_inference_lib.gkd_inference import GKDInference, demo
from test_real_world.gkd_inference_lib.write_prediction_to_json import COCO_prediction_writer
from test_real_world.gkd_inference_lib.visualize_keypoints import visualize_keypoints
from test_real_world.object_detector_lib.grounding_dino_detector import GroundingDINODetector
from test_real_world.object_detector_lib.locateanything_detector import LocateAnythingDetector

import time

#======================================================================
# Multi-object General Keypoint Detection for Real-world Applications
# Paper: https://arxiv.org/abs/2607.00752
# Date: 2026.07.18
#======================================================================

def main():    
    parser = argparse.ArgumentParser(description='General keypoint detection.')
    parser.add_argument('--cfg_file', type=str, default='./test_real_world/configs/gkd.yaml', help='config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--input_im', type=str, default='', help='path of input image to detect')
    parser.add_argument('--object_detector', type=str, default='locateanything', choices=['groundingdino', 'locateanything'], help='object detector used when --obj_type is provided')
    parser.add_argument('--obj_type', type=str, default='', help='comma-separated object names for automatic detection; omit to use the whole image as one ROI')
    parser.add_argument('--kps_texts', type=str, nargs='*', default=[], help='keypoint texts')
    parser.add_argument('--support_im', type=str, default='', help='path to 1-shot support image')
    parser.add_argument('--support_kps', type=int, nargs='*', default=[], help='support keypoints in the format of x1,y1,x2,y2,...')
    parser.add_argument('--skeleton', type=int, nargs='*', default=[], help='1-based. There are 2*N_skeleton numbers. Every two form a link. Only used for visualization.')
    parser.add_argument('opts', help='see yaml config files for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(args)

    # Input image waiting to detect.
    input_im_path = args.input_im
    bbox_on_input_im = []  # ROIs are generated automatically or default to the full image.

    # textual and visual prompt info
    obj_type = args.obj_type  # comma-separated object names for automatic detection
    user_kps_texts = args.kps_texts
    support_im_path = args.support_im  # 1-shot image path or ""
    support_kps = args.support_kps  # list of kps for an image, e.g., [x1, y1, x2, y2...] or []
    user_skeleton = args.skeleton

    object_names = [name.strip() for name in obj_type.split(',') if name.strip()]
    if not input_im_path:
        raise ValueError('--input_im must be provided.')

    if len(user_skeleton) % 2 != 0:
        raise ValueError('--skeleton must contain an even number of values.')

    time_pre = time.time()
    # With no object name, demo() receives an empty ROI list and uses the whole
    # image. Otherwise every requested class is detected automatically.
    detection_entries = []
    if len(object_names) > 0:
        if args.object_detector == 'groundingdino':
            detector = GroundingDINODetector()

            print('G-DINO init Time:  %.6f sec'%(time.time()-time_pre))
            time_pre = time.time()

            for object_name in object_names:
                detections = detector.detect(input_im_path, object_name)
                print(f'GroundingDINO detections for {object_name}: {len(detections)}')
                detection_entries.extend(detections)

            print('G-DINO detect Time:  %.6f sec'%(time.time()-time_pre))
            time_pre = time.time()
        else:
            detector = LocateAnythingDetector()

            print('LA init Time:  %.6f sec'%(time.time()-time_pre))
            time_pre = time.time()

            detection_entries = detector.detect(input_im_path, object_names)
            print(f'LocateAnything detections: {len(detection_entries)}')

            print('LA detect Time:  %.6f sec'%(time.time()-time_pre))
            time_pre = time.time()
        if not detection_entries:
            raise RuntimeError('Found no objects for the requested --obj_type values.')
        bbox_on_input_im = [coordinate for entry in detection_entries for coordinate in entry['bbox']]
        print(f'Using {len(detection_entries)} detected ROI boxes for GKD.')
    else:
        print('No --obj_type supplied; using the whole image as one GKD ROI.')

    # clean up GPU memory after objection detection
    torch.cuda.empty_cache()

    # 1. Init GKDInference
    gkd_inference = GKDInference(cfg_file=args.cfg_file, checkpoint_path=args.checkpoint, opts=args.opts)

    print('GKD init Time:  %.6f sec'%(time.time()-time_pre))
    time_pre = time.time()

    # 2. General keypoint detection. Different object classes may have
    # different keypoint schemas, so each class is processed in its own call.
    gkd_inference.gkd_model.set_cost_eval(True)
    object_groups = object_names if object_names else ['object1']
    inference_results = []
    w_h_origin = None
    for object_name in object_groups:
        object_entries = [entry for entry in detection_entries if entry['object_name'] == object_name]
        if object_names and not object_entries:
            print(f'No detected {object_name}; skipping GKD inference for this class.')
            continue

        kps_texts, skeleton, N_t, N_v = get_prompt_info(
            object_name, user_kps_texts, support_im_path, support_kps, user_skeleton
        )
        object_bboxes = [coordinate for entry in object_entries for coordinate in entry['bbox']]
        predictions_o, predict_score, w_h_origin = demo(
            gkd_inference, input_im_path, object_bboxes, support_im_path,
            support_kps, kps_texts
        )
        if not object_entries:  # namely if object_entries is []
            object_entries = [{
                'bbox': [0.0, 0.0, float(w_h_origin[0] - 1), float(w_h_origin[1] - 1)],
                'score': 1.0,
                'object_name': object_name,
            }]
        inference_results.append({
            'object_name': object_name,
            'entries': object_entries,
            'kps_texts': kps_texts,
            'skeleton': skeleton,
            'N_t': N_t,
            'N_v': N_v,
            'predictions': predictions_o.numpy(),
            'predict_score': predict_score.numpy(),
        })

    print('GKD detect Time:  %.6f sec'%(time.time()-time_pre))
    time_pre = time.time()

    cost = gkd_inference.gkd_model.get_cost_eval()
    avg_it = cost['IT1']
    print('Demo Avg Inference Time:  %.6f sec/im'%(avg_it))

    # 3. Visualization
    json_writer = COCO_prediction_writer()
    for result in inference_results:
        category_keypoints = result['kps_texts'] if result['N_t'] else [
            f'keypoint {index}' for index in range(result['N_v'])
        ]
        category_skeleton = np.array(result['skeleton']).reshape(-1, 2).tolist() if result['skeleton'] else []
        json_writer.add_category_entry(
            result['object_name'], keypoints=category_keypoints, skeleton=category_skeleton
        )
    json_writer.add_image_entry(
        file_name=os.path.basename(input_im_path),
        width=w_h_origin[0],
        height=w_h_origin[1]
    )
    category_id_by_name = {
        category['name']: category['id'] for category in json_writer.pred_data['categories']
    }
    for result in inference_results:
        for index, entry in enumerate(result['entries']):
            x1, y1, x2, y2 = entry['bbox']
            # COCO stores bboxes as [x, y, width, height], while GKD receives xyxy.
            coco_bbox = np.array([[x1, y1, x2 - x1, y2 - y1]], dtype=np.float64)
            json_writer.add_coco_predictions(
                category_id=category_id_by_name[entry['object_name']],
                image_id=json_writer.image_id_max,
                predictions=result['predictions'][index:index + 1],
                predict_score=result['predict_score'][index:index + 1],
                bboxes=coco_bbox,
                bbox_scores=np.array([entry['score']], dtype=np.float64)
            )

    # Save COCO json file 
    input_im_root = os.path.dirname(input_im_path)
    output_root = input_im_root + '_out'
    predict_json_path = os.path.join(output_root, "result.json")
    json_writer.save(predict_json_path)
    
    visualize_keypoints(
        json_file=predict_json_path,
        image_root=input_im_root,
        output_dir=output_root,
        kp_score_thresh=0.05,  # kp score threshold to show kps
        draw_skeleton=True
    )

if __name__ == '__main__':
    main()