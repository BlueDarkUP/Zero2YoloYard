import os
import argparse
from yacs.config import CfgNode
import numpy as np

import sys
sys.path.append('.')  # append pwd into system path so that it could find python modules
print(os.getcwd())

from test_real_world.predefined_keypoints import PREDEFINED_KEYPOINTS
from test_real_world.predefined_keypoints import get_prompt_info
from test_real_world.gkd_inference_lib.gkd_inference import GKDInference, demo
from test_real_world.gkd_inference_lib.write_prediction_to_json import COCO_prediction_writer
from test_real_world.gkd_inference_lib.visualize_keypoints import visualize_keypoints

#======================================================================
# Single-object General Keypoint Detection for Real-world Applications
# Paper: https://arxiv.org/abs/2607.00752
# Date: 2026.07.18
#======================================================================

def main():    
    parser = argparse.ArgumentParser(description='General keypoint detection.')
    parser.add_argument('--cfg_file', type=str, default='./test_real_world/configs/gkd.yaml', help='config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--input_im', type=str, default='', help='path of input image to detect')
    parser.add_argument('--bbox_on_input_im', type=int, nargs='*', default=[], help='A list of ROI bboxes like [x1,y1,x2,y2,...], where every four numbers represent the top-left and bottom-right corners of a box. If it is empty, the whole image becomes ROI.')
    parser.add_argument('--obj_type', type=str, default='', help='This will be used to retrieve predefined keypoint texts for the object if users do not want to manually provide kps_texts')
    parser.add_argument('--kps_texts', type=str, nargs='*', default=[], help='keypoint texts')
    parser.add_argument('--support_im', type=str, default='', help='path to 1-shot support image')
    parser.add_argument('--support_kps', type=int, nargs='*', default=[], help='support keypoints in the format of x1,y1,x2,y2,...')
    parser.add_argument('--skeleton', type=int, nargs='*', default=[], help='1-based. There are 2*N_skeleton numbers. Every two form a link. Only used for visualization.')
    parser.add_argument('opts', help='see yaml config files for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(args)

    # input image waiting to detect
    input_im_path = args.input_im
    bbox_on_input_im = args.bbox_on_input_im  # ROIs

    # textual and visual prompt info
    obj_type = args.obj_type  # optional
    kps_texts = args.kps_texts
    support_im_path = args.support_im  # 1-shot image path or ""
    support_kps = args.support_kps  # list of kps for an image, e.g., [x1, y1, x2, y2...] or []
    skeleton = args.skeleton

    # N_t: number of keypoints from textual prompt, N_v: number of keypoints from visual prompt
    kps_texts, skeleton, N_t, N_v = get_prompt_info(obj_type, kps_texts, support_im_path, support_kps, skeleton)
    
    # 1. Init GKDInference
    gkd_inference = GKDInference(cfg_file=args.cfg_file, checkpoint_path=args.checkpoint, opts=args.opts)

    # 2. General keypoint detection
    gkd_inference.gkd_model.set_cost_eval(True)
    # N_bbox x N x 2, N_bbox x N (N_bbox=1 if bbox_on_input_im=[])
    predictions_o, predict_score, w_h_origin = demo(gkd_inference, input_im_path, bbox_on_input_im, support_im_path, support_kps, kps_texts)

    cost = gkd_inference.gkd_model.get_cost_eval()
    avg_it = cost['IT1']
    print('Demo Avg Inference Time:  %.6f sec/im'%(avg_it))

    # 3. Visualization
    json_writer = COCO_prediction_writer()
    json_writer.add_category_entry(
        category_name=obj_type if obj_type != "" else "object1",
        keypoints=kps_texts if N_t > 0 else [f'keypoint {str(i)}' for i in range(N_v)],
        skeleton=np.array(skeleton).reshape(-1,2).tolist() if len(skeleton)>0 else []
    )
    json_writer.add_image_entry(
        file_name=os.path.basename(input_im_path),
        width=w_h_origin[0],
        height=w_h_origin[1]
    )
    json_writer.add_coco_predictions(
        category_id=json_writer.category_id_max,
        image_id=json_writer.image_id_max,
        predictions=predictions_o.numpy(),
        predict_score=predict_score.numpy(),
        bboxes=np.array(bbox_on_input_im).reshape(-1,4) if len(bbox_on_input_im)>0 else np.array([0, 0, *w_h_origin]).reshape(1, 4),
        bbox_scores=None  # if None, bbox scores are 1 by default.
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