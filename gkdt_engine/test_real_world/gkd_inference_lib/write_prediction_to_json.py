#====================================================
# Write GKD predictions into a coco-format json file
# Author: Written by Changsheng Lu
# Date: 2026.07.18
#====================================================

import os
import json
import copy
import numpy as np

class COCO_prediction_writer(object):
    def __init__(self, categories: list=[], images: list=[]):
        '''
        We assume all images are in a shared image root.
        '''
        self.pred_data = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Init categories and images if available
        if len(categories) > 0:
            self.pred_data['categories'] = copy.deepcopy(categories)
        if len(images) > 0:
            self.pred_data['images'] = copy.deepcopy(images)

        self.category_names = []  # category names' pool
        self.category_id_max = 0
        for cat_entry in self.pred_data['categories']:
            self.category_names.append(cat_entry['name'])
            if cat_entry['id'] >= self.category_id_max:
                self.category_id_max = cat_entry['id']
        
        self.file_names = []  # image filenames' pool
        self.image_id_max = 0
        for im_entry in self.pred_data['images']:
            self.file_names.append(im_entry['file_name'])
            if im_entry['id'] >= self.image_id_max:
                self.image_id_max = im_entry['id']

        self.anno_id_max = 0
    
    def add_category_entry(self, category_name: str="", supercategory: str="", keypoints: list=[], skeleton: list=[]):
        if category_name in self.category_names:
            print('The category with the same name existed. Skip adding.')
        else:
            category_entry = {
                "id": self.category_id_max+1,
                "name": category_name,
                "supercategory": supercategory,
                "keypoints": keypoints,
                "skeleton": skeleton,  # 1-based
            }
            self.pred_data['categories'].append(category_entry)
            self.category_names.append(category_name)  # update category_names pool
            self.category_id_max += 1

    def add_image_entry(self, file_name: str="", width=384, height=384):
        if file_name in self.file_names:
            print('The image with the same name existed. Skip adding.')
        else:
            image_entry = {
                "id": self.image_id_max+1,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
            self.pred_data['images'].append(image_entry)
            self.file_names.append(file_name)
            self.image_id_max += 1
    
    def add_coco_predictions(self, category_id: int, image_id: int, predictions: np.ndarray, predict_score: np.ndarray, bboxes: np.ndarray, bbox_scores: np.ndarray=None):
        '''
        Add N_bbox object instances of keypoints from ONE image to json file!

        predictions:   N_bbox x N x 2 (each row is a point (x, y) in [0, image width]x[0, image height])
        predict_score: N_bbox x N (0~about 1)  
        bboxes:        N_bbox x 4 (each row is [xmin, ymin, w, h])
        bbox_scores:   N_bbox (0~1). If None, 1.0 by default.
        '''
        N_bbox, N_kps = predictions.shape[:2]
        bboxes = np.round(bboxes, decimals=2)
        if bbox_scores is None:
            bbox_scores = np.ones(N_bbox, dtype=bboxes.dtype)
        bbox_scores = np.round(bbox_scores, decimals=2)
        for i in range(N_bbox):
            per_bbox = bboxes[i].tolist()
            per_bbox_score = float(bbox_scores[i])      

            # fill predicted keypoints data
            pred_kps = [0.0] * (N_kps * 3)
            for j in range(N_kps):
                x, y = predictions[i, j]
                x, y = float(x), float(y)
                keypoint_score = float(predict_score[i, j])

                pred_kps[j*3+0] = round(x, 2)
                pred_kps[j*3+1] = round(y, 2)
                pred_kps[j*3+2] = round(keypoint_score, 2)

            # record and update
            ann_entry = {
                    "id": self.anno_id_max+1,
                    "category_id": category_id,
                    "image_id": image_id,
                    "bbox": per_bbox,
                    "score": per_bbox_score,
                    "keypoints": pred_kps,
                    "num_keypoints": N_kps
            }
            self.pred_data['annotations'].append(ann_entry)
            self.anno_id_max += 1
    
    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.pred_data, f)

        # print statistics
        total_annotations = len(self.pred_data['annotations'])
        total_images = len(self.pred_data['images'])
        category_names = [cat['name'] for cat in self.pred_data['categories']]
        print("==>Saved COCO statistics:")
        print(f"Save path: {output_path}")
        print(f"Image number: {total_images}")
        print(f"Annotation number: {total_annotations}")
        print(f"Categories: {category_names}")   