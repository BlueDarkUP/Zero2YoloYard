# formats/readers/coco_reader.py
import os
import json
from typing import List, Tuple
from collections import defaultdict
from .base_reader import BaseReader
from ..internal_data import ImageAnnotation, Annotation, BBox


class CocoReader(BaseReader):
    """用于读取 COCO 数据集格式的读取器。"""

    def read(self) -> Tuple[List[ImageAnnotation], List[str]]:
        annotations_dir = os.path.join(self.dataset_path, 'annotations')

        if not os.path.isdir(annotations_dir):
            raise FileNotFoundError("COCO数据集必须包含 'annotations' 文件夹。")

        json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError("在 'annotations' 文件夹中未找到JSON文件。")

        json_path = os.path.join(annotations_dir, json_files[0])

        # --- 修改点: 为JSON文件添加UTF-8编码 ---
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        categories = coco_data['categories']
        coco_id_to_name = {cat['id']: cat['name'] for cat in categories}
        # 确保按ID排序以获得一致的类名列表
        sorted_cats = sorted(categories, key=lambda x: x['id'])
        class_names = [cat['name'] for cat in sorted_cats]
        coco_id_to_internal_id = {cat['id']: i for i, cat in enumerate(sorted_cats)}

        annotations_by_image_id = defaultdict(list)
        for ann in coco_data['annotations']:
            annotations_by_image_id[ann['image_id']].append(ann)

        all_annotations = []
        for img_info in coco_data['images']:
            image_id = img_info['id']
            file_name = img_info['file_name']
            width = img_info['width']
            height = img_info['height']

            image_path = self._find_image_path(file_name)
            if not image_path:
                print(f"警告: 找不到图像文件 {file_name}，已跳过。")
                continue

            annotations_for_image = []
            for ann in annotations_by_image_id.get(image_id, []):
                coco_cat_id = ann['category_id']
                # 处理可能不存在的类别ID
                if coco_cat_id not in coco_id_to_internal_id:
                    continue
                class_id = coco_id_to_internal_id[coco_cat_id]

                x_min, y_min, bbox_w_px, bbox_h_px = ann['bbox']

                x_center = (x_min + bbox_w_px / 2) / width
                y_center = (y_min + bbox_h_px / 2) / height
                bbox_w = bbox_w_px / width
                bbox_h = bbox_h_px / height

                bbox = BBox(x_center=x_center, y_center=y_center, width=bbox_w, height=bbox_h)
                annotations_for_image.append(Annotation(class_id=class_id, bbox=bbox))

            all_annotations.append(ImageAnnotation(
                image_path=image_path,
                width=width,
                height=height,
                annotations=annotations_for_image
            ))

        return all_annotations, class_names

    def _find_image_path(self, file_name):
        common_dirs = ['.', 'images', 'train2017', 'val2017', 'test2017', 'train', 'valid', 'test', 'data']
        for directory in common_dirs:
            path = os.path.join(self.dataset_path, directory, file_name)
            if os.path.exists(path):
                return path
        return None