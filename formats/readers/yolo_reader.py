# formats/readers/yolo_reader.py
import os
import cv2
import yaml
from typing import List, Tuple
from .base_reader import BaseReader
from ..internal_data import ImageAnnotation, Annotation, BBox


class YoloReader(BaseReader):
    def read(self) -> Tuple[List[ImageAnnotation], List[str]]:
        all_annotations = []
        class_names = self._load_class_names()

        for split in ['train', 'valid', 'test']:
            image_dir = os.path.join(self.dataset_path, split, 'images')
            label_dir = os.path.join(self.dataset_path, split, 'labels')

            if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
                continue

            for filename in os.listdir(image_dir):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

                img = cv2.imread(image_path)
                if img is None: continue
                height, width, _ = img.shape

                annotations_for_image = []
                if os.path.exists(label_path):
                    # --- 修改点 1: 为标签文件添加UTF-8编码 ---
                    with open(label_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_c, y_c, w, h = map(float, parts[1:5])
                                bbox = BBox(x_center=x_c, y_center=y_c, width=w, height=h)
                                annotations_for_image.append(Annotation(class_id=class_id, bbox=bbox))

                all_annotations.append(ImageAnnotation(
                    image_path=image_path,
                    width=width,
                    height=height,
                    annotations=annotations_for_image
                ))

        if not class_names and all_annotations:
            print("警告: 未找到类名文件 (data.yaml 或 names.txt)。将使用通用类名。")
            max_id = 0
            for ann in all_annotations:
                for item in ann.annotations:
                    if item.class_id > max_id:
                        max_id = item.class_id
            class_names = [f'class_{i}' for i in range(max_id + 1)]

        return all_annotations, class_names

    def _load_class_names(self):
        # --- 修改点 2: 为 data.yaml 和 names.txt 添加UTF-8编码 ---
        yaml_path = os.path.join(self.dataset_path, 'data.yaml')
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    return data.get('names', [])
            except Exception:
                pass  # 如果解析失败，尝试下一个方法

        names_path = os.path.join(self.dataset_path, 'names.txt')
        if os.path.exists(names_path):
            with open(names_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

        return []