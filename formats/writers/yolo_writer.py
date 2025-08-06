# formats/writers/yolo_writer.py
import os
from .base_writer import BaseWriter
from ..internal_data import ImageAnnotation


class YoloWriter(BaseWriter):
    def setup_directories(self):
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.labels_dir = os.path.join(self.output_dir, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def write(self, image_data_or_path, image_annotation: ImageAnnotation, new_filename_base: str):
        ext = os.path.splitext(image_annotation.image_path)[1]
        if not ext: ext = '.jpg'  # Fallback
        new_image_filename = new_filename_base + ext
        new_label_filename = new_filename_base + '.txt'

        image_dest_path = os.path.join(self.images_dir, new_image_filename)
        self._save_image(image_data_or_path, image_dest_path)

        label_path = os.path.join(self.labels_dir, new_label_filename)
        # --- 修改点: 添加UTF-8编码以确保一致性 ---
        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in image_annotation.annotations:
                bbox = ann.bbox
                f.write(f"{ann.class_id} {bbox.x_center:.6f} {bbox.y_center:.6f} {bbox.width:.6f} {bbox.height:.6f}\n")