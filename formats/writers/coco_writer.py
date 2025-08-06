# formats/writers/coco_writer.py
import os
import json
import datetime
from .base_writer import BaseWriter
from ..internal_data import ImageAnnotation


class CocoWriter(BaseWriter):
    """用于写入 COCO JSON 数据集格式的写入器。"""

    def __init__(self, output_dir: str, class_names: list):
        # COCO 的 __init__ 与其他写入器不同，它需要收集所有数据
        super().__init__(output_dir, class_names)

        # 初始化 COCO JSON 的基本结构
        self.coco_data = {
            "info": {
                "year": datetime.date.today().year,
                "version": "1.0",
                "description": "Exported from Universal Data Augmentor",
                "date_created": datetime.datetime.utcnow().isoformat(' ')
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "categories": self._create_categories(),
            "images": [],
            "annotations": []
        }

        # 用于生成唯一ID的计数器
        self.image_id_counter = 1
        self.annotation_id_counter = 1

    def _create_categories(self) -> list:
        """根据类名列表创建 COCO 格式的类别列表。"""
        categories = []
        # COCO 格式的类别ID通常从1开始
        for i, class_name in enumerate(self.class_names):
            categories.append({
                "id": i + 1,
                "name": class_name,
                "supercategory": "none"
            })
        return categories

    def setup_directories(self):
        """创建 'annotations' 和 'data' 目录。"""
        self.annotations_dir = os.path.join(self.output_dir, 'annotations')
        self.images_dir = os.path.join(self.output_dir, 'data')  # COCO 常用 'data' 作为图像文件夹
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def write(self, image_data_or_path, image_annotation: ImageAnnotation, new_filename_base: str):
        """
        此方法在每次调用时，并不会立即写入文件，而是将数据追加到内存中的coco_data字典。
        真正的文件写入操作在 finalize() 方法中完成。
        """
        # 1. 准备并保存图像文件
        ext = os.path.splitext(image_annotation.image_path)[1]
        new_image_filename = new_filename_base + ext
        image_dest_path = os.path.join(self.images_dir, new_image_filename)
        self._save_image(image_data_or_path, image_dest_path)

        # 2. 创建并添加 'image' 对象
        image_id = self.image_id_counter
        image_info = {
            "id": image_id,
            "file_name": new_image_filename,
            "width": image_annotation.width,
            "height": image_annotation.height,
            "license": 1,
            "date_captured": "unknown"
        }
        self.coco_data['images'].append(image_info)

        # 3. 创建并添加 'annotation' 对象
        for ann in image_annotation.annotations:
            bbox = ann.bbox

            # 将标准内部格式 [center_x, center_y, w, h] (归一化)
            # 转换为 COCO 格式 [x_min, y_min, width, height] (像素值)
            w_px = bbox.width * image_annotation.width
            h_px = bbox.height * image_annotation.height
            x_min = (bbox.x_center * image_annotation.width) - (w_px / 2)
            y_min = (bbox.y_center * image_annotation.height) - (h_px / 2)

            annotation_info = {
                "id": self.annotation_id_counter,
                "image_id": image_id,
                "category_id": ann.class_id + 1,  # 将0-indexed的class_id转为1-indexed的category_id
                "bbox": [x_min, y_min, w_px, h_px],
                "area": w_px * h_px,
                "iscrowd": 0,
                "segmentation": []  # 对于边界框检测，此项为空
            }
            self.coco_data['annotations'].append(annotation_info)
            self.annotation_id_counter += 1

        self.image_id_counter += 1

    def finalize(self):
        """
        在所有图像都处理完毕后调用此方法，将内存中的数据一次性写入JSON文件。
        """
        # 定义最终的JSON文件名
        json_filename = 'instances.json'
        json_path = os.path.join(self.annotations_dir, json_filename)

        # 写入文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, ensure_ascii=False, indent=4)
        print(f"COCO JSON file saved to {json_path}")