# formats/readers/voc_reader.py
import os
import cv2
import xml.etree.ElementTree as ET
from typing import List, Tuple
from .base_reader import BaseReader
from ..internal_data import ImageAnnotation, Annotation, BBox


class VocReader(BaseReader):
    """用于读取 Pascal VOC 数据集格式的读取器。"""

    def read(self) -> Tuple[List[ImageAnnotation], List[str]]:
        annotations_dir = os.path.join(self.dataset_path, 'Annotations')
        images_dir = os.path.join(self.dataset_path, 'JPEGImages')

        if not os.path.isdir(annotations_dir) or not os.path.isdir(images_dir):
            raise FileNotFoundError("VOC数据集必须包含 'Annotations' 和 'JPEGImages' 文件夹。")

        all_annotations = []
        class_names = set()

        xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

        # 第一次遍历以收集所有类名
        for xml_file in xml_files:
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_names.add(class_name)

        sorted_class_names = sorted(list(class_names))
        class_to_id = {name: i for i, name in enumerate(sorted_class_names)}

        # 第二次遍历以构建完整的标注数据
        for xml_file in xml_files:
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
            root = tree.getroot()

            filename = root.find('filename').text
            image_path = os.path.join(images_dir, filename)

            if not os.path.exists(image_path):
                continue

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            annotations_for_image = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = class_to_id[class_name]

                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # 将 VOC 格式 (xmin, ymin, xmax, ymax) 转换为标准化的中心坐标格式
                bbox_w = (xmax - xmin) / width
                bbox_h = (ymax - ymin) / height
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height

                bbox = BBox(x_center=x_center, y_center=y_center, width=bbox_w, height=bbox_h)
                annotations_for_image.append(Annotation(class_id=class_id, bbox=bbox))

            all_annotations.append(ImageAnnotation(
                image_path=image_path,
                width=width,
                height=height,
                annotations=annotations_for_image
            ))

        return all_annotations, sorted_class_names