# formats/writers/voc_writer.py
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from .base_writer import BaseWriter
from ..internal_data import ImageAnnotation


class VocWriter(BaseWriter):
    """用于写入 PASCAL VOC 数据集格式的写入器。"""

    def setup_directories(self):
        """创建 'Annotations' 和 'JPEGImages' 目录。"""
        self.annotations_dir = os.path.join(self.output_dir, 'Annotations')
        self.images_dir = os.path.join(self.output_dir, 'JPEGImages')
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def write(self, image_data_or_path, image_annotation: ImageAnnotation, new_filename_base: str):
        """
        将单个图像及其标注写入为 VOC 格式的 .xml 和 .jpg 文件。
        """
        # VOC 格式通常使用 .jpg 作为图像扩展名
        ext = '.jpg'
        new_image_filename = new_filename_base + ext
        new_xml_filename = new_filename_base + '.xml'

        # 1. 保存图像文件到 JPEGImages/ 目录
        image_dest_path = os.path.join(self.images_dir, new_image_filename)
        self._save_image(image_data_or_path, image_dest_path)

        # 2. 创建 XML 树结构
        annotation = Element('annotation')
        SubElement(annotation, 'folder').text = 'JPEGImages'
        SubElement(annotation, 'filename').text = new_image_filename
        SubElement(annotation, 'path').text = os.path.abspath(image_dest_path)

        source = SubElement(annotation, 'source')
        SubElement(source, 'database').text = 'Unknown'

        size = SubElement(annotation, 'size')
        SubElement(size, 'width').text = str(image_annotation.width)
        SubElement(size, 'height').text = str(image_annotation.height)
        SubElement(size, 'depth').text = '3'

        SubElement(annotation, 'segmented').text = '0'

        # 3. 为每个边界框添加 <object> 元素
        for ann in image_annotation.annotations:
            bbox = ann.bbox
            class_id = ann.class_id

            # 将标准内部格式 [center_x, center_y, w, h] (归一化)
            # 转换为 VOC 格式 [xmin, ymin, xmax, ymax] (像素值)
            xmin = (bbox.x_center * image_annotation.width) - (bbox.width * image_annotation.width / 2)
            ymin = (bbox.y_center * image_annotation.height) - (bbox.height * image_annotation.height / 2)
            xmax = xmin + (bbox.width * image_annotation.width)
            ymax = ymin + (bbox.height * image_annotation.height)

            # 边界检查，确保坐标在图像范围内，并转换为整数
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(image_annotation.width, int(xmax))
            ymax = min(image_annotation.height, int(ymax))

            obj = SubElement(annotation, 'object')
            SubElement(obj, 'name').text = self.class_names[class_id]
            SubElement(obj, 'pose').text = 'Unspecified'
            SubElement(obj, 'truncated').text = '0'
            SubElement(obj, 'difficult').text = '0'

            bndbox = SubElement(obj, 'bndbox')
            SubElement(bndbox, 'xmin').text = str(xmin)
            SubElement(bndbox, 'ymin').text = str(ymin)
            SubElement(bndbox, 'xmax').text = str(xmax)
            SubElement(bndbox, 'ymax').text = str(ymax)

        # 4. 格式化XML并保存到 Annotations/ 目录
        # 使用 minidom 来美化输出，使其具有缩进，更易于人类阅读
        xml_str = tostring(annotation, 'utf-8')
        dom = parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="    ")

        xml_path = os.path.join(self.annotations_dir, new_xml_filename)
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)