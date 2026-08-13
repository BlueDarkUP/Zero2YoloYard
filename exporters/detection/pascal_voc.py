import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData
import file_storage

@ExporterRegistry.register
class PascalVOCExporter(BaseExporter):
    key = "pascal_voc_detection"
    display_name = "Pascal VOC XML (Object Detection)"
    annotation_types = ["detection"]

    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        images_dir = os.path.join(export_dir, "JPEGImages")
        annotations_dir = os.path.join(export_dir, "Annotations")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        for frame_info in frames_data:
            base_name = f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"
            img_filename = f"{base_name}.jpg"
            xml_filename = f"{base_name}.xml"

            src_img_path = file_storage.get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])
            dst_img_path = os.path.join(images_dir, img_filename)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)

            annotation_elem = ET.Element("annotation")
            ET.SubElement(annotation_elem, "filename").text = img_filename

            size_elem = ET.SubElement(annotation_elem, "size")
            ET.SubElement(size_elem, "width").text = str(frame_info['width'])
            ET.SubElement(size_elem, "height").text = str(frame_info['height'])
            ET.SubElement(size_elem, "depth").text = "3"

            annotations: AnnotationData = frame_info["annotations"]
            for obj in annotations.get_bboxes():
                if class_list and obj.label not in class_list:
                    continue

                object_elem = ET.SubElement(annotation_elem, "object")
                ET.SubElement(object_elem, "name").text = obj.label
                ET.SubElement(object_elem, "pose").text = "Unspecified"
                ET.SubElement(object_elem, "truncated").text = "0"
                ET.SubElement(object_elem, "difficult").text = "0"

                bndbox = ET.SubElement(object_elem, "bndbox")
                x1, y1, x2, y2 = obj.bbox if obj.bbox else (0, 0, frame_info['width'], frame_info['height'])
                x1_c = max(0, min(frame_info['width'], int(x1)))
                y1_c = max(0, min(frame_info['height'], int(y1)))
                x2_c = max(0, min(frame_info['width'], int(x2)))
                y2_c = max(0, min(frame_info['height'], int(y2)))

                ET.SubElement(bndbox, "xmin").text = str(x1_c)
                ET.SubElement(bndbox, "ymin").text = str(y1_c)
                ET.SubElement(bndbox, "xmax").text = str(x2_c)
                ET.SubElement(bndbox, "ymax").text = str(y2_c)

            xml_str = minidom.parseString(ET.tostring(annotation_elem)).toprettyxml(indent="  ")
            with open(os.path.join(annotations_dir, xml_filename), "w", encoding="utf-8") as f:
                f.write(xml_str)
