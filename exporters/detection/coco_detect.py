import os
import json
import shutil
import cv2
import logging
from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData
import file_storage

@ExporterRegistry.register
class COCODetectionExporter(BaseExporter):
    key = "coco_detection"
    display_name = "COCO JSON (Object Detection)"
    annotation_types = ["detection"]

    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        images_dir = os.path.join(export_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        categories = [{"id": i + 1, "name": name, "supercategory": "object"} for i, name in enumerate(class_list)]
        class_map = {name: i + 1 for i, name in enumerate(class_list)}

        coco_data = {
            "info": {"description": "Exported from Zero2YoloYard"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }

        ann_id = 1
        for img_id, frame_info in enumerate(frames_data, start=1):
            file_name = f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}.jpg"
            src_img_path = file_storage.get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])
            dst_img_path = os.path.join(images_dir, file_name)

            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)

            width = frame_info['width']
            height = frame_info['height']

            coco_data["images"].append({
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })

            annotations: AnnotationData = frame_info["annotations"]
            for obj in annotations.get_bboxes():
                if obj.label in class_map:
                    x1, y1, x2, y2 = obj.bbox
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_map[obj.label],
                        "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                        "area": round(area, 2),
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id += 1

        annotations_json_path = os.path.join(export_dir, "annotations.json")
        with open(annotations_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
