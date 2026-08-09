import os
import json
import shutil
from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData, COCO_POSE_17_SCHEMA
import file_storage

@ExporterRegistry.register
class COCOPoseExporter(BaseExporter):
    key = "coco_pose"
    display_name = "COCO Keypoints / Pose (17 Skeleton Joints JSON)"
    annotation_types = ["pose"]

    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        images_dir = os.path.join(export_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        coco_pose_data = {
            "info": {"description": "COCO 17 Keypoints Export from Zero2YoloYard"},
            "categories": [{
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": COCO_POSE_17_SCHEMA["keypoints"],
                "skeleton": COCO_POSE_17_SCHEMA["skeleton"]
            }],
            "images": [],
            "annotations": []
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

            coco_pose_data["images"].append({
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })

            annotations: AnnotationData = frame_info["annotations"]
            for obj in annotations.get_keypoints():
                x1, y1, x2, y2 = obj.bbox if obj.bbox else (0, 0, width, height)
                w, h = max(0, x2 - x1), max(0, y2 - y1)

                keypoints_flat = []
                num_keypoints = 0
                for kp in obj.keypoints:
                    px, py, v = kp.get("x", 0), kp.get("y", 0), kp.get("v", 0)
                    if v > 0:
                        num_keypoints += 1
                    keypoints_flat.extend([round(px, 2), round(py, 2), v])

                coco_pose_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "keypoints": keypoints_flat,
                    "num_keypoints": num_keypoints,
                    "iscrowd": 0
                })
                ann_id += 1

        with open(os.path.join(export_dir, "person_keypoints.json"), 'w', encoding='utf-8') as f:
            json.dump(coco_pose_data, f, indent=2, ensure_ascii=False)
