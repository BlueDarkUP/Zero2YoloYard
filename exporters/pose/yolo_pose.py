import os
import shutil
import cv2
import yaml
import json
import numpy as np
from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData, COCO_POSE_17_SCHEMA
import file_storage

@ExporterRegistry.register
class YOLOPoseExporter(BaseExporter):
    key = "yolo_pose"
    display_name = "YOLOv8 / YOLOv11 Pose (TXT Format)"
    annotation_types = ["pose"]

    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        images_dir = os.path.join(export_dir, "images")
        labels_dir = os.path.join(export_dir, "labels")

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

        class_map = {name: i for i, name in enumerate(class_list)}
        if not class_map:
            class_map = {"person": 0}

        # Determine actual max keypoints count from frames_data
        max_kpts_count = 0
        for frame_info in frames_data:
            ann: AnnotationData = frame_info.get("annotations")
            if ann:
                for obj in ann.get_keypoints():
                    if obj.keypoints:
                        max_kpts_count = max(max_kpts_count, len(obj.keypoints))
        if max_kpts_count == 0:
            max_kpts_count = 17

        eval_percent = options.get('eval_percent', 20.0)
        test_percent = options.get('test_percent', 10.0)
        total_count = len(frames_data)
        val_count = int(total_count * eval_percent / 100.0)
        test_count = int(total_count * test_percent / 100.0)

        val_data = frames_data[:val_count]
        test_data = frames_data[val_count:val_count + test_count]
        train_data = frames_data[val_count + test_count:]

        split_groups = [('val', val_data), ('test', test_data), ('train', train_data)]

        for split, split_frames in split_groups:
            for frame_info in split_frames:
                video_uuid = frame_info['video_uuid']
                frame_num = frame_info['frame_number']
                width = frame_info['width']
                height = frame_info['height']

                base_name = f"{video_uuid}_{frame_num:05d}"
                img_filename = f"{base_name}.jpg"
                txt_filename = f"{base_name}.txt"

                src_img_path = file_storage.get_frame_path(video_uuid, frame_num)
                dst_img_path = os.path.join(images_dir, split, img_filename)
                dst_txt_path = os.path.join(labels_dir, split, txt_filename)

                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)

                annotations: AnnotationData = frame_info["annotations"]
                lines = []

                for obj in annotations.get_keypoints():
                    cls_id = class_map.get(obj.label, 0)
                    
                    # Get or compute bbox
                    if obj.bbox and len(obj.bbox) == 4:
                        x1, y1, x2, y2 = obj.bbox
                    else:
                        # Compute bbox from visible keypoints
                        v_kpts = [k for k in obj.keypoints if k.get('v', 2) > 0]
                        pts = v_kpts if v_kpts else obj.keypoints
                        if not pts:
                            continue
                        min_x = min(p['x'] for p in pts)
                        max_x = max(p['x'] for p in pts)
                        min_y = min(p['y'] for p in pts)
                        max_y = max(p['y'] for p in pts)
                        pad = 10
                        x1, y1, x2, y2 = max(0, min_x - pad), max(0, min_y - pad), min(width, max_x + pad), min(height, max_y + pad)

                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    cx = (x1 + x2) / 2.0 / width
                    cy = (y1 + y2) / 2.0 / height
                    nw = bw / width
                    nh = bh / height

                    # Format keypoints up to max_kpts_count
                    kpt_str_list = []
                    kps = obj.keypoints or []
                    for kp_idx in range(max_kpts_count):
                        if kp_idx < len(kps):
                            kp = kps[kp_idx]
                            kx = max(0.0, min(1.0, kp.get('x', 0) / width))
                            ky = max(0.0, min(1.0, kp.get('y', 0) / height))
                            kv = int(kp.get('v', 2))
                        else:
                            kx, ky, kv = 0.0, 0.0, 0
                        kpt_str_list.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])

                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kpt_str_list))

                with open(dst_txt_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

        # Generate data.yaml
        yaml_content = {
            'path': '.',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'kpt_shape': [max_kpts_count, 3],
            'names': {i: name for i, name in enumerate(class_list)} if class_list else {0: 'person'}
        }

        with open(os.path.join(export_dir, "data.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
