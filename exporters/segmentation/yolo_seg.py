import os
import shutil
import cv2
import yaml
import logging
import random
import traceback
import numpy as np
from multiprocessing import Pool, cpu_count
from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData, AnnotationObject
from ..detection.yolo_detect import build_augmentation_pipeline_for_keypoints
import file_storage
import database
import settings_manager

try:
    import albumentations as A
except ImportError:
    A = None


def process_seg_frame_worker(args):
    frame_info, target_img_dir, target_lbl_dir, class_map, augmentation_options = args

    is_augmented = frame_info.get("type") == "augmented"
    augment_pipeline = None
    if is_augmented and augmentation_options and augmentation_options.get("enabled", False):
        augment_pipeline = build_augmentation_pipeline_for_keypoints(augmentation_options)

    try:
        base_filename = f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"
        if is_augmented:
            base_filename = frame_info["augmented_id"]

        src_img_path = file_storage.get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])
        if not os.path.exists(src_img_path):
            logging.warning(f"Source file not found, skipping: {src_img_path}")
            return None

        dst_img_path = os.path.join(target_img_dir, f"{base_filename}.jpg")
        txt_filename = f"{base_filename}.txt"
        dst_lbl_path = os.path.join(target_lbl_dir, txt_filename)

        image_bgr = cv2.imread(src_img_path)
        if image_bgr is None:
            logging.warning(f"Failed to read image at {src_img_path}")
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        width = frame_info['width']
        height = frame_info['height']

        annotations: AnnotationData = frame_info["annotations"]
        polygons_data = []
        for obj in annotations.objects:
            if obj.label not in class_map:
                continue
            if obj.type == "polygon" and obj.polygon:
                polygons_data.append(obj)
            elif obj.type == "bbox" and obj.bbox:
                x1, y1, x2, y2 = obj.bbox
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                polygons_data.append(AnnotationObject(id=obj.id, type="polygon", label=obj.label, points=poly))

        if is_augmented and augment_pipeline and polygons_data:
            flat_kpts = []
            kpt_labels = []
            poly_info = []
            for poly_idx, obj in enumerate(polygons_data):
                poly_pts = obj.polygon
                poly_info.append((class_map[obj.label], len(poly_pts)))
                for pt in poly_pts:
                    flat_kpts.append([float(pt[0]), float(pt[1])])
                    kpt_labels.append(poly_idx)

            transformed = augment_pipeline(image=image_rgb, keypoints=flat_kpts, keypoint_labels=kpt_labels)
            aug_image_rgb = transformed['image']
            aug_kpts = transformed['keypoints']
            aug_kpt_labels = transformed['keypoint_labels']

            aug_h, aug_w, _ = aug_image_rgb.shape
            final_image_bgr = cv2.cvtColor(aug_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_img_path, final_image_bgr)

            lines = []
            for poly_idx, (cls_idx, count) in enumerate(poly_info):
                pts = [aug_kpts[idx] for idx, l_idx in enumerate(aug_kpt_labels) if l_idx == poly_idx]
                if len(pts) >= 3:
                    poly_coords = []
                    for pt in pts:
                        px = max(0.0, min(1.0, pt[0] / aug_w))
                        py = max(0.0, min(1.0, pt[1] / aug_h))
                        poly_coords.extend([f"{px:.6f}", f"{py:.6f}"])
                    lines.append(f"{cls_idx} " + " ".join(poly_coords))

            with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
        else:
            shutil.copy(src_img_path, dst_img_path)
            lines = []
            for obj in polygons_data:
                cls_idx = class_map[obj.label]
                poly_coords = []
                if len(obj.polygon) >= 3:
                    for pt in obj.polygon:
                        px = max(0.0, min(1.0, pt[0] / width))
                        py = max(0.0, min(1.0, pt[1] / height))
                        poly_coords.extend([f"{px:.6f}", f"{py:.6f}"])
                    if poly_coords:
                        lines.append(f"{cls_idx} " + " ".join(poly_coords))

            with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))

        return dst_img_path

    except Exception as e:
        logging.error(f"Error processing seg frame {frame_info.get('frame_number')}: {e}")
        logging.error(traceback.format_exc())
        return None


@ExporterRegistry.register
class YoloSegmentationExporter(BaseExporter):
    key = "yolo_segmentation"
    display_name = "YOLOv8 / YOLOv11 Segmentation (Polygons)"
    annotation_types = ["segmentation", "detection"]

    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        dataset_uuid = options.get("dataset_uuid")
        eval_percent = options.get("eval_percent", 20.0)
        test_percent = options.get("test_percent", 10.0)
        augmentation_options = options.get("augmentation_options", {})

        class_map = {name: i for i, name in enumerate(class_list)}

        is_aug_enabled = A is not None and augmentation_options.get("enabled", False)
        multiplication_factor = int(augmentation_options.get("multiply_factor", 1)) if is_aug_enabled else 1

        shuffled_frames = list(frames_data)
        random.shuffle(shuffled_frames)
        total_count = len(shuffled_frames)
        val_count = int(total_count * eval_percent / 100.0)
        test_count = int(total_count * test_percent / 100.0)

        raw_val = shuffled_frames[:val_count]
        raw_test = shuffled_frames[val_count:val_count + test_count]
        raw_train = shuffled_frames[val_count + test_count:]

        val_data = [{"type": "original", **f} for f in raw_val]
        test_data = [{"type": "original", **f} for f in raw_test]

        train_data = []
        for frame_info in raw_train:
            train_data.append({"type": "original", **frame_info})
            if is_aug_enabled and multiplication_factor > 1:
                for i in range(multiplication_factor - 1):
                    aug_id = f"aug_{i}_{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"
                    train_data.append({"type": "augmented", "augmented_id": aug_id, **frame_info})

        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        dir_map = {
            'train': (os.path.join(export_dir, 'images', 'train'), os.path.join(export_dir, 'labels', 'train')),
            'val': (os.path.join(export_dir, 'images', 'val'), os.path.join(export_dir, 'labels', 'val')),
            'test': (os.path.join(export_dir, 'images', 'test'), os.path.join(export_dir, 'labels', 'test')),
        }
        for img_dir, lbl_dir in dir_map.values():
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

        all_tasks = []
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            img_dir, lbl_dir = dir_map[split_name]
            for frame_info in split_data:
                all_tasks.append((frame_info, img_dir, lbl_dir, class_map, augmentation_options))

        settings = settings_manager.load_settings()
        max_workers_setting = settings.get('max_workers', 8)
        if max_workers_setting == 'auto':
            safe_workers = min(4, max(1, cpu_count() // 2))
        else:
            safe_workers = min(int(max_workers_setting), max(1, cpu_count()))

        if dataset_uuid:
            database.update_dataset_status(
                dataset_uuid,
                status="PROCESSING",
                message=f"Processing {len(all_tasks)} segmentation images across {safe_workers} CPU cores..."
            )

        processed_count = 0
        with Pool(processes=safe_workers) as pool:
            for result in pool.imap_unordered(process_seg_frame_worker, all_tasks):
                if result:
                    processed_count += 1
                    if processed_count % 50 == 0 and dataset_uuid:
                        progress_msg = f"Processed {processed_count}/{len(all_tasks)} segmentation images..."
                        database.update_dataset_status(dataset_uuid, status="PROCESSING", message=progress_msg)

        if yaml:
            yaml_content = {
                'path': f"../datasets/{os.path.basename(export_dir)}",
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': len(class_list),
                'names': class_list
            }
            with open(os.path.join(export_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, sort_keys=False)
