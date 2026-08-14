import os
import shutil
import cv2
import yaml
import zipfile
import traceback
import logging
import random
from multiprocessing import Pool, cpu_count
import numpy as np

try:
    import albumentations as A
except ImportError:
    A = None

from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData
import file_storage
import database
import config
import settings_manager

def tight_fit_bbox(orig_yolos, aug_yolos, orig_ids=None, aug_ids=None):
    """
    针对 Rotate (Fine Rotation) / ShiftScaleRotate / Affine 等几何旋转剪切增强后，
    轴对齐外接正矩形导致包围盒变大几圈的现象，进行保真面积与比例等比例几何收敛修正。
    """
    if not orig_yolos or not aug_yolos:
        return aug_yolos

    corrected = []
    
    if orig_ids is not None and aug_ids is not None:
        orig_map = {oid: bbox for oid, bbox in zip(orig_ids, orig_yolos)}
        for aid, aug in zip(aug_ids, aug_yolos):
            if aid in orig_map:
                orig = orig_map[aid]
                orig_w, orig_h = orig[2], orig[3]
                aug_cx, aug_cy, aug_w, aug_h = aug[0], aug[1], aug[2], aug[3]
                orig_area = orig_w * orig_h
                aug_area = aug_w * aug_h

                if orig_area > 0 and aug_area > orig_area * 1.02:
                    scale_factor = float(np.sqrt(orig_area / aug_area))
                    new_w = max(0.002, aug_w * scale_factor)
                    new_h = max(0.002, aug_h * scale_factor)
                    corrected.append([aug_cx, aug_cy, new_w, new_h])
                else:
                    corrected.append([aug_cx, aug_cy, aug_w, aug_h])
            else:
                corrected.append(aug)
    else:
        if len(orig_yolos) != len(aug_yolos):
            return aug_yolos
        for orig, aug in zip(orig_yolos, aug_yolos):
            orig_w, orig_h = orig[2], orig[3]
            aug_cx, aug_cy, aug_w, aug_h = aug[0], aug[1], aug[2], aug[3]

            orig_area = orig_w * orig_h
            aug_area = aug_w * aug_h

            if orig_area > 0 and aug_area > orig_area * 1.02:
                scale_factor = float(np.sqrt(orig_area / aug_area))
                new_w = max(0.002, aug_w * scale_factor)
                new_h = max(0.002, aug_h * scale_factor)
                corrected.append([aug_cx, aug_cy, new_w, new_h])
            else:
                corrected.append([aug_cx, aug_cy, aug_w, aug_h])

    return corrected

def build_augmentation_pipeline(options):
    if A is None: return None
    transforms = []
    if options.get('hflip', {}).get('enabled'):
        transforms.append(A.HorizontalFlip(p=options['hflip']['p']))
    if options.get('vflip', {}).get('enabled'):
        transforms.append(A.VerticalFlip(p=options['vflip']['p']))
    if options.get('rotate90', {}).get('enabled'):
        transforms.append(A.RandomRotate90(p=options['rotate90']['p']))
    if options.get('rotate', {}).get('enabled'):
        transforms.append(
            A.Rotate(limit=options['rotate']['limit'], p=options['rotate']['p'], border_mode=cv2.BORDER_CONSTANT,
                     value=0))
    if options.get('ssr', {}).get('enabled'):
        transforms.append(A.ShiftScaleRotate(shift_limit=options['ssr']['shift'], scale_limit=options['ssr']['scale'],
                                             rotate_limit=options['ssr']['rotate'], p=options['ssr']['p'],
                                             border_mode=cv2.BORDER_CONSTANT, value=0))
    if options.get('affine', {}).get('enabled'):
        limit = options['affine']['shear']
        transforms.append(
            A.Affine(shear={'x': (-limit, limit), 'y': (-limit, limit)}, p=options['affine']['p'], cval=0))
    if options.get('crop', {}).get('enabled'):
        transforms.append(A.CropAndPad(percent=(-0.15, 0.0), pad_mode=cv2.BORDER_CONSTANT, p=options['crop']['p']))

    if options.get('grayscale', {}).get('enabled'):
        transforms.append(A.ToGray(p=options['grayscale']['p']))
    if options.get('hsv', {}).get('enabled'):
        transforms.append(A.HueSaturationValue(hue_shift_limit=options['hsv']['h'], sat_shift_limit=options['hsv']['s'],
                                               val_shift_limit=options['hsv']['v'], p=options['hsv']['p']))
    if options.get('bc', {}).get('enabled'):
        transforms.append(
            A.RandomBrightnessContrast(brightness_limit=options['bc']['b'], contrast_limit=options['bc']['c'],
                                       p=options['bc']['p']))

    if options.get('blur', {}).get('enabled'):
        transforms.append(A.GaussianBlur(blur_limit=(3, options['blur']['limit']), p=options['blur']['p']))
    if options.get('noise', {}).get('enabled'):
        transforms.append(A.GaussNoise(var_limit=(10.0, options['noise']['limit']), p=options['noise']['p']))

    if not transforms: return None
    return A.Compose(transforms,
                     bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels', 'track_ids'], min_visibility=0.1))

def build_augmentation_pipeline_for_keypoints(options):
    if A is None: return None
    transforms = []
    if options.get('hflip', {}).get('enabled'):
        transforms.append(A.HorizontalFlip(p=options['hflip']['p']))
    if options.get('vflip', {}).get('enabled'):
        transforms.append(A.VerticalFlip(p=options['vflip']['p']))
    if options.get('rotate90', {}).get('enabled'):
        transforms.append(A.RandomRotate90(p=options['rotate90']['p']))
    if options.get('rotate', {}).get('enabled'):
        transforms.append(
            A.Rotate(limit=options['rotate']['limit'], p=options['rotate']['p'], border_mode=cv2.BORDER_CONSTANT,
                     value=0))
    if options.get('ssr', {}).get('enabled'):
        transforms.append(A.ShiftScaleRotate(shift_limit=options['ssr']['shift'], scale_limit=options['ssr']['scale'],
                                             rotate_limit=options['ssr']['rotate'], p=options['ssr']['p'],
                                             border_mode=cv2.BORDER_CONSTANT, value=0))
    if options.get('affine', {}).get('enabled'):
        limit = options['affine']['shear']
        transforms.append(
            A.Affine(shear={'x': (-limit, limit), 'y': (-limit, limit)}, p=options['affine']['p'], cval=0))

    if options.get('grayscale', {}).get('enabled'):
        transforms.append(A.ToGray(p=options['grayscale']['p']))
    if options.get('hsv', {}).get('enabled'):
        transforms.append(A.HueSaturationValue(hue_shift_limit=options['hsv']['h'], sat_shift_limit=options['hsv']['s'],
                                               val_shift_limit=options['hsv']['v'], p=options['hsv']['p']))
    if options.get('bc', {}).get('enabled'):
        transforms.append(
            A.RandomBrightnessContrast(brightness_limit=options['bc']['b'], contrast_limit=options['bc']['c'],
                                       p=options['bc']['p']))

    if options.get('blur', {}).get('enabled'):
        transforms.append(A.GaussianBlur(blur_limit=(3, options['blur']['limit']), p=options['blur']['p']))
    if options.get('noise', {}).get('enabled'):
        transforms.append(A.GaussNoise(var_limit=(10.0, options['noise']['limit']), p=options['noise']['p']))

    if not transforms: return None
    return A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'], remove_invisible=False))

class BboxSafeCoarseDropout: # Placeholder to keep compatibility
    pass

def process_frame_worker(args):
    frame_info, target_img_dir, target_lbl_dir, class_map, augmentation_options = args

    augment_pipeline = None
    is_augmented = frame_info.get("type") == "augmented"
    if is_augmented and augmentation_options and augmentation_options.get("enabled", False):
        augment_pipeline = build_augmentation_pipeline(augmentation_options)

    try:
        if is_augmented:
            base_filename = frame_info["augmented_id"]
        else:
            base_filename = f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"

        src_img_path = file_storage.get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])
        if not os.path.exists(src_img_path):
            logging.warning(f"Source file not found, skipping: {src_img_path}")
            return None

        image = cv2.imread(src_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process annotations
        annotations: AnnotationData = frame_info["annotations"]
        bboxes = annotations.get_bboxes()
        img_w = frame_info['width']
        img_h = frame_info['height']
        
        yolo_bboxes = []
        class_indices = []
        track_ids = []
        for i, obj in enumerate(bboxes):
            if obj.label in class_map:
                x1, y1, x2, y2 = obj.bbox
                # Convert to YOLO format (cx, cy, w, h) normalized
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                # Clamp between 0 and 1
                cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
                w, h = max(0, min(1, w)), max(0, min(1, h))
                if w > 0 and h > 0:
                    yolo_bboxes.append([cx, cy, w, h])
                    class_indices.append(class_map[obj.label])
                    track_ids.append(i)

        if not yolo_bboxes and not is_augmented:
            # We can still export images without labels for background, but typical YOLO requires at least one bbox or an empty txt
            pass # Keep it for empty label file

        if is_augmented and augment_pipeline and yolo_bboxes:
            transformed = augment_pipeline(image=image, bboxes=yolo_bboxes, class_labels=class_indices, track_ids=track_ids)
            image_aug_rgb = transformed['image']
            bboxes_aug_yolo_tuples = transformed['bboxes']
            labels_aug_indices = transformed['class_labels']
            aug_track_ids = transformed['track_ids']

            corrected_tuples = tight_fit_bbox(yolo_bboxes, bboxes_aug_yolo_tuples, orig_ids=track_ids, aug_ids=aug_track_ids)
            bboxes_aug_yolo = [(labels_aug_indices[i], *box) for i, box in enumerate(corrected_tuples)]
        else:
            image_aug_rgb = image
            bboxes_aug_yolo = [(class_indices[i], *box) for i, box in enumerate(yolo_bboxes)]

        final_image_bgr = cv2.cvtColor(image_aug_rgb, cv2.COLOR_RGB2BGR)
        output_image_path = os.path.join(target_img_dir, base_filename + '.jpg')
        cv2.imwrite(output_image_path, final_image_bgr)

        yolo_content_lines = [f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for class_id, x, y, w, h in bboxes_aug_yolo]
        output_label_path = os.path.join(target_lbl_dir, base_filename + '.txt')
        with open(output_label_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_content_lines))

        return output_image_path

    except Exception as e:
        logging.error(f"Error processing frame {frame_info.get('augmented_id') or frame_info.get('frame_number')}: {e}")
        logging.error(traceback.format_exc())
        return None

@ExporterRegistry.register
class YoloDetectionExporter(BaseExporter):
    key = "yolo_v8_detect"
    display_name = "YOLO Detection (v5/v7/v8/v9/v11)"
    annotation_types = ["detection"]

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

        if os.path.exists(export_dir): shutil.rmtree(export_dir)
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
            database.update_dataset_status(dataset_uuid, status="PROCESSING",
                                           message=f"Processing {len(all_tasks)} images across {safe_workers} CPU cores...")
        
        processed_count = 0
        with Pool(processes=safe_workers) as pool:
            for result in pool.imap_unordered(process_frame_worker, all_tasks):
                if result:
                    processed_count += 1
                    if processed_count % 50 == 0 and dataset_uuid:
                        progress_msg = f"Processed {processed_count}/{len(all_tasks)} images..."
                        database.update_dataset_status(dataset_uuid, status="PROCESSING", message=progress_msg)

        if yaml:
            yaml_content = {'path': f"../datasets/{os.path.basename(export_dir)}", 'train': 'images/train', 'val': 'images/val',
                            'test': 'images/test', 'nc': len(class_list), 'names': class_list}
            with open(os.path.join(export_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, sort_keys=False)
        
