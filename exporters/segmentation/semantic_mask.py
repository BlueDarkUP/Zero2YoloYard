import os
import cv2
import numpy as np
import shutil
import logging
import traceback
import random
from concurrent.futures import ThreadPoolExecutor

try:
    import albumentations as A
except ImportError:
    A = None

from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData
import file_storage
import settings_manager


def build_semantic_aug_pipeline(options):
    if A is None:
        return None
    transforms = []
    if options.get('hflip', {}).get('enabled'):
        transforms.append(A.HorizontalFlip(p=options['hflip']['p']))
    if options.get('vflip', {}).get('enabled'):
        transforms.append(A.VerticalFlip(p=options['vflip']['p']))
    if options.get('rotate90', {}).get('enabled'):
        transforms.append(A.RandomRotate90(p=options['rotate90']['p']))
    if options.get('rotate', {}).get('enabled'):
        transforms.append(
            A.Rotate(limit=options['rotate']['limit'], p=options['rotate']['p'], border_mode=cv2.BORDER_CONSTANT, value=0)
        )
    if options.get('ssr', {}).get('enabled'):
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=options['ssr']['shift'],
                scale_limit=options['ssr']['scale'],
                rotate_limit=options['ssr']['rotate'],
                p=options['ssr']['p'],
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            )
        )
    if options.get('affine', {}).get('enabled'):
        limit = options['affine']['shear']
        transforms.append(
            A.Affine(shear={'x': (-limit, limit), 'y': (-limit, limit)}, p=options['affine']['p'], cval=0)
        )
    if options.get('crop', {}).get('enabled'):
        transforms.append(A.CropAndPad(percent=(-0.15, 0.0), pad_mode=cv2.BORDER_CONSTANT, p=options['crop']['p']))

    if options.get('grayscale', {}).get('enabled'):
        transforms.append(A.ToGray(p=options['grayscale']['p']))
    if options.get('hsv', {}).get('enabled'):
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=options['hsv']['h'],
                sat_shift_limit=options['hsv']['s'],
                val_shift_limit=options['hsv']['v'],
                p=options['hsv']['p']
            )
        )
    if options.get('bc', {}).get('enabled'):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=options['bc']['b'],
                contrast_limit=options['bc']['c'],
                p=options['bc']['p']
            )
        )

    if options.get('blur', {}).get('enabled'):
        transforms.append(A.GaussianBlur(blur_limit=(3, options['blur']['limit']), p=options['blur']['p']))
    if options.get('noise', {}).get('enabled'):
        transforms.append(A.GaussNoise(var_limit=(10.0, options['noise']['limit']), p=options['noise']['p']))

    if not transforms:
        return None
    return A.Compose(transforms)


def process_semantic_frame_worker(args):
    frame_info, target_img_dir, target_mask_dir, class_map, augmentation_options = args
    try:
        is_augmented = frame_info.get("type") == "augmented"
        base_filename = (
            frame_info["augmented_id"]
            if is_augmented
            else f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"
        )

        dst_img_path = os.path.join(target_img_dir, base_filename + '.jpg')
        dst_mask_path = os.path.join(target_mask_dir, base_filename + '.png')

        src_img_path = file_storage.get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])
        if not os.path.exists(src_img_path):
            logging.warning(f"Source frame not found: {src_img_path}")
            return None

        image = cv2.imread(src_img_path)
        if image is None:
            return None
        
        height, width = image.shape[:2]
        # Background is 0, class IDs are 1..N
        mask = np.zeros((height, width), dtype=np.uint8)

        annotations: AnnotationData = frame_info["annotations"]
        polygons_data = annotations.get_polygons()
        bboxes_data = annotations.get_bboxes()

        if polygons_data:
            for obj in polygons_data:
                if obj.label in class_map and len(obj.polygon) >= 3:
                    cls_val = class_map[obj.label] + 1  # 0 is Background
                    pts = np.array(obj.polygon, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], color=int(cls_val))
        elif bboxes_data:
            # Fallback to Bbox rasterization if no polygon annotations present
            for obj in bboxes_data:
                if obj.label in class_map:
                    cls_val = class_map[obj.label] + 1
                    x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                    cv2.rectangle(mask, (x1, y1), (x2, y2), color=int(cls_val), thickness=-1)

        if is_augmented and A is not None and augmentation_options.get("enabled", False):
            pipeline = build_semantic_aug_pipeline(augmentation_options)
            if pipeline:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed = pipeline(image=image_rgb, mask=mask)
                image_aug = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                mask_aug = transformed['mask']
            else:
                image_aug = image
                mask_aug = mask
        else:
            image_aug = image
            mask_aug = mask

        cv2.imwrite(dst_img_path, image_aug)
        cv2.imwrite(dst_mask_path, mask_aug)
        return dst_img_path

    except Exception as e:
        logging.error(f"Error processing semantic frame {frame_info.get('frame_number')}: {e}")
        logging.error(traceback.format_exc())
        return None


@ExporterRegistry.register
class SemanticMaskExporter(BaseExporter):
    key = "semantic_mask"
    display_name = "Semantic Segmentation Mask (PNG 8-bit Indexed)"
    annotation_types = ["segmentation", "detection"]

    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        eval_percent = options.get("eval_percent", 20.0)
        test_percent = options.get("test_percent", 10.0)
        augmentation_options = options.get("augmentation_options", {})

        class_map = {name: i for i, name in enumerate(class_list)}

        is_aug_enabled = A is not None and augmentation_options.get("enabled", False)
        multiplication_factor = int(augmentation_options.get("multiply_factor", 1)) if is_aug_enabled else 1

        final_frames_to_process = []
        if is_aug_enabled and multiplication_factor > 1:
            for frame_info in frames_data:
                final_frames_to_process.append({"type": "original", **frame_info})
                for i in range(multiplication_factor - 1):
                    aug_id = f"aug_{i}_{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"
                    final_frames_to_process.append({"type": "augmented", "augmented_id": aug_id, **frame_info})
        else:
            final_frames_to_process = [{"type": "original", **frame_info} for frame_info in frames_data]

        random.shuffle(final_frames_to_process)
        total_count = len(final_frames_to_process)
        val_count = int(total_count * eval_percent / 100.0)
        test_count = int(total_count * test_percent / 100.0)

        val_data = final_frames_to_process[:val_count]
        test_data = final_frames_to_process[val_count:val_count + test_count]
        train_data = final_frames_to_process[val_count + test_count:]

        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        dir_map = {
            'train': (os.path.join(export_dir, 'images', 'train'), os.path.join(export_dir, 'masks', 'train')),
            'val': (os.path.join(export_dir, 'images', 'val'), os.path.join(export_dir, 'masks', 'val')),
            'test': (os.path.join(export_dir, 'images', 'test'), os.path.join(export_dir, 'masks', 'test')),
        }
        for img_dir, mask_dir in dir_map.values():
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

        all_tasks = []
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            img_dir, mask_dir = dir_map[split_name]
            for frame_info in split_data:
                all_tasks.append((frame_info, img_dir, mask_dir, class_map, augmentation_options))

        settings = settings_manager.load_settings()
        max_workers = int(settings.get('export_cpu_workers', os.cpu_count() or 4))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(process_semantic_frame_worker, all_tasks))

        # Write classes.txt
        classes_txt_path = os.path.join(export_dir, 'classes.txt')
        with open(classes_txt_path, 'w', encoding='utf-8') as f:
            f.write("0: background\n")
            for idx, name in enumerate(class_list):
                f.write(f"{idx + 1}: {name}\n")
