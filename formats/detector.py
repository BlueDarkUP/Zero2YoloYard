# formats/detector.py
import os
import json
import xml.etree.ElementTree as ET

def detect_format(dataset_path: str) -> str:
    if not os.path.isdir(dataset_path):
        return "Unknown"

    files_in_root = os.listdir(dataset_path)
    
    # Check if directory contains image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    has_images = any(f.lower().endswith(tuple(image_extensions)) for f in files_in_root)
    if has_images:
        return "New Dataset"

    # --- 检查 TFRecord 格式 ---
    has_tfrecord = any(f.endswith('.tfrecord') for f in files_in_root)
    has_label_map = any(f == 'label_map.pbtxt' for f in files_in_root)
    if has_tfrecord and has_label_map:
        return "TFRecord"

    # --- 检查 COCO 格式 ---
    annotations_dir_coco = os.path.join(dataset_path, 'annotations')
    if os.path.isdir(annotations_dir_coco):
        json_files = [f for f in os.listdir(annotations_dir_coco) if f.endswith('.json')]
        if json_files:
            try:
                with open(os.path.join(annotations_dir_coco, json_files[0]), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'images' in data and 'annotations' in data and 'categories' in data:
                    return "COCO"
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                pass

    # --- 检查 Pascal VOC 格式 ---
    annotations_dir_voc = os.path.join(dataset_path, 'Annotations')
    if os.path.isdir(annotations_dir_voc):
        xml_files = [f for f in os.listdir(annotations_dir_voc) if f.endswith('.xml')]
        if xml_files:
            try:
                tree = ET.parse(os.path.join(annotations_dir_voc, xml_files[0]))
                if tree.getroot().tag == 'annotation':
                    return "Pascal VOC"
            except ET.ParseError:
                pass

    # --- 检查 YOLO 格式 ---
    dirs_to_check = [dataset_path] + [os.path.join(dataset_path, split) for split in ['train', 'valid', 'test']]
    for check_dir in dirs_to_check:
        images_dir_yolo = os.path.join(check_dir, 'images')
        labels_dir_yolo = os.path.join(check_dir, 'labels')
        if os.path.isdir(images_dir_yolo) and os.path.isdir(labels_dir_yolo):
            if any(f.endswith('.txt') for f in os.listdir(labels_dir_yolo)):
                return "YOLO"

    return "Unknown"