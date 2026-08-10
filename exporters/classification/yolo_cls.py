import os
import re
import shutil
import random
from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData
import file_storage


def _sanitize_class_name(name: str) -> str:
    """清洗类别名称，防止路径穿越攻击（如 '../../etc/passwd'）。"""
    # 移除路径分隔符及连续点（防止目录穿越）
    name = name.replace('/', '_').replace('\\', '_')
    name = re.sub(r'\.{2,}', '_', name)
    # 只保留字母、数字、空格、下划线、连字符、加号
    name = re.sub(r'[^\w\s\-+]', '_', name)
    return name.strip() or '_unknown'

@ExporterRegistry.register
class YoloClassificationExporter(BaseExporter):
    key = "yolo_cls"
    display_name = "YOLO Classification (Folder Structure)"
    annotation_types = ["classification"]

    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        eval_percent = options.get("eval_percent", 20.0)
        test_percent = options.get("test_percent", 10.0)

        # Multi-label export strategy: 'first' (default), 'composite', 'skip', 'duplicate'
        multilabel_strategy = options.get("multilabel_strategy")
        if not multilabel_strategy:
            export_opts = options.get("export_options") or {}
            multilabel_strategy = export_opts.get("multilabel_strategy", "first")
        if multilabel_strategy not in ["first", "composite", "skip", "duplicate"]:
            multilabel_strategy = "first"

        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        shuffled_data = list(frames_data)
        random.shuffle(shuffled_data)

        total_count = len(shuffled_data)
        val_count = int(total_count * eval_percent / 100.0)
        test_count = int(total_count * test_percent / 100.0)

        val_data = shuffled_data[:val_count]
        test_data = shuffled_data[val_count:val_count + test_count]
        train_data = shuffled_data[val_count + test_count:]

        splits = [('train', train_data), ('val', val_data), ('test', test_data)]

        export_opts = options.get("export_options") or {}
        include_unlabeled = bool(options.get("include_unlabeled", export_opts.get("include_unlabeled", False)))

        for split_name, split_frames in splits:
            split_dir = os.path.join(export_dir, split_name)
            for class_name in class_list:
                os.makedirs(os.path.join(split_dir, _sanitize_class_name(class_name)), exist_ok=True)
            if include_unlabeled:
                os.makedirs(os.path.join(split_dir, "_unlabeled"), exist_ok=True)

            for frame_info in split_frames:
                base_name = f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}.jpg"
                src_img_path = file_storage.get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])

                if not os.path.exists(src_img_path):
                    continue

                annotations: AnnotationData = frame_info["annotations"]
                tags = annotations.classifications if annotations.classifications else []

                if not tags:
                    if include_unlabeled:
                        shutil.copy(src_img_path, os.path.join(split_dir, "_unlabeled", base_name))
                    else:
                        continue
                elif len(tags) == 1:
                    target_cls = _sanitize_class_name(tags[0])
                    target_dir = os.path.join(split_dir, target_cls)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy(src_img_path, os.path.join(target_dir, base_name))
                else:
                    # Multi-label (>1 tags)
                    if multilabel_strategy == "first":
                        target_cls = _sanitize_class_name(tags[0])
                        target_dir = os.path.join(split_dir, target_cls)
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.copy(src_img_path, os.path.join(target_dir, base_name))
                    elif multilabel_strategy == "composite":
                        sorted_tags = sorted(tags)
                        composite_cls = "+".join(_sanitize_class_name(t) for t in sorted_tags)
                        target_dir = os.path.join(split_dir, composite_cls)
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.copy(src_img_path, os.path.join(target_dir, base_name))
                    elif multilabel_strategy == "skip":
                        # Skip image with multiple labels entirely
                        continue
                    elif multilabel_strategy == "duplicate":
                        # Legacy behavior: copy into each label directory
                        for target_cls in tags:
                            target_dir = os.path.join(split_dir, _sanitize_class_name(target_cls))
                            os.makedirs(target_dir, exist_ok=True)
                            shutil.copy(src_img_path, os.path.join(target_dir, base_name))

        # Clean up empty directories in splits
        for split_name, _ in splits:
            split_dir = os.path.join(export_dir, split_name)
            if os.path.exists(split_dir):
                for sub_dir in os.listdir(split_dir):
                    sub_path = os.path.join(split_dir, sub_dir)
                    if os.path.isdir(sub_path) and not os.listdir(sub_path):
                        os.rmdir(sub_path)
                if not os.listdir(split_dir):
                    os.rmdir(split_dir)
