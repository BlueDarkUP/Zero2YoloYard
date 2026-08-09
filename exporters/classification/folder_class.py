import os
import shutil
from ..base import BaseExporter
from .. import ExporterRegistry
from annotation_model import AnnotationData
import file_storage

@ExporterRegistry.register
class FolderClassificationExporter(BaseExporter):
    key = "folder_classification"
    display_name = "Image Classification (Folder Structure)"
    annotation_types = ["classification"]

    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        os.makedirs(export_dir, exist_ok=True)
        for class_name in class_list:
            os.makedirs(os.path.join(export_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(export_dir, "_unlabeled"), exist_ok=True)

        for frame_info in frames_data:
            base_name = f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}.jpg"
            src_img_path = file_storage.get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])

            if not os.path.exists(src_img_path):
                continue

            annotations: AnnotationData = frame_info["annotations"]
            if annotations.classifications:
                for target_cls in annotations.classifications:
                    target_dir = os.path.join(export_dir, target_cls)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy(src_img_path, os.path.join(target_dir, base_name))
            else:
                shutil.copy(src_img_path, os.path.join(export_dir, "_unlabeled", base_name))
