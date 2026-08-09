from abc import ABC, abstractmethod
from typing import List

class BaseExporter(ABC):
    key: str = ""
    display_name: str = ""
    annotation_types: List[str] = []

    @abstractmethod
    def export(self, export_dir: str, frames_data: list, class_list: list, **options) -> None:
        """
        :param export_dir: Directory where files should be written
        :param frames_data: List of dicts, e.g. [{"frame": 1, "image_path": "...", "annotations": AnnotationData}]
        :param class_list: List of class names, e.g. ["car", "person"]
        :param options: additional options like dataset split ratios, model version, etc.
        """
        pass

    def supports(self, annotation_type: str) -> bool:
        return annotation_type in self.annotation_types
