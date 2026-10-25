from typing import Type, Dict, List
from .base import BaseExporter

class ExporterRegistry:
    _exporters: Dict[str, Type[BaseExporter]] = {}

    @classmethod
    def register(cls, exporter_class: Type[BaseExporter]):
        cls._exporters[exporter_class.key] = exporter_class
        return exporter_class

    @classmethod
    def get(cls, format_key: str) -> BaseExporter:
        alias_map = {
            "yolo_v8_pose": "yolo_pose",
            "yolo_v11_pose": "yolo_pose",
            "yolo_pose": "yolo_pose",
            "yolo_v8_seg": "yolo_segmentation",
            "yolo_v11_seg": "yolo_segmentation",
            "yolo_segmentation": "yolo_segmentation",
            "yolo_v8_detect": "yolo_v8_detect",
            "yolo_v11_detect": "yolo_v8_detect",
            "yolo_detect": "yolo_v8_detect",
            "yolo_v8_cls": "yolo_cls",
            "yolo_v11_cls": "yolo_cls",
            "yolo_cls": "yolo_cls",
            "coco_pose": "coco_pose",
            "coco_detect": "coco_detection",
            "coco_detection": "coco_detection",
            "voc_detect": "pascal_voc_detection",
            "pascal_voc": "pascal_voc_detection",
        }
        resolved = alias_map.get(format_key.lower().strip() if format_key else "", format_key)
        if resolved not in cls._exporters:
            raise ValueError(f"Unknown export format: {format_key}")
        return cls._exporters[resolved]()

    @classmethod
    def list_for_type(cls, annotation_type: str) -> List[Dict[str, str]]:
        result = []
        for key, exporter_cls in cls._exporters.items():
            if annotation_type in exporter_cls.annotation_types:
                result.append({
                    "key": key,
                    "name": exporter_cls.display_name,
                    "type": annotation_type
                })
        return result

    @classmethod
    def list_all(cls) -> List[Dict[str, str]]:
        result = []
        for key, exporter_cls in cls._exporters.items():
            result.append({
                "key": key,
                "name": exporter_cls.display_name,
                "types": exporter_cls.annotation_types
            })
        return result

# import all exporters to trigger registration
from .detection import *
from .segmentation import *
from .classification import *
from .pose import *
