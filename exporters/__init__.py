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
        if format_key not in cls._exporters:
            raise ValueError(f"Unknown export format: {format_key}")
        return cls._exporters[format_key]()

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
