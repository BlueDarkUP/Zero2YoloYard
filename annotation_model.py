import json
from typing import List, Dict, Any, Optional

COCO_POSE_17_SCHEMA = {
    "name": "COCO-Pose-17",
    "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"],
    "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                 [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                 [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
    "flip_map": [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17]]
}

class AnnotationObject:
    def __init__(self, id: str, type: str, label: str, bbox: Optional[List[float]] = None, 
                 points: Optional[List[List[float]]] = None, polygon: Optional[List[List[float]]] = None,
                 keypoints: Optional[List[Dict[str, Any]]] = None):
        self.id = id
        self.type = type
        self.label = label
        self.bbox = bbox
        self.points = polygon if polygon is not None else points
        self.keypoints = keypoints

    @property
    def polygon(self) -> Optional[List[List[float]]]:
        return self.points

    @polygon.setter
    def polygon(self, value: Optional[List[List[float]]]):
        self.points = value

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "type": self.type,
            "label": self.label
        }
        if self.bbox is not None:
            d["bbox"] = self.bbox
        if self.points is not None:
            d["points"] = self.points
            d["polygon"] = self.points
        if self.keypoints is not None:
            d["keypoints"] = self.keypoints
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AnnotationObject':
        points_val = data.get("points")
        if points_val is None:
            points_val = data.get("polygon")
        return AnnotationObject(
            id=data.get("id", ""),
            type=data.get("type", "bbox"),
            label=data.get("label", ""),
            bbox=data.get("bbox"),
            points=points_val,
            polygon=data.get("polygon"),
            keypoints=data.get("keypoints")
        )

class AnnotationData:
    def __init__(self, version: int = 1, objects: Optional[List[AnnotationObject]] = None, classifications: Optional[List[str]] = None):
        self.version = version
        self.objects = objects if objects is not None else []
        self.classifications = classifications if classifications is not None else []

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AnnotationData':
        if not data or not isinstance(data, dict):
            return AnnotationData.empty()
        objects = [AnnotationObject.from_dict(obj) for obj in data.get("objects", [])]
        return AnnotationData(
            version=data.get("version", 1),
            objects=objects,
            classifications=data.get("classifications", [])
        )

    @staticmethod
    def from_json(json_str: str) -> 'AnnotationData':
        if not json_str or json_str.strip() == "":
            return AnnotationData.empty()
        
        try:
            data = json.loads(json_str)
            return AnnotationData.from_dict(data)
        except (json.JSONDecodeError, TypeError):
            return AnnotationData.empty()

    @staticmethod
    def empty() -> 'AnnotationData':
        return AnnotationData(version=1, objects=[], classifications=[])

    def to_json(self) -> str:
        return json.dumps({
            "version": self.version,
            "objects": [obj.to_dict() for obj in self.objects],
            "classifications": self.classifications
        })

    def get_bboxes(self) -> List[AnnotationObject]:
        return [obj for obj in self.objects if obj.type == "bbox"]

    def get_polygons(self) -> List[AnnotationObject]:
        return [obj for obj in self.objects if obj.type == "polygon"]

    def get_keypoints(self) -> List[AnnotationObject]:
        return [obj for obj in self.objects if obj.type == "keypoint"]

    def get_unique_labels(self) -> set[str]:
        labels = set()
        for obj in self.objects:
            labels.add(obj.label)
        for cls in self.classifications:
            labels.add(cls)
        return labels

def bbox_to_polygon(bbox: List[float]) -> List[List[float]]:
    """Convert [x1, y1, x2, y2] to [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]"""
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
