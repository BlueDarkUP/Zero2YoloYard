from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BBox:
    x_center: float
    y_center: float
    width: float
    height: float

@dataclass
class Annotation:
    class_id: int
    bbox: BBox
    track_id: Optional[str] = None
    track_name: Optional[str] = None

@dataclass
class ImageAnnotation:
    image_path: str
    width: int
    height: int
    annotations: List[Annotation] = field(default_factory=list)