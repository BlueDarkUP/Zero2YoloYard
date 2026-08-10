"""Single-image, per-class GroundingDINO detection adapter for GKD."""

import os
import sys

import numpy as np
import torch
from PIL import Image


_OBJECT_DETECTOR_ROOT = os.path.dirname(os.path.abspath(__file__))
_GROUNDING_DINO_ROOT = os.path.join(_OBJECT_DETECTOR_ROOT, 'groundingdino')
_DEFAULT_CONFIG = os.path.join(
    _GROUNDING_DINO_ROOT, 'config/GroundingDINO_SwinT_OGC.py'
)
_DEFAULT_CHECKPOINT = os.path.join(
    _OBJECT_DETECTOR_ROOT, 'weights/groundingdino_swint_ogc.pth'
)


class GroundingDINODetector:
    """Detect named objects and return GKD-compatible ``[x1, y1, x2, y2]`` boxes."""

    def __init__(self, config_path=_DEFAULT_CONFIG, checkpoint_path=_DEFAULT_CHECKPOINT,
                 box_threshold=0.30, text_threshold=0.25):
        if _OBJECT_DETECTOR_ROOT not in sys.path:
            sys.path.insert(0, _OBJECT_DETECTOR_ROOT)

        from groundingdino.datasets import transforms as transforms
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f'GroundingDINO config not found: {config_path}')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'GroundingDINO checkpoint not found: {checkpoint_path}')

        self.transforms = transforms
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = SLConfig.fromfile(config_path)
        config.device = str(self.device)
        self.model = build_model(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_result = self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print(f'Loaded GroundingDINO checkpoint: {load_result}')
        self.model = self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.RandomResize([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def detect(self, image_path, object_name):
        """Return detections as dictionaries with GKD box, score, and requested name."""
        caption = object_name.lower().strip().rstrip('.') + '.'
        image_pil = Image.open(image_path).convert('RGB')
        image, _ = self.transform(image_pil, None)

        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0).to(self.device), captions=[caption])

        logits = outputs['pred_logits'].sigmoid()[0].cpu()
        boxes = outputs['pred_boxes'][0].cpu()
        scores, _ = logits.max(dim=1)
        keep = scores > self.box_threshold
        scores, boxes = scores[keep], boxes[keep]

        width, height = image_pil.size
        detections = []
        for score, box in zip(scores, boxes):
            cx, cy, box_w, box_h = box.numpy() * np.array([width, height, width, height])
            x1 = max(0.0, cx - box_w / 2)
            y1 = max(0.0, cy - box_h / 2)
            x2 = min(float(width - 1), cx + box_w / 2)
            y2 = min(float(height - 1), cy + box_h / 2)
            if x2 > x1 and y2 > y1:
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'object_name': object_name,
                })

        detections.sort(key=lambda detection: detection['score'], reverse=True)
        return detections