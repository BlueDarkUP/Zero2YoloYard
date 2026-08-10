"""Local LocateAnything multi-category detector adapter for GKD."""

import os
import re

from PIL import Image

from test_real_world.object_detector_lib.locateanything_worker import LocateAnythingWorker


_OBJECT_DETECTOR_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(
    _OBJECT_DETECTOR_ROOT, 'weights', 'LocateAnything-3B'
)
_TOKEN_PATTERN = re.compile(
    r'<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>',
    flags=re.DOTALL,
)


class LocateAnythingDetector:
    """Detect several named classes in one LocateAnything inference call."""

    def __init__(self, model_path=_DEFAULT_MODEL_PATH):
        if not os.path.isdir(model_path) or not os.path.isfile(
            os.path.join(model_path, 'config.json')
        ):
            raise FileNotFoundError(
                'LocateAnything weights are not available locally. Download or copy the '
                f'complete nvidia/LocateAnything-3B snapshot into: {model_path}'
            )

        self.worker = LocateAnythingWorker(model_path)

    def detect(self, image_path, object_names):
        """Return category-aligned GKD ``[x1, y1, x2, y2]`` detections.

        LocateAnything emits coordinates as integers normalized to 0--1000 and
        has no detection confidence field, so every parsed detection uses 1.0.
        """
        image = Image.open(image_path).convert('RGB')
        categories = [name.strip() for name in object_names if name.strip()]
        if not categories:
            return []

        # Keep LocateAnything's official task prompt and decoding defaults.
        # Greedy decoding with altered penalties does not emit its normal end
        # marker reliably and can repeat the final box until max_new_tokens.
        answer = self.worker.detect(image, categories)['answer']
        print(f'LocateAnything output: {answer}')
        name_by_casefold = {name.casefold(): name for name in categories}
        width, height = image.size
        detections = []
        object_name = None
        seen = set()
        for label, x1, y1, x2, y2 in _TOKEN_PATTERN.findall(answer):
            if label:
                object_name = name_by_casefold.get(label.strip().casefold())
                if object_name is None:
                    print(f'Ignoring LocateAnything reference with unexpected label: {label!r}')
                continue
            if object_name is None:
                continue
            x1, y1, x2, y2 = [int(value) for value in (x1, y1, x2, y2)]
            raw_box = (object_name, x1, y1, x2, y2)
            if raw_box in seen:
                continue
            seen.add(raw_box)
            x1 = max(0.0, min(float(width - 1), x1 / 1000.0 * width))
            y1 = max(0.0, min(float(height - 1), y1 / 1000.0 * height))
            x2 = max(0.0, min(float(width - 1), x2 / 1000.0 * width))
            y2 = max(0.0, min(float(height - 1), y2 / 1000.0 * height))
            if x2 > x1 and y2 > y1:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': 1.0,
                    'object_name': object_name,
                })
        return detections
