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
        import shutil
        import logging

        # 候选路径：优先检查外部用户数据目录（exe 同级目录）
        try:
            import config as app_cfg
            ext_path = os.path.join(app_cfg.BASE_DIR, 'gkdt_engine', 'test_real_world', 'object_detector_lib', 'weights', 'LocateAnything-3B')
            f1_ext = os.path.join(ext_path, 'model-00001-of-00002.safetensors')
            f2_ext = os.path.join(ext_path, 'model-00002-of-00002.safetensors')
            f_single_ext = os.path.join(ext_path, 'model.safetensors')
            if (os.path.isfile(f1_ext) and os.path.isfile(f2_ext)) or os.path.isfile(f_single_ext):
                model_path = ext_path
        except Exception:
            pass

        f1 = os.path.join(model_path, 'model-00001-of-00002.safetensors')
        f2 = os.path.join(model_path, 'model-00002-of-00002.safetensors')
        f_single = os.path.join(model_path, 'model.safetensors')
        has_weights = (os.path.isfile(f1) and os.path.isfile(f2)) or os.path.isfile(f_single)

        # 若外部目录拥有权重，自动从内置目录同步代码和配置资产（确保 transformers AutoModel.from_pretrained 可执行）
        if has_weights and model_path != _DEFAULT_MODEL_PATH and os.path.exists(_DEFAULT_MODEL_PATH):
            for item in os.listdir(_DEFAULT_MODEL_PATH):
                if item.endswith('.safetensors') or item.endswith('.tmp'):
                    continue
                src = os.path.join(_DEFAULT_MODEL_PATH, item)
                dst = os.path.join(model_path, item)
                if not os.path.exists(dst):
                    try:
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                    except Exception as e:
                        logging.warning(f"Failed to sync LocateAnything asset {item} to {model_path}: {e}")

        if not os.path.isdir(model_path) or not os.path.isfile(os.path.join(model_path, 'config.json')) or not has_weights:
            raise FileNotFoundError(
                'LocateAnything weights are not available locally. Please download weights from Settings -> Local Model Management '
                f'or copy the complete nvidia/LocateAnything-3B safetensors weights into: {model_path}'
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
