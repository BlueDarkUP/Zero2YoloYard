# ai_models.py
import numpy as np
import cv2
import traceback

try:
    import torch
    from ultralytics import YOLO
    from segment_anything import sam_model_registry, SamPredictor

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class ModelLoadingError(Exception):
    pass


class ModelManager:
    _instances = {}

    def __new__(cls, model_path=None, model_name=None, model_type=None, *args, **kwargs):
        key = model_path if model_path else model_name
        if not key:
            raise ValueError("ModelManager需要 model_path 或 model_name。")

        if key not in cls._instances:
            print(f"正在加载模型: {key}...")
            instance = None
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"

                if not model_type and model_path:
                    if model_path.endswith('.pt'):
                        model_type = 'yolo'
                    elif model_path.endswith('.pth'):
                        model_type = 'sam'

                if model_type == 'yolo':
                    yolo_target = model_path if model_path else f"{model_name}.pt"
                    instance = YOLO(yolo_target)
                elif model_type == 'sam':
                    sam_target_path = model_path
                    if not sam_target_path:
                        sam_model_arch = model_name.split('_', 1)[1] if '_' in model_name else 'vit_h'
                        sam_target_path = f"sam_{sam_model_arch}.pth"

                    sam_model_arch = 'vit_h'
                    if 'vit_b' in sam_target_path: sam_model_arch = 'vit_b'
                    if 'vit_l' in sam_target_path: sam_model_arch = 'vit_l'

                    sam = sam_model_registry[sam_model_arch](checkpoint=sam_target_path)
                    sam.to(device=device)
                    instance = SamPredictor(sam)
                else:
                    raise ModelLoadingError(f"不支持的模型类型 '{model_type}' 或无法从路径推断。")

                if instance is None:
                    raise ModelLoadingError(f"模型 '{key}' 加载失败，但未捕获到明确异常。")

            except FileNotFoundError as e:
                raise ModelLoadingError(f"找不到模型文件 '{getattr(e, 'filename', key)}'。请确保文件路径正确。") from e
            except Exception as e:
                raise ModelLoadingError(f"加载模型 '{key}' 时发生未知错误: {e}") from e

            cls._instances[key] = instance
            print(f"模型 '{key}' 加载成功。")

        return cls._instances.get(key)


def run_yolo_diagnosis(yolo_model, image_path, confidence_threshold=0.25):
    if not AI_AVAILABLE or yolo_model is None:
        return []
    results = yolo_model(image_path, conf=confidence_threshold, verbose=False)
    predictions = []
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            predictions.append({'box': [x1, y1, x2, y2], 'confidence': conf, 'class_id': cls})
    return predictions


def run_sam_prediction(sam_predictor, image, point_coords):
    if not AI_AVAILABLE or sam_predictor is None or image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)

    point_labels = np.array([1])

    masks, scores, logits = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_mask = masks[np.argmax(scores)]

    y_indices, x_indices = np.where(best_mask > 0)
    if len(x_indices) == 0:
        return None

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    return [x_min, y_min, x_max, y_max]