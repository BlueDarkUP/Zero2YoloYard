import logging
import os
import random
import threading
import time
import cv2
import gc
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.amp import autocast
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.ops import nms, box_iou
from torchvision.transforms.functional import to_tensor

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    logging.warning("scikit-learn not found. Sub-prototype clustering will be disabled. Run 'pip install scikit-learn'")
    KMeans = None
    silhouette_score = None

import config
import database
import file_storage
import settings_manager
from bbox_writer import convert_text_to_rects_and_labels

try:
    import ultralytics_sam_tasks as sam_tasks
except ImportError:
    logging.warning("ultralytics_sam_tasks.py not found or failed to import. All SAM features will be disabled.")
    sam_tasks = None


# ==============================================================================
# 内存与缓存管理优化 (LRU Cache Implementation)
# ==============================================================================

class LRUCache(OrderedDict):
    """
    动态大小最近最少使用缓存。
    当缓存超过 settings 中的 max_cache_size 时，自动剔除最旧的数据，并清理显存。
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

        # 动态获取最大缓存大小，默认 30 帧
        settings = settings_manager.load_settings()
        maxsize = int(settings.get('max_cache_size', 30))

        while len(self) > maxsize:
            oldest = next(iter(self))
            del self[oldest]
            # 触发显存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


models = {}
# 使用 LRU Cache 替代普通字典，防止显存爆炸
PREPROCESSED_DATA_CACHE = LRUCache()

# New structure: {'class_name': {'semantic': Tensor, 'color': np.array}}
PROTOTYPE_CACHE = {}

AI_MODEL_LOCK = threading.RLock()
PROTOTYPE_LOCKS = {}
_PROTOTYPE_LOCKS_LOCK = threading.Lock()
_cache_save_lock = threading.Lock()
_last_cache_save_time = 0


def _get_class_lock(class_name):
    with _PROTOTYPE_LOCKS_LOCK:
        if class_name not in PROTOTYPE_LOCKS:
            PROTOTYPE_LOCKS[class_name] = threading.Lock()
        return PROTOTYPE_LOCKS[class_name]


_mobilenet_pytorch_cache = {"model": None, "name": None}


def get_features_for_single_bbox(pil_image, target_rects):
    """
    使用 MobileNetV3 提取指定 BBox 区域的语义特征向量
    """
    if 'feature_extractor_pytorch' not in models:
        # 尝试懒加载
        startup_ai_models()
        if 'feature_extractor_pytorch' not in models:
            raise RuntimeError("PyTorch特征提取模型未加载，请检查启动过程。")

    input_size = 0
    if target_rects is not None:
        if isinstance(target_rects, np.ndarray):
            input_size = target_rects.shape[0]
        else:
            try:
                input_size = len(target_rects)
            except TypeError:
                input_size = 0

    if input_size == 0:
        return None

    DEVICE = settings_manager.get_device()
    pytorch_model = models['feature_extractor_pytorch']

    with torch.no_grad():
        img_tensor = to_tensor(pil_image).to(DEVICE)

        if not isinstance(target_rects, np.ndarray):
            target_rects_np = np.array(target_rects, dtype=np.float32)
        else:
            target_rects_np = target_rects.astype(np.float32)

        boxes_for_crop = torch.from_numpy(target_rects_np).to(DEVICE)

        # 构建 ROI Align 输入格式:[batch_index, x1, y1, x2, y2]
        box_indices = torch.zeros(boxes_for_crop.size(0), 1, device=DEVICE)
        boxes_for_roi = torch.cat([box_indices, boxes_for_crop], dim=1)

        # ROI Align 提取特征图
        batch_of_crops = torchvision.ops.roi_align(
            img_tensor.unsqueeze(0),
            boxes_for_roi,
            output_size=(224, 224),
            spatial_scale=1.0,
            aligned=True
        )

        # ImageNet 标准化
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        batch_tensor = (batch_of_crops - IMAGENET_MEAN) / IMAGENET_STD

        # 混合精度推理 (受设置控制)
        use_autocast = settings_manager.load_settings().get('use_autocast', True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda' and use_autocast)):
            features_map = pytorch_model.features(batch_tensor)
            pooled_features = pytorch_model.avgpool(features_map)
            final_features = torch.flatten(pooled_features, 1)

        return final_features


def save_prototypes_to_disk():
    try:
        with _get_class_lock("__global_save__"):
            # Deep copy and move tensors to CPU for saving
            cpu_cache = {}
            # Use list(...) to snapshot to avoid RuntimeError during iteration
            current_items = list(PROTOTYPE_CACHE.items())

            for k, v in current_items:
                if isinstance(v, dict):
                    cpu_cache[k] = {
                        'semantic': v['semantic'].cpu() if isinstance(v.get('semantic'), torch.Tensor) else v.get(
                            'semantic'),
                        'color': v.get('color')  # numpy array, already cpu
                    }
                elif isinstance(v, torch.Tensor):
                    # Legacy support during transition
                    cpu_cache[k] = {'semantic': v.cpu(), 'color': None}

        torch.save(cpu_cache, config.PROTOTYPE_FILE)
        logging.info(f"成功将 {len(cpu_cache)} 个原型保存至磁盘。")
    except Exception as e:
        logging.error(f"保存原型文件失败: {e}", exc_info=True)


def save_preprocessed_cache_to_disk():
    """
    保存预处理缓存到磁盘。
    注意：LRUCache 中的数据已经在 CPU 上，可以直接保存。
    """
    global _last_cache_save_time
    with _cache_save_lock:
        logging.info("正在尝试后台保存预处理缓存...")
        # 转换为普通字典保存，避免序列化问题
        cache_copy = dict(PREPROCESSED_DATA_CACHE)
        if not cache_copy:
            logging.info("预处理缓存为空，跳过保存。")
            return

        try:
            # 这里的 value 已经是 CPU tensor 了
            torch.save(cache_copy, config.PREPROCESSED_CACHE_FILE)
            _last_cache_save_time = time.time()
            logging.info(f"成功将 {len(cache_copy)} 个预处理帧数据保存至文件。")
        except Exception as e:
            logging.error(f"保存预处理缓存文件失败: {e}", exc_info=True)


def load_prototypes_from_disk():
    global PROTOTYPE_CACHE
    DEVICE = settings_manager.get_device()
    if os.path.exists(config.PROTOTYPE_FILE):
        try:
            # FIX: weights_only=False required for PyTorch 2.6+ with numpy/dicts
            try:
                loaded_cache = torch.load(config.PROTOTYPE_FILE, map_location=DEVICE, weights_only=False)
            except TypeError:
                loaded_cache = torch.load(config.PROTOTYPE_FILE, map_location=DEVICE)

            # Check for legacy format (Direct Tensor vs Dict)
            if loaded_cache and isinstance(next(iter(loaded_cache.values())), torch.Tensor):
                logging.warning("检测到旧版原型缓存格式。正在清除缓存以强制重建...")
                PROTOTYPE_CACHE = {}
            else:
                PROTOTYPE_CACHE = loaded_cache
                logging.info(f"成功加载 {len(PROTOTYPE_CACHE)} 个类别原型。")
        except Exception as e:
            logging.error(f"加载原型文件失败: {e}")
            PROTOTYPE_CACHE = {}
    else:
        PROTOTYPE_CACHE = {}


def clear_feature_extractor_cache():
    global _mobilenet_pytorch_cache
    logging.info("清除 Feature Extractor 模型缓存。")
    _mobilenet_pytorch_cache = {"model": None, "name": None}
    if 'feature_extractor_pytorch' in models:
        del models['feature_extractor_pytorch']
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_preprocessed_cache_from_disk():
    """
    从磁盘加载缓存，并填充到 LRUCache 中。
    注意：数据加载到内存(CPU)，只有在使用时才上传 GPU。
    """
    global PREPROCESSED_DATA_CACHE
    # 重新初始化 LRU Cache 确保清空
    PREPROCESSED_DATA_CACHE = LRUCache()

    if os.path.exists(config.PREPROCESSED_CACHE_FILE):
        try:
            logging.info("正在从磁盘加载预处理缓存...")
            try:
                loaded_cache = torch.load(config.PREPROCESSED_CACHE_FILE, map_location='cpu', weights_only=False)
            except TypeError:
                loaded_cache = torch.load(config.PREPROCESSED_CACHE_FILE, map_location='cpu')

            # 填充 LRUCache，保持在 CPU 上
            for key, value in loaded_cache.items():
                PREPROCESSED_DATA_CACHE[key] = {
                    'all_boxes': value['all_boxes'].cpu(),
                    'all_features': value['all_features'].cpu()
                }
            logging.info(f"成功加载 {len(PREPROCESSED_DATA_CACHE)} 个预处理帧数据 (CPU Mode)。")
        except Exception as e:
            logging.error(f"加载预处理缓存文件失败: {e}")
            # 出错时重置
            PREPROCESSED_DATA_CACHE = LRUCache()
    else:
        logging.info("未找到预处理缓存文件。")


def startup_ai_models():
    """
    初始化 AI 模型，会根据系统设置选择性地加载。
    """
    load_prototypes_from_disk()
    load_preprocessed_cache_from_disk()

    global _mobilenet_pytorch_cache
    DEVICE = settings_manager.get_device()
    settings = settings_manager.load_settings()  # 获取最新设置

    # 1. 检查并初始化 SAM 模型（如果用户已启用）
    if settings.get('enable_sam_model', True):
        if sam_tasks:
            logging.info("正在初始化 SAM2 环境 (根据设置已启用)...")
            try:
                # 触发一次检查，这可能会预加载模型
                sam_tasks.get_sam_model()
            except Exception as e:
                logging.error(f"SAM2 初始化检查失败: {e}")
        else:
            logging.warning("SAM 库未安装，即使已在设置中启用，相关功能也无法使用。")
    else:
        logging.warning("SAM 模型已在系统设置中被禁用。将跳过SAM模型加载以节省资源。")

    # 2. 检查并加载 MobileNet 特征提取器（如果用户已启用）
    if not settings.get('enable_feature_extractor', True):
        logging.warning("特征提取器已在系统设置中被禁用。将跳过模型加载以节省资源。")
        # 确保如果之前加载过模型，现在也被彻底清理掉
        if 'feature_extractor_pytorch' in models:
            del models['feature_extractor_pytorch']
        _mobilenet_pytorch_cache = {"model": None, "name": None}
        # 如果特征提取器被禁用，直接结束函数，不再继续加载
        return

    # 只有在特征提取器启用时，才执行以下加载逻辑
    try:
        target_model_name = settings.get("feature_extractor_model_name", "mobilenet_v3_large")

        # 检查缓存，如果模型已存在且配置匹配，则跳过
        if (_mobilenet_pytorch_cache.get("model") is not None and
                _mobilenet_pytorch_cache.get("name") == target_model_name and
                next(_mobilenet_pytorch_cache["model"].parameters()).device == DEVICE):
            models['feature_extractor_pytorch'] = _mobilenet_pytorch_cache["model"]
            logging.info(f"已从缓存加载 PyTorch 特征提取器 '{target_model_name}'。")
            return

        logging.info(f"正在加载 PyTorch 特征提取器 '{target_model_name}'到设备'{DEVICE}' (根据设置已启用)...")

        if target_model_name == "mobilenet_v3_small":
            model = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        # 移除分类头，我们只需要特征向量
        model.classifier = torch.nn.Identity()

        model.eval()
        model.to(DEVICE)

        # 更新缓存
        _mobilenet_pytorch_cache["model"] = model
        _mobilenet_pytorch_cache["name"] = target_model_name
        models['feature_extractor_pytorch'] = model

        logging.info(f"PyTorch 特征提取器 '{target_model_name}' 加载成功。")

    except Exception as e:
        logging.error(f"加载 PyTorch 特征提取器失败: {e}", exc_info=True)
        # 加载失败时，清空所有相关引用
        if 'feature_extractor_pytorch' in models:
            del models['feature_extractor_pytorch']
        _mobilenet_pytorch_cache = {"model": None, "name": None}


def postprocess_sam_results(results, nms_iou_threshold):
    # 此函数主要用于旧版 SAM1，SAM2 已内置 NMS，保留以防万一
    DEVICE = settings_manager.get_device()
    if not results or not results[0].masks:
        return torch.empty(0, 4, device=DEVICE), torch.empty(0, 1, 1, device=DEVICE)
    all_boxes = results[0].boxes.xyxy.to(DEVICE)
    all_scores = results[0].boxes.conf.to(DEVICE)
    all_masks = results[0].masks.data.to(DEVICE)
    kept_indices = nms(all_boxes, all_scores, nms_iou_threshold)
    final_boxes = all_boxes[kept_indices]
    final_masks = all_masks[kept_indices]
    return final_boxes, final_masks


def find_best_matching_masks_by_iou(reference_boxes_np, candidate_boxes_tensor):
    DEVICE = settings_manager.get_device()
    if len(reference_boxes_np) == 0 or len(candidate_boxes_tensor) == 0:
        return torch.tensor([], dtype=torch.long, device=DEVICE)
    reference_boxes_tensor = torch.tensor(reference_boxes_np, dtype=torch.float32, device=DEVICE)
    iou_matrix = box_iou(reference_boxes_tensor, candidate_boxes_tensor)
    best_match_indices = torch.argmax(iou_matrix, dim=1)
    return best_match_indices


def get_features_for_all_masks(video_uuid, frame_number):
    """
    核心函数：获取当前帧所有潜在物体的特征。
    优化策略：
    1. 优先查缓存。
    2. 缓存命中时，从 CPU 搬运到 GPU。
    3. 缓存未命中时，计算后将数据搬运到 CPU 并存入缓存。
    """
    if 'feature_extractor_pytorch' not in models:
        # 尝试重新加载
        startup_ai_models()
        if 'feature_extractor_pytorch' not in models:
            raise RuntimeError("PyTorch特征提取模型加载失败。")

    DEVICE = settings_manager.get_device()
    cache_key = f"{video_uuid}_{frame_number}"

    # 1. 检查一级缓存 (快速路径)
    if cache_key in PREPROCESSED_DATA_CACHE:
        # 数据在 CPU 上，转移到 GPU 用于计算
        data_cpu = PREPROCESSED_DATA_CACHE[cache_key]
        return {
            "all_boxes": data_cpu["all_boxes"].to(DEVICE),
            "all_features": data_cpu["all_features"].to(DEVICE)
        }

    with AI_MODEL_LOCK:
        # 2. 检查二级缓存 (防止并发计算)
        if cache_key in PREPROCESSED_DATA_CACHE:
            data_cpu = PREPROCESSED_DATA_CACHE[cache_key]
            return {
                "all_boxes": data_cpu["all_boxes"].to(DEVICE),
                "all_features": data_cpu["all_features"].to(DEVICE)
            }

        use_autocast = settings_manager.load_settings().get('use_autocast', True)
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type,
                                                 enabled=(DEVICE.type == 'cuda' and use_autocast)):
            logging.info(f"正在为 {cache_key} 生成掩码特征 (SAM2 + MobileNet)...")

            frame_path = file_storage.get_frame_path(video_uuid, frame_number)
            if not os.path.exists(frame_path):
                raise FileNotFoundError(f"Frame image not found: {frame_path}")

            all_boxes, all_masks = sam_tasks.generate_masks_for_frame(video_uuid, frame_number)

            # 处理未检测到对象的情况
            if all_boxes is None or len(all_boxes) == 0:
                cached_data_empty = {
                    "all_boxes": torch.empty(0, 4, device='cpu'),
                    "all_features": torch.empty(0, 1, device='cpu')
                }
                PREPROCESSED_DATA_CACHE[cache_key] = cached_data_empty
                return {
                    "all_boxes": torch.empty(0, 4, device=DEVICE),
                    "all_features": torch.empty(0, 1, device=DEVICE)
                }

            # 3. 提取语义特征 (MobileNet)
            try:
                pil_image = Image.open(frame_path).convert("RGB")
            except Exception as e:
                logging.error(f"Image load failed: {e}")
                return None

            # 这里的 all_boxes 在 GPU 上，需要转 numpy 给 feature extractor (crop logic)
            # 注意：get_features_for_single_bbox 内部会将 numpy 转回 GPU tensor
            all_features = get_features_for_single_bbox(pil_image, all_boxes.cpu().numpy())

            if all_features is None:
                raise RuntimeError(f"Feature extraction returned None for {cache_key}")

            # 4. 存入缓存 (CRITICAL OPTIMIZATION: Move to CPU)
            cached_data_cpu = {
                "all_boxes": all_boxes.cpu(),
                "all_features": all_features.cpu()
            }
            PREPROCESSED_DATA_CACHE[cache_key] = cached_data_cpu

            logging.info(f"缓存成功: {cache_key} (Count: {len(PREPROCESSED_DATA_CACHE)})")

            # 5. 触发后台保存 (定时)
            settings = settings_manager.load_settings()
            cache_save_interval = settings.get('cache_save_interval_seconds')
            # 容错：如果 JSON 中存入了 null 或者非法值，强制恢复默认值 30
            if cache_save_interval is None:
                cache_save_interval = 30

            if time.time() - _last_cache_save_time > cache_save_interval:
                threading.Thread(target=save_preprocessed_cache_to_disk).start()

            # 返回 GPU 数据给当前计算使用
            return {"all_boxes": all_boxes, "all_features": all_features}


def get_features_for_specific_bboxes(video_uuid, frame_number, target_rects):
    try:
        processed_data = get_features_for_all_masks(video_uuid, frame_number)
        if processed_data is None: return None

        all_boxes = processed_data.get("all_boxes")
        all_features = processed_data.get("all_features")

        if all_boxes is None or all_boxes.numel() == 0 or all_features is None or all_features.numel() == 0:
            return None

        matching_indices = find_best_matching_masks_by_iou(np.array(target_rects), all_boxes)
        if matching_indices.numel() > 0:
            return all_features[matching_indices]
        else:
            return None

    except Exception as e:
        logging.warning(f"Feature match failed for {frame_number}: {e}")
        return None


def get_prototypes_from_drawn_boxes(drawn_samples_data):
    all_prototypes = []
    if not drawn_samples_data:
        return None

    logging.info(f"Building on-the-fly prototypes from {len(drawn_samples_data)} user-drawn sample frames.")

    for frame_key, rects in drawn_samples_data.items():
        try:
            video_uuid, frame_number_str = frame_key.split(';')
            frame_num = int(frame_number_str)
            target_rects = [np.array(rect) for rect in rects]
            if not target_rects: continue

            embeddings = get_features_for_specific_bboxes(video_uuid, frame_num, target_rects)

            if embeddings is not None and embeddings.numel() > 0:
                all_prototypes.append(embeddings)

        except Exception as e:
            logging.warning(f"Skipping frame {frame_key} for prototype building: {e}")

    if not all_prototypes:
        logging.error("Could not extract any valid prototypes from user samples.")
        return None

    return torch.cat(all_prototypes, dim=0)


def _calculate_region_color_hist(image_bgr, rect):
    """
    计算区域颜色直方图 (Center Crop 50%)
    """
    x1, y1, x2, y2 = map(int, rect)
    h, w = image_bgr.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    crop_margin_x = int(bw * 0.25)
    crop_margin_y = int(bh * 0.25)

    cx1 = x1 + crop_margin_x
    cy1 = y1 + crop_margin_y
    cx2 = x2 - crop_margin_x
    cy2 = y2 - crop_margin_y

    if (cx2 - cx1) < 2 or (cy2 - cy1) < 2:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    roi = image_bgr[cy1:cy2, cx1:cx2]

    if roi.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def predict_from_one_shot(video_uuid, frame_number, positive_prompt_box, use_color=False):
    """
    智能选择 (Smart Select)
    """
    with AI_MODEL_LOCK:
        processed_data = get_features_for_all_masks(video_uuid, frame_number)
        all_boxes = processed_data.get("all_boxes")
        all_features = processed_data.get("all_features")

        if all_boxes is None or all_boxes.numel() == 0: return []

        prompt_rect = [positive_prompt_box['x1'], positive_prompt_box['y1'], positive_prompt_box['x2'],
                       positive_prompt_box['y2']]

        # 1. Semantic Similarity
        target_feature_tensor = get_features_for_specific_bboxes(video_uuid, frame_number, [prompt_rect])
        if target_feature_tensor is None or target_feature_tensor.numel() == 0:
            raise ValueError("Could not extract features for the prompt box.")

        target_feature = target_feature_tensor[0].unsqueeze(0)
        DEVICE = settings_manager.get_device()

        use_autocast = settings_manager.load_settings().get('use_autocast', True)
        with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda' and use_autocast)):
            sim_scores = F.cosine_similarity(target_feature, all_features, dim=1)

        # 2. Color Similarity (Optional)
        if use_color:
            try:
                frame_path = file_storage.get_frame_path(video_uuid, frame_number)
                image_bgr = cv2.imread(frame_path)

                if image_bgr is not None:
                    target_hist = _calculate_region_color_hist(image_bgr, prompt_rect)

                    if target_hist is not None:
                        color_scores = []
                        all_boxes_cpu = all_boxes.cpu().numpy()

                        for box in all_boxes_cpu:
                            cand_hist = _calculate_region_color_hist(image_bgr, box)
                            if cand_hist is not None:
                                dist = cv2.compareHist(target_hist, cand_hist, cv2.HISTCMP_BHATTACHARYYA)
                                score = max(0.0, 1.0 - dist)
                                color_scores.append(score)
                            else:
                                color_scores.append(0.0)

                        color_scores_tensor = torch.tensor(color_scores, device=DEVICE, dtype=torch.float32)

                        # Veto Logic
                        veto_mask = (color_scores_tensor < 0.65).float()
                        penalty_factor = (1.0 - veto_mask) + (veto_mask * 0.1)
                        combined = (sim_scores * 0.5) + (color_scores_tensor * 0.5)
                        sim_scores = combined * penalty_factor
            except Exception as e:
                logging.error(f"Color similarity error: {e}")

        settings = settings_manager.load_settings()
        nms_iou = settings.get('nms_iou_threshold', 0.7)

        kept_indices = nms(all_boxes, sim_scores, nms_iou)

        final_results = []
        final_scores_np = sim_scores.cpu().numpy()

        for i in kept_indices:
            box_coords = all_boxes[i].cpu().numpy().astype(int).tolist()
            score = float(final_scores_np[i])
            if score > 0.05:
                final_results.append({"box": box_coords, "score": score})

        return final_results


def _calculate_similarity_scores(all_embeddings, positive_prototypes_dict, negative_prototypes=None):
    settings = settings_manager.load_settings()
    score_temperature = settings.get('prototype_temperature', 0.07)
    use_autocast = settings.get('use_autocast', True)
    DEVICE = settings_manager.get_device()

    positive_semantic = positive_prototypes_dict['semantic']

    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda' and use_autocast)):
        sim_matrix = F.cosine_similarity(all_embeddings.unsqueeze(1), positive_semantic.unsqueeze(0), dim=2)
        positive_scores_sim, _ = torch.max(sim_matrix, dim=1)

        if negative_prototypes is not None and len(negative_prototypes) > 0:
            if negative_prototypes.dim() > 1 and negative_prototypes.shape[0] > 1:
                neg_sim_matrix = F.cosine_similarity(all_embeddings.unsqueeze(1), negative_prototypes.unsqueeze(0),
                                                     dim=2)
                negative_scores_sim, _ = torch.max(neg_sim_matrix, dim=1)
            else:
                mean_negative_prototype = torch.mean(negative_prototypes, dim=0, keepdim=True)
                negative_scores_sim = F.cosine_similarity(all_embeddings, mean_negative_prototype)

            logits = torch.stack([negative_scores_sim, positive_scores_sim], dim=1)
            probabilities = F.softmax(logits / score_temperature, dim=1)
            final_scores = probabilities[:, 1]
        else:
            final_scores = torch.sigmoid(positive_scores_sim / score_temperature)

    return final_scores


def predict_with_prototypes(video_uuid, frame_number, positive_prototypes_dict, negative_prototypes=None,
                            confidence_threshold=0.5, use_color=False):
    """
    使用原型库进行预测
    """
    with AI_MODEL_LOCK:
        processed_data = get_features_for_all_masks(video_uuid, frame_number)
        all_boxes = processed_data.get("all_boxes")
        all_features = processed_data.get("all_features")

        if all_boxes is None or all_boxes.numel() == 0:
            return []

        # 1. Semantic
        final_scores = _calculate_similarity_scores(all_features, positive_prototypes_dict, negative_prototypes)

        # 2. Color Veto
        if use_color and 'color' in positive_prototypes_dict and positive_prototypes_dict['color'] is not None:
            try:
                frame_path = file_storage.get_frame_path(video_uuid, frame_number)
                image_bgr = cv2.imread(frame_path)

                if image_bgr is not None:
                    target_hist = positive_prototypes_dict['color']
                    color_scores = []
                    all_boxes_cpu = all_boxes.cpu().numpy()

                    for box in all_boxes_cpu:
                        cand_hist = _calculate_region_color_hist(image_bgr, box)
                        if cand_hist is not None:
                            dist = cv2.compareHist(target_hist, cand_hist, cv2.HISTCMP_BHATTACHARYYA)
                            score = max(0.0, 1.0 - dist)
                            color_scores.append(score)
                        else:
                            color_scores.append(0.0)

                    DEVICE = settings_manager.get_device()
                    color_scores_tensor = torch.tensor(color_scores, device=DEVICE, dtype=torch.float32)

                    veto_mask = (color_scores_tensor < 0.65).float()
                    penalty_factor = (1.0 - veto_mask) + (veto_mask * 0.1)

                    combined = (final_scores * 0.7) + (color_scores_tensor * 0.3)
                    final_scores = combined * penalty_factor
            except Exception as e:
                logging.error(f"Color proto calc error: {e}")

        settings = settings_manager.load_settings()
        nms_iou = settings.get('nms_iou_threshold', 0.7)

        high_conf_indices = torch.where(final_scores > confidence_threshold)[0]
        if high_conf_indices.numel() == 0:
            return []

        boxes_to_nms = all_boxes[high_conf_indices]
        scores_to_nms = final_scores[high_conf_indices]

        kept_indices_after_nms = nms(boxes_to_nms, scores_to_nms, nms_iou)
        final_kept_indices = high_conf_indices[kept_indices_after_nms]

        final_results = []
        final_scores_np = final_scores.cpu().numpy()
        for i in final_kept_indices:
            box_coords = all_boxes[i].cpu().numpy().astype(int).tolist()
            final_results.append({"box": box_coords, "score": float(final_scores_np[i])})

        return final_results


def _calculate_prototype_from_db(class_name):
    """
    从数据库计算原型
    """
    all_class_features_tensors = []
    all_color_hists = []

    sample_frames = database.get_all_frames_with_class(class_name)
    if not sample_frames:
        logging.warning(f"No samples found for class '{class_name}'.")
        return None

    settings = settings_manager.load_settings()
    sample_limit = int(settings.get('prototype_sample_limit', 50))

    if len(sample_frames) > sample_limit:
        sample_frames = random.sample(sample_frames, sample_limit)

    grouped_boxes = defaultdict(list)
    for frame_data in sample_frames:
        frame_key = (frame_data['video_uuid'], frame_data['frame_number'])
        rects, labels, _ = convert_text_to_rects_and_labels(frame_data['bboxes_text'])
        target_rects_in_frame = [rect for i, rect in enumerate(rects) if labels[i] == class_name]
        if target_rects_in_frame:
            grouped_boxes[frame_key].extend(target_rects_in_frame)

    if not grouped_boxes: return None

    for (video_uuid, frame_number), all_rects_for_frame in grouped_boxes.items():
        try:
            pil_image = Image.open(file_storage.get_frame_path(video_uuid, frame_number)).convert("RGB")
            image_bgr = cv2.imread(file_storage.get_frame_path(video_uuid, frame_number))

            # 1. Semantic
            for i in range(0, len(all_rects_for_frame), 64):
                rect_chunk = all_rects_for_frame[i:i + 64]
                features = get_features_for_single_bbox(pil_image, rect_chunk)
                if features is not None and features.numel() > 0:
                    all_class_features_tensors.append(features)

            # 2. Color
            if image_bgr is not None:
                for rect in all_rects_for_frame:
                    hist = _calculate_region_color_hist(image_bgr, rect)
                    if hist is not None:
                        all_color_hists.append(hist)

        except Exception as e:
            logging.warning(f"Error extracting features for prototype {video_uuid}/{frame_number}: {e}")

    if not all_class_features_tensors: return None

    all_features = torch.cat(all_class_features_tensors, dim=0)
    num_samples = all_features.shape[0]

    # Clustering
    MIN_SAMPLES = 15
    if KMeans is None or num_samples < MIN_SAMPLES:
        semantic_prototype = torch.mean(all_features, dim=0, keepdim=True)
    else:
        best_k = 1
        best_score = -1
        max_clusters = min(5, num_samples - 1)
        features_np = all_features.cpu().numpy()

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(features_np)
            try:
                score = silhouette_score(features_np, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError:
                pass

        if best_k > 1 and best_score > 0.55:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
            kmeans.fit(features_np)
            prototypes_np = kmeans.cluster_centers_
            semantic_prototype = torch.from_numpy(prototypes_np).to(all_features.device)
        else:
            semantic_prototype = torch.mean(all_features, dim=0, keepdim=True)

    color_prototype = None
    if all_color_hists:
        color_prototype = np.mean(np.array(all_color_hists), axis=0)

    return {'semantic': semantic_prototype, 'color': color_prototype}


def build_prototypes_for_class(class_name):
    if class_name in PROTOTYPE_CACHE:
        return PROTOTYPE_CACHE[class_name]

    class_lock = _get_class_lock(class_name)
    with class_lock:
        if class_name in PROTOTYPE_CACHE:
            return PROTOTYPE_CACHE[class_name]

        prototype_dict = _calculate_prototype_from_db(class_name)

        if prototype_dict is not None:
            if prototype_dict['semantic'].dim() == 1:
                prototype_dict['semantic'] = prototype_dict['semantic'].unsqueeze(0)

            PROTOTYPE_CACHE[class_name] = prototype_dict
            logging.info(f"Prototype built for '{class_name}'.")
            save_prototypes_to_disk()

        return prototype_dict


def update_prototype_for_class(class_name):
    class_lock = _get_class_lock(class_name)
    with class_lock:
        new_prototype = _calculate_prototype_from_db(class_name)
        if new_prototype is not None:
            PROTOTYPE_CACHE[class_name] = new_prototype
            save_prototypes_to_disk()


def get_all_prototypes():
    all_labels = database.get_all_class_labels()
    prototype_library = {}
    for label in all_labels:
        prototype = build_prototypes_for_class(label)
        if prototype is not None:
            prototype_library[label] = prototype
    return prototype_library


def lam_predict(video_uuid, frame_number, point_coords):
    with AI_MODEL_LOCK:
        frame_path = file_storage.get_frame_path(video_uuid, frame_number)

        # 1. SAM Point Predict
        bbox_dict = sam_tasks.predict_box_from_point_ultralytics(frame_path, point_coords)
        if not bbox_dict: return None, "SAM failed to find object."

        # 2. Extract Features
        bbox_coords_list = [bbox_dict['x1'], bbox_dict['y1'], bbox_dict['x2'], bbox_dict['y2']]
        feature_vector = get_features_for_specific_bboxes(video_uuid, frame_number, [bbox_coords_list])

        if feature_vector is None or feature_vector.numel() == 0:
            return None, "Failed to extract features for LAM."

        # 3. Match against prototypes
        prototype_library = get_all_prototypes()
        if not prototype_library:
            return {"bbox": bbox_dict, "suggestions": []}, None

        scores = []
        DEVICE = settings_manager.get_device()
        use_autocast = settings_manager.load_settings().get('use_autocast', True)

        with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda' and use_autocast)):
            for class_name, prototype_dict in prototype_library.items():
                semantic_proto = prototype_dict['semantic']
                sim_matrix = F.cosine_similarity(feature_vector.unsqueeze(1), semantic_proto.unsqueeze(0), dim=2)
                max_similarity = torch.max(sim_matrix)
                scores.append({"label": class_name, "score": round(max_similarity.item(), 4)})

        sorted_suggestions = sorted(scores, key=lambda x: x['score'], reverse=True)
        return {"bbox": bbox_dict, "suggestions": sorted_suggestions[:5]}, None