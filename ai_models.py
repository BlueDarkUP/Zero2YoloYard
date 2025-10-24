import logging
import os
import random
import threading
import time

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    logging.warning("scikit-learn not found. Sub-prototype clustering will be disabled. Run 'pip install scikit-learn'")
    KMeans = None
    silhouette_score = None

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.amp import autocast
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.ops import nms, box_iou
from torchvision.transforms.functional import to_tensor, normalize

import config
import database
import file_storage
import settings_manager
from bbox_writer import convert_text_to_rects_and_labels
from collections import defaultdict

try:
    import ultralytics_sam_tasks as sam_tasks
except ImportError:
    logging.warning("ultralytics_sam_tasks.py not found or failed to import. All SAM features will be disabled.")
    sam_tasks = None

models = {}
PREPROCESSED_DATA_CACHE = {}
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

        box_indices = torch.zeros(boxes_for_crop.size(0), 1, device=DEVICE)
        boxes_for_roi = torch.cat([box_indices, boxes_for_crop], dim=1)

        batch_of_crops = torchvision.ops.roi_align(
            img_tensor.unsqueeze(0),
            boxes_for_roi,
            output_size=(224, 224),
            spatial_scale=1.0,
            aligned=True
        )

        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        batch_tensor = (batch_of_crops - IMAGENET_MEAN) / IMAGENET_STD

        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            features_map = pytorch_model.features(batch_tensor)
            pooled_features = pytorch_model.avgpool(features_map)
            final_features = torch.flatten(pooled_features, 1)

        return final_features

def save_prototypes_to_disk():
    try:
        with _get_class_lock("__global_save__"):
            cpu_cache = {k: v.cpu() for k, v in PROTOTYPE_CACHE.items()}
        torch.save(cpu_cache, config.PROTOTYPE_FILE)
        logging.info(f"成功将 {len(cpu_cache)} 个原型保存至 {config.PROTOTYPE_FILE}")
    except Exception as e:
        logging.error(f"保存原型文件失败: {e}", exc_info=True)


def save_preprocessed_cache_to_disk():
    global _last_cache_save_time
    with _cache_save_lock:
        logging.info("正在尝试保存预处理缓存...")
        cache_copy = dict(PREPROCESSED_DATA_CACHE)
        if not cache_copy:
            logging.info("预处理缓存为空，无需保存。")
            return

        try:
            cpu_cache = {}
            for key, value in cache_copy.items():
                cpu_cache[key] = {
                    'all_boxes': value['all_boxes'].cpu(),
                    'all_features': value['all_features'].cpu()
                }

            torch.save(cpu_cache, config.PREPROCESSED_CACHE_FILE)
            _last_cache_save_time = time.time()
            logging.info(f"成功将 {len(cpu_cache)} 个预处理帧数据保存至文件。")
        except Exception as e:
            logging.error(f"保存预处理缓存文件失败: {e}", exc_info=True)


def load_prototypes_from_disk():
    global PROTOTYPE_CACHE
    DEVICE = settings_manager.get_device()
    if os.path.exists(config.PROTOTYPE_FILE):
        try:
            loaded_cache = torch.load(config.PROTOTYPE_FILE, map_location=DEVICE)
            PROTOTYPE_CACHE = loaded_cache
            logging.info(f"成功从文件加载了 {len(PROTOTYPE_CACHE)} 个类别原型。")
        except Exception as e:
            logging.error(f"加载原型文件失败，将在需要时重新构建: {e}")
            PROTOTYPE_CACHE = {}
    else:
        logging.info("未找到原型文件。将在首次需要时自动创建。")
        PROTOTYPE_CACHE = {}


def clear_feature_extractor_cache():
    global _mobilenet_cache
    logging.info("Clearing Feature Extractor model cache due to setting change.")
    _mobilenet_cache = {"model": None, "name": None}
    if 'feature_extractor' in models:
        del models['feature_extractor']
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_preprocessed_cache_from_disk():
    global PREPROCESSED_DATA_CACHE
    DEVICE = settings_manager.get_device()
    if os.path.exists(config.PREPROCESSED_CACHE_FILE):
        try:
            logging.info("正在从磁盘加载预处理缓存...")
            loaded_cache = torch.load(config.PREPROCESSED_CACHE_FILE, map_location='cpu')

            for key, value in loaded_cache.items():
                PREPROCESSED_DATA_CACHE[key] = {
                    'all_boxes': value['all_boxes'].to(DEVICE),
                    'all_features': value['all_features'].to(DEVICE)
                }
            logging.info(f"成功从文件加载了 {len(PREPROCESSED_DATA_CACHE)} 个预处理帧数据。")
        except Exception as e:
            logging.error(f"加载预处理缓存文件失败: {e}")
            PREPROCESSED_DATA_CACHE = {}
    else:
        logging.info("未找到预处理缓存文件。")
        PREPROCESSED_DATA_CACHE = {}


def startup_ai_models():
    load_prototypes_from_disk()
    load_preprocessed_cache_from_disk()

    global _mobilenet_pytorch_cache
    DEVICE = settings_manager.get_device()

    if sam_tasks:
        logging.info("正在检查 SAM 点选/跟踪模型...")
        sam_tasks.get_sam_model()
        logging.info("SAM 点选/跟踪模型检查完成。")

    try:
        settings = settings_manager.load_settings()
        target_model_name = settings.get("feature_extractor_model_name", "mobilenet_v3_large")

        if (_mobilenet_pytorch_cache.get("model") is not None and
                _mobilenet_pytorch_cache.get("name") == target_model_name and
                next(_mobilenet_pytorch_cache["model"].parameters()).device == DEVICE):
            models['feature_extractor_pytorch'] = _mobilenet_pytorch_cache["model"]
            logging.info(f"已从缓存加载 PyTorch 特征提取器 '{target_model_name}'。")
            return

        logging.info(f"正在加载原生 PyTorch 特征提取器 '{target_model_name}' 到设备 '{DEVICE}'...")

        if target_model_name == "mobilenet_v3_small":
            model = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        model.classifier = torch.nn.Identity()

        model.eval()
        model.to(DEVICE)

        _mobilenet_pytorch_cache["model"] = model
        _mobilenet_pytorch_cache["name"] = target_model_name

        models['feature_extractor_pytorch'] = model

        logging.info(f"PyTorch 特征提取器 '{target_model_name}' 加载成功。")

    except Exception as e:
        logging.error(f"加载 PyTorch 特征提取器失败: {e}", exc_info=True)
        if 'feature_extractor_pytorch' in models:
            del models['feature_extractor_pytorch']
        _mobilenet_pytorch_cache = {"model": None, "name": None}


def postprocess_sam_results(results, nms_iou_threshold):
    DEVICE = settings_manager.get_device()
    if not results or not results[0].masks:
        return torch.empty(0, 4, device=DEVICE), torch.empty(0, 1, 1, device=DEVICE)
    all_boxes = results[0].boxes.xyxy.to(DEVICE)
    all_scores = results[0].boxes.conf.to(DEVICE)
    all_masks = results[0].masks.data.to(DEVICE)
    kept_indices = nms(all_boxes, all_scores, nms_iou_threshold)
    logging.info(f"[智能选择] NMS: 从 {len(all_boxes)} 个初始掩码中保留了 {len(kept_indices)} 个。")
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
    if 'feature_extractor_pytorch' not in models:
        raise RuntimeError("PyTorch特征提取模型未加载。")

    DEVICE = settings_manager.get_device()
    cache_key = f"{video_uuid}_{frame_number}"

    if cache_key in PREPROCESSED_DATA_CACHE:
        return PREPROCESSED_DATA_CACHE[cache_key]

    with AI_MODEL_LOCK:
        if cache_key in PREPROCESSED_DATA_CACHE:
            return PREPROCESSED_DATA_CACHE[cache_key]

        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            logging.info(f"正在为 {cache_key} 开始新的预处理...")
            frame_path = file_storage.get_frame_path(video_uuid, frame_number)
            if not os.path.exists(frame_path):
                raise FileNotFoundError(f"帧图像文件未找到于 {frame_path}")

            sam_model = sam_tasks.get_sam_model()
            if not sam_model:
                raise RuntimeError("SAM model not loaded.")

            settings = settings_manager.load_settings()
            results = sam_model(frame_path, verbose=False, conf=settings.get('sam_mask_confidence', 0.35))
            all_boxes, all_masks = postprocess_sam_results(results,
                                                           nms_iou_threshold=settings.get('nms_iou_threshold', 0.7))

            if len(all_masks) == 0:
                cached_data = {"all_boxes": torch.empty(0, 4, device=DEVICE),
                               "all_features": torch.empty(0, 1, device=DEVICE)}
                PREPROCESSED_DATA_CACHE[cache_key] = cached_data
                return cached_data

            pil_image = Image.open(frame_path).convert("RGB")

            all_features = get_features_for_single_bbox(pil_image, all_boxes.cpu().numpy())

            if all_features is None:
                raise RuntimeError(f"为 {cache_key} 提取特征时返回了 None。")

            cached_data = {"all_boxes": all_boxes, "all_features": all_features}
            PREPROCESSED_DATA_CACHE[cache_key] = cached_data
            logging.info(f"为 {cache_key} 的预处理完成并已缓存。")

            cache_save_interval = settings.get('cache_save_interval_seconds', 30)
            if time.time() - _last_cache_save_time > cache_save_interval:
                threading.Thread(target=save_preprocessed_cache_to_disk).start()

            return cached_data


def get_features_for_specific_bboxes(video_uuid, frame_number, target_rects):
    try:
        processed_data = get_features_for_all_masks(video_uuid, frame_number)
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
        logging.warning(f"Skipping frame {frame_number} for specific feature extraction due to error: {e}")
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
            logging.warning(f"Skipping frame {frame_key} for on-the-fly prototype building due to error: {e}")

    if not all_prototypes:
        logging.error("Could not extract any valid on-the-fly prototypes after processing drawn samples.")
        return None

    return torch.cat(all_prototypes, dim=0)


def predict_from_one_shot(video_uuid, frame_number, positive_prompt_box):
    with AI_MODEL_LOCK:
        processed_data = get_features_for_all_masks(video_uuid, frame_number)
        all_boxes = processed_data.get("all_boxes")
        all_features = processed_data.get("all_features")

        if all_boxes is None or all_boxes.numel() == 0: return []

        prompt_rect = [positive_prompt_box['x1'], positive_prompt_box['y1'], positive_prompt_box['x2'],
                       positive_prompt_box['y2']]

        target_feature_tensor = get_features_for_specific_bboxes(video_uuid, frame_number, [prompt_rect])
        if target_feature_tensor is None or target_feature_tensor.numel() == 0:
            raise ValueError("Could not extract features for the provided positive prompt box.")

        target_feature = target_feature_tensor[0].unsqueeze(0)
        DEVICE = settings_manager.get_device()
        with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            sim_scores = F.cosine_similarity(target_feature, all_features, dim=1)
        settings = settings_manager.load_settings()
        nms_iou = settings.get('nms_iou_threshold', 0.7)
        kept_indices = nms(all_boxes, sim_scores, nms_iou)

        final_results = []
        final_scores_np = sim_scores.cpu().numpy()
        for i in kept_indices:
            box_coords = all_boxes[i].cpu().numpy().astype(int).tolist()
            final_results.append({"box": box_coords, "score": float(final_scores_np[i])})

        return final_results


def _calculate_similarity_scores(all_embeddings, positive_prototypes, negative_prototypes=None):
    settings = settings_manager.load_settings()
    score_temperature = settings.get('prototype_temperature', 0.07)
    DEVICE = settings_manager.get_device()

    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):

        sim_matrix = F.cosine_similarity(all_embeddings.unsqueeze(1), positive_prototypes.unsqueeze(0), dim=2)

        positive_scores_sim, _ = torch.max(sim_matrix, dim=1)

        if negative_prototypes is not None and len(negative_prototypes) > 0:
            if negative_prototypes.dim() > 1 and negative_prototypes.shape[0] > 1:
                 neg_sim_matrix = F.cosine_similarity(all_embeddings.unsqueeze(1), negative_prototypes.unsqueeze(0), dim=2)
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


def predict_with_prototypes(video_uuid, frame_number, positive_prototypes, negative_prototypes=None,
                            confidence_threshold=0.5):
    with AI_MODEL_LOCK:
        processed_data = get_features_for_all_masks(video_uuid, frame_number)
        all_boxes = processed_data.get("all_boxes")
        all_features = processed_data.get("all_features")

        if all_boxes is None or all_boxes.numel() == 0:
            return []

        final_scores = _calculate_similarity_scores(all_features, positive_prototypes, negative_prototypes)

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
    all_class_features_tensors = []
    sample_frames = database.get_all_frames_with_class(class_name)
    if not sample_frames:
        logging.warning(f"在数据库中找不到类别 '{class_name}' 的任何样本。")
        return None

    settings = settings_manager.load_settings()
    sample_limit = settings.get('prototype_sample_limit', 50)

    if len(sample_frames) > sample_limit:
        sample_frames = random.sample(sample_frames, sample_limit)

    grouped_boxes = defaultdict(list)
    for frame_data in sample_frames:
        frame_key = (frame_data['video_uuid'], frame_data['frame_number'])
        rects, labels, _ = convert_text_to_rects_and_labels(frame_data['bboxes_text'])
        target_rects_in_frame = [rect for i, rect in enumerate(rects) if labels[i] == class_name]
        if target_rects_in_frame:
            grouped_boxes[frame_key].extend(target_rects_in_frame)

    if not grouped_boxes:
        return None

    for (video_uuid, frame_number), all_rects_for_frame in grouped_boxes.items():
        try:
            pil_image = Image.open(file_storage.get_frame_path(video_uuid, frame_number)).convert("RGB")
            for i in range(0, len(all_rects_for_frame), 64):
                rect_chunk = all_rects_for_frame[i:i + 64]
                features = get_features_for_single_bbox(pil_image, rect_chunk)
                if features is not None and features.numel() > 0:
                    all_class_features_tensors.append(features)
        except Exception as e:
            logging.warning(f"为原型构建跳过帧 {video_uuid}/{frame_number} 时出错: {e}")

    if not all_class_features_tensors:
        logging.error(f"未能为类别 '{class_name}' 提取任何有效的特征向量。")
        return None

    all_features = torch.cat(all_class_features_tensors, dim=0)
    num_samples = all_features.shape[0]

    MIN_SAMPLES_FOR_CLUSTERING = 15
    if KMeans is None or num_samples < MIN_SAMPLES_FOR_CLUSTERING:
        logging.info(f"样本过少或 scikit-learn 未安装，为 '{class_name}' 创建单个平均原型。")
        return torch.mean(all_features, dim=0, keepdim=True)

    logging.info(f"正在为 '{class_name}' ({num_samples} 个样本) 运行聚类分析以发现子原型...")
    best_k = 1
    best_score = -1
    max_clusters = min(5, num_samples - 1)

    features_np = all_features.cpu().numpy()

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features_np)
        try:
            score = silhouette_score(features_np, labels)
            logging.info(f"  - 测试 k={k}, 轮廓系数(Silhouette Score): {score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
        except ValueError:
            logging.warning(f"  - k={k} 无法计算轮廓系数，跳过。")

    SILHOUETTE_THRESHOLD = 0.55
    if best_k > 1 and best_score > SILHOUETTE_THRESHOLD:
        logging.info(
            f"发现 {best_k} 个清晰的子类别 (得分: {best_score:.4f})。正在为 '{class_name}' 创建 {best_k} 个子原型。")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        kmeans.fit(features_np)
        prototypes_np = kmeans.cluster_centers_
        prototypes = torch.from_numpy(prototypes_np).to(all_features.device)
    else:
        logging.info(f"未发现足够清晰的子类别 (最高分: {best_score:.4f})。为 '{class_name}' 创建单个平均原型。")
        prototypes = torch.mean(all_features, dim=0, keepdim=True)

    return prototypes


def build_prototypes_for_class(class_name):
    if class_name in PROTOTYPE_CACHE:
        return PROTOTYPE_CACHE[class_name]

    class_lock = _get_class_lock(class_name)
    with class_lock:
        if class_name in PROTOTYPE_CACHE:
            return PROTOTYPE_CACHE[class_name]
        prototype_tensor = _calculate_prototype_from_db(class_name)

        if prototype_tensor is not None:
            if prototype_tensor.dim() == 1:
                prototype_tensor = prototype_tensor.unsqueeze(0)

            PROTOTYPE_CACHE[class_name] = prototype_tensor
            logging.info(f"类别 '{class_name}' 的原型构建完成并已缓存。Shape: {prototype_tensor.shape}")
            save_prototypes_to_disk()

        return prototype_tensor


def update_prototype_for_class(class_name):
    class_lock = _get_class_lock(class_name)
    with class_lock:
        logging.info(f"后台任务开始更新类别 '{class_name}' 的原型。")
        new_prototype = _calculate_prototype_from_db(class_name)

        if new_prototype is not None:
            PROTOTYPE_CACHE[class_name] = new_prototype
            logging.info(f"类别 '{class_name}' 的原型已在后台成功更新。")
            save_prototypes_to_disk()
        else:
            logging.error(f"后台更新原型失败: 无法为 '{class_name}' 计算新原型。")


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
        sam_model = sam_tasks.get_sam_model()
        if not sam_model:
            raise RuntimeError("SAM 模型不可用。")

        results = sam_model(frame_path, points=[point_coords], labels=[1], verbose=False)
        if not results or not results[0].boxes or results[0].boxes.xyxy.numel() == 0:
            return None, "SAM 未在指定点找到对象。"

        box_tensor = results[0].boxes.xyxy[0]
        bbox_coords = box_tensor.cpu().numpy()
        bbox_dict = {'x1': int(bbox_coords[0]), 'y1': int(bbox_coords[1]), 'x2': int(bbox_coords[2]),
                     'y2': int(bbox_coords[3])}

        feature_vector = get_features_for_specific_bboxes(video_uuid, frame_number, [bbox_coords])
        if feature_vector is None or feature_vector.numel() == 0:
            return None, "无法为 SAM 找到的物体提取特征。"

        prototype_library = get_all_prototypes()
        if not prototype_library:
            return {"bbox": bbox_dict, "suggestions": []}, None

        scores = []
        DEVICE = settings_manager.get_device()

        with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            for class_name, prototype in prototype_library.items():
                sim_matrix = F.cosine_similarity(feature_vector.unsqueeze(1), prototype.unsqueeze(0), dim=2)
                max_similarity = torch.max(sim_matrix)
                scores.append({"label": class_name, "score": round(max_similarity.item(), 4)})

        sorted_suggestions = sorted(scores, key=lambda x: x['score'], reverse=True)

        return {"bbox": bbox_dict, "suggestions": sorted_suggestions[:5]}, None
