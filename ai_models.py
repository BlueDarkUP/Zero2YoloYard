import logging
import os
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.ops import nms, box_iou
from torch.amp import autocast
import numpy as np
import random
import time
import threading
import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize
import config
import database
import file_storage
import settings_manager
from bbox_writer import convert_text_to_rects_and_labels
from torchvision.transforms.functional import crop
import onnxruntime as ort

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


_mobilenet_cache = {"model": None, "name": None}


def get_features_for_single_bbox(pil_image, target_rects):
    if 'feature_extractor' not in models:
        raise RuntimeError("Feature extractor model failed to load.")
    if not target_rects:
        return None

    DEVICE = settings_manager.get_device()

    img_tensor = to_tensor(pil_image).to(DEVICE)

    boxes_for_crop = torch.tensor(target_rects, dtype=torch.float32, device=DEVICE)
    box_indices = torch.zeros(boxes_for_crop.size(0), 1, device=DEVICE)
    boxes_for_roi = torch.cat([box_indices, boxes_for_crop], dim=1)

    OUTPUT_SIZE = (224, 224)
    batch_of_crops = torchvision.ops.roi_align(
        img_tensor.unsqueeze(0),
        boxes_for_roi,
        output_size=OUTPUT_SIZE,
        spatial_scale=1.0,
        aligned=True
    )

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    batch_tensor = normalize(batch_of_crops, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    ort_session = models['feature_extractor']
    input_name = models['feature_extractor_input_name']
    output_name = models['feature_extractor_output_name']

    numpy_input = batch_tensor.cpu().numpy()
    ort_outputs = ort_session.run([output_name], {input_name: numpy_input})
    features = torch.from_numpy(ort_outputs[0]).to(DEVICE)

    return features

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
    global _mobilenet_cache
    DEVICE = settings_manager.get_device()

    if sam_tasks:
        logging.info("正在检查 SAM 点选/跟踪模型...")
        sam_tasks.get_sam_model()
        logging.info("SAM 点选/跟踪模型检查完成。")

    try:
        settings = settings_manager.load_settings()
        target_model_name = settings.get("feature_extractor_model_name", "mobilenet_v3_large")

        if _mobilenet_cache["model"] is not None and _mobilenet_cache["name"] == target_model_name:
            models['feature_extractor'] = _mobilenet_cache["model"]
            return

        if _mobilenet_cache["model"] is not None:
            clear_feature_extractor_cache()

        logging.info(f"正在加载 ONNX Feature Extractor 模型 '{target_model_name}'...")

        onnx_paths = {
            "mobilenet_v3_large": config.MOBILENET_LARGE_ONNX,
            "mobilenet_v3_small": config.MOBILENET_SMALL_ONNX
        }
        target_onnx_path = onnx_paths.get(target_model_name)

        if not target_onnx_path or not os.path.exists(target_onnx_path):
            logging.warning(
                f"ONNX 模型 '{target_model_name}' 未找到于 '{target_onnx_path}', 将使用默认的 mobilenet_v3_large。")
            target_onnx_path = config.MOBILENET_LARGE_ONNX
            target_model_name = "mobilenet_v3_large"

        if target_model_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1

        sess_options = ort.SessionOptions()

        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_provider_options = {'cudnn_conv_algo_search': 'DEFAULT'}

        providers = (
            [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
            if DEVICE.type == 'cuda'
            else ['CPUExecutionProvider']
        )

        logging.info(f"使用 ONNX Runtime Providers: {providers}")

        ort_session = ort.InferenceSession(target_onnx_path, sess_options, providers=providers)

        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        logging.info(f"ONNX 模型输入名: '{input_name}', 输出名: '{output_name}'")

        _mobilenet_cache["model"] = ort_session
        _mobilenet_cache["name"] = target_model_name
        _mobilenet_cache["input_name"] = input_name
        _mobilenet_cache["output_name"] = output_name

        models['feature_extractor'] = ort_session
        models['feature_extractor_transforms'] = weights.transforms()
        models['feature_extractor_input_name'] = input_name
        models['feature_extractor_output_name'] = output_name

        logging.info(f"ONNX Feature Extractor 模型 '{target_model_name}' 加载完成。")

    except Exception as e:
        logging.error(f"加载 ONNX Feature Extractor 模型失败: {e}", exc_info=True)
        if 'feature_extractor' in models:
            del models['feature_extractor']
        _mobilenet_cache = {"model": None, "name": None}


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
    if 'feature_extractor' not in models:
        raise RuntimeError("Feature extractor model failed to load. Cannot perform feature extraction.")

    DEVICE = settings_manager.get_device()
    cache_key = f"{video_uuid}_{frame_number}"
    if cache_key in PREPROCESSED_DATA_CACHE:
        return PREPROCESSED_DATA_CACHE[cache_key]

    with AI_MODEL_LOCK:
        if cache_key in PREPROCESSED_DATA_CACHE:
            return PREPROCESSED_DATA_CACHE[cache_key]

        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            logging.info(f"Starting new preprocessing for {cache_key}...")
            frame_path = file_storage.get_frame_path(video_uuid, frame_number)
            if not os.path.exists(frame_path):
                raise FileNotFoundError(f"Frame image not found at {frame_path}")

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
            img_tensor = to_tensor(pil_image).to(DEVICE)

            box_indices = torch.zeros(all_boxes.size(0), 1, device=DEVICE)
            boxes_for_crop = torch.cat([box_indices, all_boxes], dim=1)

            OUTPUT_SIZE = (224, 224)
            batch_of_crops = torchvision.ops.roi_align(
                img_tensor.unsqueeze(0),
                boxes_for_crop,
                output_size=OUTPUT_SIZE,
                spatial_scale=1.0,
                aligned=True
            )


            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]

            batch_tensor = normalize(batch_of_crops, mean=IMAGENET_MEAN, std=IMAGENET_STD)

            ort_session = models['feature_extractor']
            input_name = models['feature_extractor_input_name']
            output_name = models['feature_extractor_output_name']

            numpy_input = batch_tensor.cpu().numpy()
            ort_outputs = ort_session.run([output_name], {input_name: numpy_input})
            all_features = torch.from_numpy(ort_outputs[0]).to(DEVICE)

            cached_data = {"all_boxes": all_boxes, "all_features": all_features}
            PREPROCESSED_DATA_CACHE[cache_key] = cached_data
            logging.info(f"Preprocessing for {cache_key} complete and cached.")

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
        mean_positive_prototype = torch.mean(positive_prototypes, dim=0, keepdim=True)
        positive_scores_sim = F.cosine_similarity(all_embeddings, mean_positive_prototype)

        if negative_prototypes is not None and len(negative_prototypes) > 0:
            mean_negative_prototype = torch.mean(negative_prototypes, dim=0, keepdim=True)
            negative_scores_sim = F.cosine_similarity(all_embeddings, mean_negative_prototype)
            logits = torch.stack([negative_scores_sim, positive_scores_sim], dim=1)
            probabilities = F.softmax(logits / score_temperature, dim=1)
            final_scores = probabilities[:, 1]
        else:
            final_scores = torch.sigmoid(positive_scores_sim / score_temperature)

    return final_scores


def predict_with_prototypes(video_uuid, frame_number, positive_prototypes, negative_prototypes=None):
    with AI_MODEL_LOCK:
        processed_data = get_features_for_all_masks(video_uuid, frame_number)
        all_boxes = processed_data.get("all_boxes")
        all_features = processed_data.get("all_features")

        if all_boxes is None or all_boxes.numel() == 0:
            return []

        final_scores = _calculate_similarity_scores(all_features, positive_prototypes, negative_prototypes)

        settings = settings_manager.load_settings()
        nms_iou = settings.get('nms_iou_threshold', 0.7)
        kept_indices = nms(all_boxes, final_scores, nms_iou)

        final_results = []
        final_scores_np = final_scores.cpu().numpy()
        for i in kept_indices:
            box_coords = all_boxes[i].cpu().numpy().astype(int).tolist()
            final_results.append({"box": box_coords, "score": float(final_scores_np[i])})

        return final_results


def _calculate_prototype_from_db(class_name):
    sample_frames = database.get_all_frames_with_class(class_name)
    if not sample_frames:
        logging.warning(f"在数据库中找不到类别 '{class_name}' 的任何样本。")
        return None

    settings = settings_manager.load_settings()
    sample_limit = settings.get('prototype_sample_limit', 50)

    if len(sample_frames) > sample_limit:
        sample_frames = random.sample(sample_frames, sample_limit)

    logging.info(f"正在从 {len(sample_frames)} 个样本为 '{class_name}' 计算新原型...")

    all_class_features = []
    for frame_data in sample_frames:
        try:
            frame_path = file_storage.get_frame_path(frame_data['video_uuid'], frame_data['frame_number'])
            if not os.path.exists(frame_path):
                continue

            pil_image = Image.open(frame_path).convert("RGB")

            rects, labels, _ = convert_text_to_rects_and_labels(frame_data['bboxes_text'])
            target_rects = [rects[i] for i, label in enumerate(labels) if label == class_name]
            if not target_rects:
                continue

            features = get_features_for_single_bbox(pil_image, target_rects)

            if features is not None and features.numel() > 0:
                all_class_features.append(features)
        except Exception as e:
            logging.warning(f"为原型构建跳过帧 {frame_data['frame_number']} 时出错: {e}")

    if not all_class_features:
        logging.error(f"未能为类别 '{class_name}' 提取任何有效的特征向量。")
        return None

    return torch.mean(torch.cat(all_class_features, dim=0), dim=0)


def build_prototypes_for_class(class_name):
    if class_name in PROTOTYPE_CACHE:
        return PROTOTYPE_CACHE[class_name]

    class_lock = _get_class_lock(class_name)
    with class_lock:
        if class_name in PROTOTYPE_CACHE:
            return PROTOTYPE_CACHE[class_name]

        prototype_tensor = _calculate_prototype_from_db(class_name)

        if prototype_tensor is not None:
            PROTOTYPE_CACHE[class_name] = prototype_tensor
            logging.info(f"类别 '{class_name}' 的原型构建完成并已缓存。")
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
                similarity = F.cosine_similarity(feature_vector, prototype.unsqueeze(0))
                scores.append({"label": class_name, "score": round(similarity.item(), 4)})

        sorted_suggestions = sorted(scores, key=lambda x: x['score'], reverse=True)

        return {"bbox": bbox_dict, "suggestions": sorted_suggestions[:5]}, None
