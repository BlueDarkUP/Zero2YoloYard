import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import nms, box_iou
from torch.cuda.amp import autocast
import numpy as np
import random
import database
import file_storage
import settings_manager
from bbox_writer import convert_text_to_rects_and_labels

try:
    import ultralytics_sam_tasks as sam_tasks
except ImportError:
    logging.warning("ultralytics_sam_tasks.py not found or failed to import. All SAM features will be disabled.")
    sam_tasks = None

models = {}
PREPROCESSED_DATA_CACHE = {}

_mobilenet_cache = {"model": None, "name": None}
SCORE_TEMPERATURE = 0.07  # Add this back for softmax scaling


def clear_feature_extractor_cache():
    global _mobilenet_cache
    logging.info("Clearing Feature Extractor (MobileNetV3) model cache due to setting change.")
    if _mobilenet_cache["model"] is not None and hasattr(_mobilenet_cache["model"], 'cpu'):
        _mobilenet_cache["model"].cpu()
        del _mobilenet_cache["model"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _mobilenet_cache = {"model": None, "name": None}
    if 'feature_extractor' in models:
        del models['feature_extractor']


def startup_ai_models():
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
            logging.info(f"Feature Extractor model '{target_model_name}' is already loaded.")
            _mobilenet_cache["model"].to(DEVICE)
            models['feature_extractor'] = _mobilenet_cache["model"]
            return

        if _mobilenet_cache["model"] is not None:
            clear_feature_extractor_cache()

        logging.info(f"正在加载 Feature Extractor 模型 '{target_model_name}' 至 {DEVICE}...")
        feature_extractor = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        feature_extractor.classifier = nn.Identity()
        feature_extractor.to(DEVICE).eval()

        _mobilenet_cache["model"] = feature_extractor
        _mobilenet_cache["name"] = target_model_name
        models['feature_extractor'] = feature_extractor
        models['feature_extractor_transforms'] = MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()

        logging.info(f"Feature Extractor 模型 '{target_model_name}' 加载完成。")

    except Exception as e:
        logging.error(f"加载 Feature Extractor 模型失败: {e}", exc_info=True)
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
    startup_ai_models()
    if 'feature_extractor' not in models:
        raise RuntimeError("Feature extractor model failed to load. Cannot perform feature extraction.")

    DEVICE = settings_manager.get_device()
    settings = settings_manager.load_settings()

    cache_key = f"{video_uuid}_{frame_number}"
    if cache_key in PREPROCESSED_DATA_CACHE:
        logging.info(f"Reusing cached preprocessed data for: {cache_key}")
        return PREPROCESSED_DATA_CACHE[cache_key]

    with torch.no_grad(), autocast(enabled=(DEVICE.type == 'cuda')):
        logging.info(f"Starting new preprocessing for {cache_key}...")
        frame_path = file_storage.get_frame_path(video_uuid, frame_number)
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame image not found at {frame_path}")

        sam_model = sam_tasks.get_sam_model()
        if not sam_model:
            raise RuntimeError("SAM model not loaded.")

        sam_conf = settings.get('sam_mask_confidence', 0.35)
        results = sam_model(frame_path, verbose=False, conf=sam_conf)
        nms_iou = settings.get('nms_iou_threshold', 0.7)
        all_boxes, all_masks = postprocess_sam_results(results, nms_iou_threshold=nms_iou)

        if len(all_masks) == 0:
            logging.warning(f"SAM found no objects in {cache_key} with current settings.")
            cached_data = {"all_boxes": torch.empty(0, 4, device=DEVICE),
                           "all_features": torch.empty(0, 1, device=DEVICE)}
            PREPROCESSED_DATA_CACHE[cache_key] = cached_data
            return cached_data

        pil_image = Image.open(frame_path).convert("RGB")
        transform = models['feature_extractor_transforms']
        preprocessed_crops = []

        for i in range(all_masks.shape[0]):
            mask = all_masks[i].cpu()
            y_indices, x_indices = torch.where(mask > 0)
            if len(y_indices) == 0: continue

            y_min, y_max = y_indices.min().item(), y_indices.max().item()
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            if x_min >= x_max or y_min >= y_max: continue

            cropped_pil = pil_image.crop((x_min, y_min, x_max, y_max))
            image_input = transform(cropped_pil).unsqueeze(0)
            preprocessed_crops.append(image_input)

        if not preprocessed_crops:
            raise RuntimeError("Could not crop any valid images from SAM masks.")

        batch_tensor = torch.cat(preprocessed_crops, dim=0).to(DEVICE)
        all_features = models['feature_extractor'](batch_tensor)

        cached_data = {"all_boxes": all_boxes, "all_features": all_features}
        PREPROCESSED_DATA_CACHE[cache_key] = cached_data
        logging.info(f"Preprocessing for {cache_key} complete and cached.")
        return cached_data


def get_features_for_specific_bboxes(video_uuid, frame_number, target_rects):
    try:
        processed_data = get_features_for_all_masks(video_uuid, frame_number)
        all_boxes = processed_data.get("all_boxes")
        all_features = processed_data.get("all_features")

        if all_boxes is None or all_boxes.numel() == 0 or all_features.numel() == 0:
            return torch.empty(0, all_features.shape[1], device=all_features.device)

        matching_indices = find_best_matching_masks_by_iou(np.array(target_rects), all_boxes)
        if matching_indices.numel() > 0:
            return all_features[matching_indices]
        else:
            return torch.empty(0, all_features.shape[1], device=all_features.device)

    except Exception as e:
        logging.warning(f"Skipping frame {frame_number} for specific feature extraction due to error: {e}",
                        exc_info=True)
        return None


def get_prototypes_for_class(class_name):
    sample_frames = database.get_all_frames_with_class(class_name)
    if not sample_frames:
        logging.warning(f"Database query returned no frames for class '{class_name}'.")
        return None

    all_prototypes = []
    if len(sample_frames) > 50:
        sample_frames = random.sample(sample_frames, 50)

    logging.info(f"Building prototypes for '{class_name}' from {len(sample_frames)} sample frames.")

    for frame_data in sample_frames:
        try:
            rects, labels, _ = convert_text_to_rects_and_labels(frame_data['bboxes_text'])
            target_rects = [np.array(rects[i]) for i, label in enumerate(labels) if label == class_name]
            if not target_rects: continue

            features = get_features_for_specific_bboxes(frame_data['video_uuid'], frame_data['frame_number'],
                                                        target_rects)
            if features is not None and features.numel() > 0:
                all_prototypes.append(features)

        except Exception as e:
            logging.warning(f"Skipping frame {frame_data['frame_number']} for prototype building due to error: {e}")

    if not all_prototypes:
        logging.error(f"Could not extract any valid prototypes for class '{class_name}' after processing samples.")
        return None

    return torch.cat(all_prototypes, dim=0)


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
    mean_positive_prototype = torch.mean(positive_prototypes, dim=0, keepdim=True)
    positive_scores_sim = F.cosine_similarity(all_embeddings, mean_positive_prototype)

    if negative_prototypes is not None and len(negative_prototypes) > 0:
        mean_negative_prototype = torch.mean(negative_prototypes, dim=0, keepdim=True)
        negative_scores_sim = F.cosine_similarity(all_embeddings, mean_negative_prototype)

        logits = torch.stack([negative_scores_sim, positive_scores_sim], dim=1)
        probabilities = F.softmax(logits / SCORE_TEMPERATURE, dim=1)
        final_scores = probabilities[:, 1]
    else:
        final_scores = torch.sigmoid(positive_scores_sim / SCORE_TEMPERATURE)

    return final_scores


def predict_with_prototypes(video_uuid, frame_number, positive_prototypes, negative_prototypes=None):
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