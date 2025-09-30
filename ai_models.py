import logging
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
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

_dinov2_model_cache = {"model": None, "name": None}


def clear_dinov2_cache():
    global _dinov2_model_cache
    logging.info("Clearing DINOv2 model cache due to setting change.")
    if _dinov2_model_cache["model"] is not None and hasattr(_dinov2_model_cache["model"], 'cpu'):
        _dinov2_model_cache["model"].cpu()
        del _dinov2_model_cache["model"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _dinov2_model_cache = {"model": None, "name": None}
    if 'dinov2' in models:
        del models['dinov2']


DEFAULT_INTERPOLATION_SIZE = (100, 100)
DEFAULT_BATCH_SIZE = 16
DEFAULT_FEATURE_FUSION_WEIGHTS = torch.tensor([0.1, 0.2, 0.3, 0.4], device=settings_manager.get_device()).view(4, 1, 1,
                                                                                                               1)
SCORE_TEMPERATURE = 0.07


def startup_ai_models():
    global _dinov2_model_cache
    DEVICE = settings_manager.get_device()
    if sam_tasks:
        logging.info("正在检查 SAM 点选/跟踪模型...")
        sam_tasks.get_sam_model()
        logging.info("SAM 点选/跟踪模型检查完成。")

    try:
        settings = settings_manager.load_settings()
        target_dinov2_model_name = settings.get("dinov2_model_name", "dinov2_vits14")

        if _dinov2_model_cache["model"] is not None and _dinov2_model_cache["name"] == target_dinov2_model_name:
            logging.info(f"DINOv2 model '{target_dinov2_model_name}' is already loaded.")
            _dinov2_model_cache["model"].to(DEVICE)
            models['dinov2'] = _dinov2_model_cache["model"]
            return

        if _dinov2_model_cache["model"] is not None:
            clear_dinov2_cache()

        logging.info(f"正在加载 DINOv2 模型 '{target_dinov2_model_name}' 至 {DEVICE}...")
        dinov2_model = torch.hub.load('facebookresearch/dinov2', target_dinov2_model_name, trust_repo=True)
        dinov2_model.to(DEVICE).eval()

        _dinov2_model_cache["model"] = dinov2_model
        _dinov2_model_cache["name"] = target_dinov2_model_name
        models['dinov2'] = dinov2_model

        models['dinov2_features'] = []

        def hook(module, input, output):
            models['dinov2_features'].append(output)

        for i in range(-4, 0, 1):
            models['dinov2'].blocks[i].register_forward_hook(hook)

        models['dinov2_transforms'] = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        logging.info(f"DINOv2 模型 '{target_dinov2_model_name}' 及特征钩子加载完成。")

    except Exception as e:
        logging.error(f"加载 DINOv2 模型失败: {e}", exc_info=True)
        if 'dinov2' in models:
            del models['dinov2']
        _dinov2_model_cache = {"model": None, "name": None}


def postprocess_sam_results(results, nms_iou_threshold):
    DEVICE = settings_manager.get_device()
    if not results or not results[0].masks:
        return torch.empty(0, 4, device=DEVICE), torch.empty(0, 1, 1, device=DEVICE), torch.empty(0, device=DEVICE)
    all_boxes = results[0].boxes.xyxy.to(DEVICE)
    all_scores = results[0].boxes.conf.to(DEVICE)
    all_masks = results[0].masks.data.to(DEVICE)
    kept_indices = nms(all_boxes, all_scores, nms_iou_threshold)
    logging.info(f"[智能选择] NMS: 从 {len(all_boxes)} 个初始掩码中保留了 {len(kept_indices)} 个。")
    final_boxes = all_boxes[kept_indices]
    final_masks = all_masks[kept_indices]
    final_scores = all_scores[kept_indices]
    return final_boxes, final_masks, final_scores


def get_all_embeddings_batched(fused_feature_map, all_masks, batch_size, interpolation_size):
    DEVICE = settings_manager.get_device()
    if len(all_masks) == 0:
        return torch.empty(0, fused_feature_map.shape[1], device=DEVICE)

    with torch.no_grad():
        upsampled_features = F.interpolate(fused_feature_map, size=interpolation_size, mode='bilinear',
                                           align_corners=False).squeeze(0)
        attention_map = torch.norm(upsampled_features, p=2, dim=0)
        all_embeddings_list = []
        for i in range(0, len(all_masks), batch_size):
            masks_batch = all_masks[i:i + batch_size]
            downsampled_masks = F.interpolate(masks_batch.unsqueeze(1).float(), size=interpolation_size,
                                              mode='bilinear', align_corners=False).to(DEVICE)
            final_weights = downsampled_masks * attention_map
            weight_sums = torch.clamp(final_weights.sum(dim=(-1, -2)), min=1e-6)
            weighted_features = upsampled_features * final_weights
            summed_embeddings = weighted_features.sum(dim=(-1, -2))
            avg_embeddings = summed_embeddings / weight_sums
            final_embeddings_batch = F.normalize(avg_embeddings, p=2, dim=-1)
            all_embeddings_list.append(final_embeddings_batch)

        return torch.cat(all_embeddings_list, dim=0)


def find_best_matching_masks_by_iou(reference_boxes_np, candidate_boxes_tensor):
    DEVICE = settings_manager.get_device()
    if len(reference_boxes_np) == 0 or len(candidate_boxes_tensor) == 0:
        return torch.tensor([], dtype=torch.long, device=DEVICE)
    reference_boxes_tensor = torch.tensor(reference_boxes_np, dtype=torch.float32, device=DEVICE)
    iou_matrix = box_iou(reference_boxes_tensor, candidate_boxes_tensor)
    best_match_indices = torch.argmax(iou_matrix, dim=1)
    return best_match_indices


def _get_hyperparams_from_request(data):
    DEVICE = settings_manager.get_device()
    hp = data.get('hyperparameters', {})

    if not hp:
        settings = settings_manager.load_settings()
        defaults = settings.get('smart_select_defaults', {})
        weights_list = defaults.get('weights', [0.1, 0.2, 0.3, 0.4])
        batch_size = defaults.get('batch_size', DEFAULT_BATCH_SIZE)
        interp_size_val = defaults.get('interp_size', 100)
    else:
        weights_list = hp.get('weights', [0.1, 0.2, 0.3, 0.4])
        batch_size = hp.get('batch_size', DEFAULT_BATCH_SIZE)
        interp_size_val = hp.get('interpolation_size', 100)

    fusion_weights = torch.tensor(weights_list, device=DEVICE).view(4, 1, 1, 1)
    interpolation_size = (interp_size_val, interp_size_val)
    return fusion_weights, batch_size, interpolation_size


def _preprocess_and_get_embeddings(video_uuid, frame_number, hyperparameters=None):

    startup_ai_models()
    if 'dinov2' not in models:
        raise RuntimeError("DINOv2 model failed to load. Cannot perform feature extraction.")

    DEVICE = settings_manager.get_device()
    settings = settings_manager.load_settings()

    cache_key = f"{video_uuid}_{frame_number}"
    if cache_key in PREPROCESSED_DATA_CACHE:
        logging.info(f"Reusing cached preprocessed data for: {cache_key}")
        return PREPROCESSED_DATA_CACHE[cache_key]

    fusion_weights, batch_size, interpolation_size = _get_hyperparams_from_request({'hyperparameters': hyperparameters})

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
        all_boxes, all_masks, _ = postprocess_sam_results(results, nms_iou_threshold=nms_iou)

        if len(all_masks) == 0:
            logging.warning(f"SAM found no objects in {cache_key} with current settings.")
            PREPROCESSED_DATA_CACHE[cache_key] = {"all_boxes": torch.empty(0, 4, device=DEVICE),
                                                  "all_masks": torch.empty(0, 1, 1, device=DEVICE),
                                                  "all_embeddings": torch.empty(0, 1, device=DEVICE)}
            return PREPROCESSED_DATA_CACHE[cache_key]

        image = Image.open(frame_path).convert("RGB")
        processed_image = models['dinov2_transforms'](image).unsqueeze(0).to(DEVICE)

        models['dinov2_features'].clear()
        _ = models['dinov2'](processed_image)

        raw_feature_maps_list = [fmap[:, 1:, :] for fmap in models['dinov2_features']]
        patch_size = models['dinov2'].patch_embed.patch_size[0]
        h_patches = w_patches = 224 // patch_size

        reshaped_weights = fusion_weights.view(4, 1, 1)
        fused_patch_tokens_list = []
        for i in range(len(raw_feature_maps_list)):
            feature_map_transposed = raw_feature_maps_list[i].transpose(1, 2)
            feature_map_reshaped = feature_map_transposed.reshape(1, feature_map_transposed.shape[1], h_patches,
                                                                  w_patches)
            interpolated_feature = F.interpolate(feature_map_reshaped, size=(h_patches, w_patches), mode='bilinear',
                                                 align_corners=False)
            fused_patch_tokens_list.append(interpolated_feature.flatten(2).transpose(1, 2) * reshaped_weights[i])

        fused_patch_tokens = torch.sum(torch.stack(fused_patch_tokens_list), dim=0)
        num_channels = fused_patch_tokens.shape[-1]
        feature_map = fused_patch_tokens.reshape(1, h_patches, w_patches, num_channels).permute(0, 3, 1, 2)

        all_embeddings = get_all_embeddings_batched(feature_map, all_masks, batch_size, interpolation_size)

        PREPROCESSED_DATA_CACHE[cache_key] = {"all_boxes": all_boxes, "all_masks": all_masks,
                                              "all_embeddings": all_embeddings}
        logging.info(f"Preprocessing for {cache_key} complete and cached.")
        return PREPROCESSED_DATA_CACHE[cache_key]


def get_prototypes_for_class(class_name, hyperparameters=None):
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

            frame_uuid = frame_data['video_uuid']
            frame_num = frame_data['frame_number']

            processed_data = _preprocess_and_get_embeddings(frame_uuid, frame_num, hyperparameters)

            all_boxes = processed_data.get("all_boxes")
            all_embeddings = processed_data.get("all_embeddings")

            if all_boxes is None or all_boxes.numel() == 0:
                continue

            matching_indices = find_best_matching_masks_by_iou(np.array(target_rects), all_boxes)
            if matching_indices.numel() > 0:
                embeddings = all_embeddings[matching_indices]
                all_prototypes.append(embeddings)

        except Exception as e:
            logging.warning(f"Skipping frame {frame_data['frame_number']} for prototype building due to error: {e}",
                            exc_info=True)

    if not all_prototypes:
        logging.error(f"Could not extract any valid prototypes for class '{class_name}' after processing samples.")
        return None

    return torch.cat(all_prototypes, dim=0)


def get_prototypes_from_drawn_boxes(drawn_samples_data, hyperparameters=None):
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
            processed_data = _preprocess_and_get_embeddings(video_uuid, frame_num, hyperparameters)

            all_boxes = processed_data.get("all_boxes")
            all_embeddings = processed_data.get("all_embeddings")

            if all_boxes is None or all_boxes.numel() == 0:
                continue
            matching_indices = find_best_matching_masks_by_iou(np.array(target_rects), all_boxes)
            if matching_indices.numel() > 0:
                embeddings = all_embeddings[matching_indices]
                all_prototypes.append(embeddings)

        except Exception as e:
            logging.warning(f"Skipping frame {frame_key} for on-the-fly prototype building due to error: {e}",
                            exc_info=True)

    if not all_prototypes:
        logging.error("Could not extract any valid on-the-fly prototypes after processing drawn samples.")
        return None

    return torch.cat(all_prototypes, dim=0)


def predict_with_prototypes(video_uuid, frame_number, positive_prototypes, has_negative_prototypes=False,
                            negative_prototypes=None, hyperparameters=None):
    processed_data = _preprocess_and_get_embeddings(video_uuid, frame_number, hyperparameters)
    all_boxes = processed_data.get("all_boxes")
    all_embeddings = processed_data.get("all_embeddings")

    if all_boxes is None or all_boxes.numel() == 0:
        return []

    final_scores = _calculate_similarity_scores(all_embeddings, positive_prototypes,
                                                negative_prototypes if has_negative_prototypes else None)

    settings = settings_manager.load_settings()
    nms_iou = settings.get('nms_iou_threshold', 0.7)
    kept_indices = nms(all_boxes, final_scores, nms_iou)

    final_results = []
    final_scores_np = final_scores.cpu().numpy()
    for i in kept_indices:
        box_coords = all_boxes[i].cpu().numpy().astype(int).tolist()
        final_results.append({"box": box_coords, "score": float(final_scores_np[i])})

    return final_results


def _calculate_similarity_scores(all_embeddings, positive_prototypes, negative_prototypes=None):
    K_pos = min(3, len(positive_prototypes))
    positive_sim_matrix = F.cosine_similarity(all_embeddings.unsqueeze(1), positive_prototypes.unsqueeze(0), dim=-1)
    top_k_positive_sim, _ = torch.topk(positive_sim_matrix, K_pos, dim=1)
    positive_scores_sim = torch.mean(top_k_positive_sim, dim=1)

    if negative_prototypes is not None and len(negative_prototypes) > 0:
        K_neg = min(3, len(negative_prototypes))
        negative_sim_matrix = F.cosine_similarity(all_embeddings.unsqueeze(1), negative_prototypes.unsqueeze(0), dim=-1)
        top_k_negative_sim, _ = torch.topk(negative_sim_matrix, K_neg, dim=1)
        negative_scores_sim = torch.mean(top_k_negative_sim, dim=1)
        logits = torch.stack([negative_scores_sim, positive_scores_sim], dim=1)
        probabilities = F.softmax(logits / SCORE_TEMPERATURE, dim=1)
        final_scores = probabilities[:, 1]
    else:
        final_scores = torch.sigmoid(positive_scores_sim / SCORE_TEMPERATURE)

    return final_scores