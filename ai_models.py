import json
import logging
import os
import threading
from collections import defaultdict

import cv2
import numpy as np
import torch
from torchvision.ops import nms, box_iou

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

AI_MODEL_LOCK = threading.RLock()


def get_retrieval_text_for_class(class_name):
    try:
        prompt = database.get_class_sam3_prompt(class_name)
    except Exception as e:
        logging.warning(f"Failed to read sam3_prompt for class '{class_name}': {e}")
        prompt = None
    return prompt if prompt else class_name


def class_has_labeled_examples(class_name):
    try:
        sample_frames = database.get_all_frames_with_class(class_name)
        return bool(sample_frames)
    except Exception:
        return False


def postprocess_sam_results(results, nms_iou_threshold):
    """旧版 SAM1 结果后处理。SAM2/SAM3 已经内置置信度过滤/NMS，这里保留以防旧调用方还在用。"""
    DEVICE = settings_manager.get_device()
    if not results or not results[0].masks:
        return torch.empty(0, 4, device=DEVICE), torch.empty(0, 1, 1, device=DEVICE)
    all_boxes = results[0].boxes.xyxy.to(DEVICE)
    all_scores = results[0].boxes.conf.to(DEVICE)
    all_masks = results[0].masks.data.to(DEVICE)
    kept_indices = nms(all_boxes, all_scores, nms_iou_threshold)
    return all_boxes[kept_indices], all_masks[kept_indices]


def find_best_matching_masks_by_iou(reference_boxes_np, candidate_boxes_tensor):
    """给一组参考框，在候选框里按 IoU 找最佳匹配的下标。纯几何计算，和特征提取无关，原样保留。"""
    DEVICE = settings_manager.get_device()
    if len(reference_boxes_np) == 0 or len(candidate_boxes_tensor) == 0:
        return torch.tensor([], dtype=torch.long, device=DEVICE)
    reference_boxes_tensor = torch.tensor(reference_boxes_np, dtype=torch.float32, device=DEVICE)
    iou_matrix = box_iou(reference_boxes_tensor, candidate_boxes_tensor)
    return torch.argmax(iou_matrix, dim=1)


def _best_iou_match(target_box, candidate_results, min_iou=0.1):
    """
    在 sam3_query_frame() 的返回结果里，找和 target_box 重叠度最高的那一个。
    返回 (score, iou)；candidate_results 为空、或者最佳重叠度低于 min_iou（意味着
    "候选里其实没有一个真的和目标框对应"，只是矮子里拔将军选了个不相关的框）时，
    返回 (0.0, best_iou) —— 不能把一个跟目标框根本不重叠的候选的分数误当成目标框
    自己的匹配分数。
    """
    if not candidate_results:
        return 0.0, 0.0
    boxes = np.array([r["box"] for r in candidate_results], dtype=np.float32)
    scores = [r["score"] for r in candidate_results]
    cand = torch.from_numpy(boxes)
    tgt = torch.from_numpy(np.array([target_box], dtype=np.float32))
    ious = box_iou(tgt, cand)[0]
    best_idx = int(torch.argmax(ious).item())
    best_iou = float(ious[best_idx].item())
    if best_iou < min_iou:
        return 0.0, best_iou
    return float(scores[best_idx]), best_iou


def _calculate_region_color_hist(image_bgr, rect):
    """
    计算区域颜色直方图（HSV，中心裁剪 50% 避免背景/边缘干扰）。
    和 MobileNet 无关，是纯 OpenCV 的颜色特征，原样保留，一致性检查和智能选择的
    可选颜色校验都还在用它。
    """
    x1, y1, x2, y2 = map(int, rect)
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    bw, bh = x2 - x1, y2 - y1
    margin_x, margin_y = int(bw * 0.25), int(bh * 0.25)
    cx1, cy1, cx2, cy2 = x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y
    if (cx2 - cx1) < 2 or (cy2 - cy1) < 2:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    roi = image_bgr[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def _color_hist_distance(hist_a, hist_b):
    return float(np.sum(np.abs(hist_a - hist_b)))


# ==============================================================================
# 启动 / 缓存管理
# ==============================================================================

def startup_ai_models():
    """
    初始化 AI 模型：SAM2 (点选/追踪) + SAM3 (开放词汇/框样例检索)。
    MobileNet 特征提取器已移除，不再需要单独的"特征提取器"加载分支。
    """
    settings = settings_manager.load_settings()

    if settings.get('enable_sam_model', True):
        if sam_tasks:
            logging.info("正在检查 SAM2 环境 (根据设置已启用)...")
            try:
                sam_tasks.get_sam_model()
            except Exception as e:
                logging.error(f"SAM2 初始化检查失败: {e}")
        else:
            logging.warning("SAM 库未安装，即使已在设置中启用，相关功能也无法使用。")
    else:
        logging.warning("SAM 模型已在系统设置中被禁用，将跳过加载以节省资源。")

    if settings.get('enable_feature_extractor', True):
        if sam_tasks:
            logging.info("正在预热 SAM3 检索引擎 (智能选择/LAM/批量应用/一致性检查)...")
            try:
                sam_tasks._load_sam3_models(mode="image")
            except Exception as e:
                logging.error(f"SAM3 检索引擎初始化检查失败: {e}")
    else:
        logging.warning("SAM3 检索类功能 (enable_feature_extractor) 已在系统设置中被禁用。")

    if settings.get('enable_cls_model', True):
        logging.info("正在检查分类模型 (CLIP) 环境 (根据设置已启用)...")
        try:
            import clip_model
            clip_model.clip_manager.get_available_models()
        except Exception as e:
            logging.warning(f"CLIP 分类模型初始化检查失败: {e}")
    else:
        logging.warning("分类模型 (CLIP) 已在系统设置中被禁用，将跳过预加载。")

    if settings.get('enable_pose_model', True):
        logging.info("正在检查姿态估计模型 (Grounded Pose) 环境 (根据设置已启用)...")
        try:
            import gkdt_tasks
        except Exception as e:
            logging.warning(f"姿态估计模型初始化检查失败: {e}")
    else:
        logging.warning("姿态估计模型 (Grounded Pose) 已在系统设置中被禁用，将跳过预加载。")


def clear_retrieval_engine_cache():
    if sam_tasks:
        try:
            sam_tasks.clear_sam3_frame_state_cache()
        except Exception as e:
            logging.error(f"Failed to clear SAM3 frame cache: {e}")


def warm_frame_cache(video_uuid, frame_number):
    if sam_tasks is None:
        raise RuntimeError("SAM feature is not installed.")
    return sam_tasks.warm_frame_cache(video_uuid, frame_number)


def is_frame_cached(video_uuid, frame_number):
    if sam_tasks is None:
        return False
    return sam_tasks.is_frame_cached(video_uuid, frame_number)


def lam_predict(video_uuid, frame_number, point_coords):
    if sam_tasks is None:
        return None, "SAM feature is not installed."

    with AI_MODEL_LOCK:
        frame_path = file_storage.get_frame_path(video_uuid, frame_number)

        bbox_dict = sam_tasks.predict_box_from_point_ultralytics(frame_path, point_coords)
        if not bbox_dict:
            return None, "SAM failed to find object at this point."

        clicked_box = [bbox_dict['x1'], bbox_dict['y1'], bbox_dict['x2'], bbox_dict['y2']]

        all_labels = database.get_all_class_labels()
        if not all_labels:
            return {"bbox": bbox_dict, "suggestions": []}, None

        suggestions = []
        for class_name in all_labels:
            retrieval_text = get_retrieval_text_for_class(class_name)
            try:
                class_results = sam_tasks.sam3_query_frame(
                    video_uuid, frame_number, text_prompt=retrieval_text, confidence=0.05
                )
            except Exception as e:
                logging.warning(f"LAM: SAM3 query failed for class '{class_name}': {e}")
                continue

            score, iou = _best_iou_match(clicked_box, class_results)
            if iou > 0.3:
                suggestions.append({"label": class_name, "score": round(score, 4)})

        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return {"bbox": bbox_dict, "suggestions": suggestions[:5]}, None


def predict_from_one_shot(video_uuid, frame_number, positive_prompt_box, negative_prompt_boxes=None,
                           use_color=False):
    if sam_tasks is None:
        raise RuntimeError("SAM feature is not installed.")

    prompt_rect = [positive_prompt_box['x1'], positive_prompt_box['y1'],
                    positive_prompt_box['x2'], positive_prompt_box['y2']]
    negative_rects = [
        [b['x1'], b['y1'], b['x2'], b['y2']] for b in (negative_prompt_boxes or [])
    ]

    settings = settings_manager.load_settings()
    confidence = float(settings.get('default_preannotation_conf', 0.5))

    with AI_MODEL_LOCK:
        results = sam_tasks.sam3_query_frame(
            video_uuid, frame_number,
            positive_boxes=[prompt_rect],
            negative_boxes=negative_rects if negative_rects else None,
            confidence=confidence
        )

        if use_color and results:
            image_bgr = cv2.imread(file_storage.get_frame_path(video_uuid, frame_number))
            if image_bgr is not None:
                pos_hist = _calculate_region_color_hist(image_bgr, prompt_rect)
                if pos_hist is not None:
                    filtered = []
                    color_veto_threshold = float(settings.get('color_veto_threshold', 1.0))
                    for r in results:
                        cand_hist = _calculate_region_color_hist(image_bgr, r['box'])
                        if cand_hist is None or _color_hist_distance(cand_hist, pos_hist) < color_veto_threshold:
                            filtered.append(r)
                    results = filtered

    return results


def predict_by_class_text(video_uuid, frame_number, class_name, confidence_threshold=0.5):
    if sam_tasks is None:
        raise RuntimeError("SAM feature is not installed.")

    retrieval_text = get_retrieval_text_for_class(class_name)
    with AI_MODEL_LOCK:
        results = sam_tasks.sam3_query_frame(
            video_uuid, frame_number, text_prompt=retrieval_text, confidence=confidence_threshold
        )
    return results


def _extract_crop_color_hist(image_bgr, rect=None):
    """
    Extract 48-dim color histogram, quad spatial color means, texture, and aspect ratio vector for an image or ROI crop.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros(48, dtype=np.float32)

    if rect is not None:
        h_img, w_img = image_bgr.shape[:2]
        x1 = max(0, min(w_img - 1, int(rect[0])))
        y1 = max(0, min(h_img - 1, int(rect[1])))
        x2 = max(x1 + 1, min(w_img, int(rect[2])))
        y2 = max(y1 + 1, min(h_img, int(rect[3])))
        roi = image_bgr[y1:y2, x1:x2]
    else:
        roi = image_bgr

    if roi.size == 0:
        return np.zeros(48, dtype=np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    aspect_ratio = roi.shape[1] / max(1, roi.shape[0])

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)

    hist_h = hist_h.flatten()
    hist_s = hist_s.flatten()
    hist_v = hist_v.flatten()

    h, w = hsv.shape[:2]
    quads = [
        hsv[0:h//2, 0:w//2], hsv[0:h//2, w//2:w],
        hsv[h//2:h, 0:w//2], hsv[h//2:h, w//2:w]
    ]
    quad_means = []
    for q in quads:
        if q.size > 0:
            quad_means.append(np.mean(q, axis=(0, 1)) / [180.0, 255.0, 255.0])
        else:
            quad_means.append(np.zeros(3, dtype=np.float32))
    quad_feat = np.concatenate(quad_means).astype(np.float32)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.float32([np.mean(edges) / 255.0])
    mean_val = np.float32([np.mean(gray) / 255.0])
    std_val = np.float32([np.std(gray) / 255.0])
    ar_feat = np.float32([np.log10(aspect_ratio)])

    vec = np.concatenate([hist_h, hist_s, hist_v, quad_feat, edge_density, mean_val, std_val, ar_feat])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def check_dataset_consistency(dataset_uuid, enable_color_check=True):
    """
    Quality control and outlier detection for DET, SEG, and CLS datasets.
    
    Uses CLIP deep feature vectors + Category Prototype Centroid comparison + Strict Color Check.
    Returns: (outlier_image_indices: set[int], all_bboxes_info: list[dict], message: str)
    """
    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset:
        return None, None, "Dataset not found."

    video_uuids = json.loads(dataset.get('video_uuids') or '[]')
    video_info_cache = {vu: database.get_video_entity(vu) for vu in video_uuids}

    export_format = dataset.get('export_format', '')
    is_classification = (export_format in ['folder_classification', 'yolo_cls']) or any(
        video_info_cache.get(vu, {}).get('annotation_type') == 'classification' for vu in video_uuids
    )

    outlier_image_indices = set()
    all_bboxes_info = []

    use_clip = True
    try:
        import clip_model
    except Exception as e:
        logging.warning(f"CLIP model import failed: {e}")
        use_clip = False

    if is_classification:
        # === CLASSIFICATION (CLS) DATASET QUALITY SCAN ===
        all_frames = []
        for vu in video_uuids:
            frames = database.get_video_frames(vu)
            for f in frames:
                ann_json = (f.get('annotations_json') or '').strip()
                tags = []
                if ann_json:
                    try:
                        from annotation_model import AnnotationData
                        ann_data = AnnotationData.from_json(ann_json)
                        tags = ann_data.classifications or []
                    except Exception:
                        pass
                if not tags and (f.get('bboxes_text') or '').strip():
                    from app import extract_labels
                    tags = extract_labels(f['bboxes_text'])
                if tags:
                    all_frames.append({
                        'video_uuid': vu,
                        'frame_number': f['frame_number'],
                        'tags': tags
                    })

        if not all_frames:
            return set(), [], "No labeled classification images found in this dataset."

        items = []
        for i, f_item in enumerate(all_frames):
            img_path = file_storage.get_frame_path(f_item['video_uuid'], f_item['frame_number'])
            if not os.path.exists(img_path):
                continue
            img_bgr = cv2.imread(img_path)
            if img_bgr is None or img_bgr.size == 0:
                continue

            if use_clip:
                try:
                    sem_vec = clip_model.clip_manager.extract_image_feature_vector(img_bgr)
                except Exception:
                    sem_vec = _extract_crop_color_hist(img_bgr)
            else:
                sem_vec = _extract_crop_color_hist(img_bgr)

            sem_norm = np.linalg.norm(sem_vec)
            if sem_norm > 0:
                sem_vec = sem_vec / sem_norm

            if enable_color_check:
                color_vec = _extract_crop_color_hist(img_bgr)
                combined_vec = np.concatenate([sem_vec * 0.7, color_vec * 0.3])
            else:
                combined_vec = sem_vec

            c_norm = np.linalg.norm(combined_vec)
            if c_norm > 0:
                combined_vec = combined_vec / c_norm

            items.append({
                'index': i,
                'tags': f_item['tags'],
                'vector': combined_vec
            })

        class_vectors = defaultdict(list)
        for it in items:
            for tag in it['tags']:
                class_vectors[tag].append(it['vector'])

        class_centroids = {}
        for tag, vecs in class_vectors.items():
            mean_v = np.mean(vecs, axis=0)
            norm_v = np.linalg.norm(mean_v)
            class_centroids[tag] = mean_v / norm_v if norm_v > 0 else mean_v

        in_class_sims = defaultdict(list)
        for it in items:
            for tag in it['tags']:
                if tag in class_centroids:
                    sim = float(np.dot(it['vector'], class_centroids[tag]))
                    in_class_sims[tag].append(sim)

        class_stats = {}
        for tag, sims in in_class_sims.items():
            class_stats[tag] = {
                'mean': float(np.mean(sims)),
                'std': float(np.std(sims)) if len(sims) > 1 else 0.0
            }

        for it in items:
            idx = it['index']
            vec = it['vector']
            tags_set = set(it['tags'])
            is_anomaly = False

            for tag in tags_set:
                if tag not in class_centroids:
                    continue
                own_sim = float(np.dot(vec, class_centroids[tag]))
                stats = class_stats.get(tag, {'mean': 0.8, 'std': 0.1})

                thresh = max(0.50, stats['mean'] - 2.0 * stats['std'])
                if own_sim < thresh:
                    is_anomaly = True

                for other_tag, centroid in class_centroids.items():
                    if other_tag not in tags_set:
                        other_sim = float(np.dot(vec, centroid))
                        if other_sim > own_sim + 0.06 and other_sim > 0.65:
                            is_anomaly = True
                            break

            if is_anomaly:
                outlier_image_indices.add(idx)

    else:
        all_frames = []
        for vu in video_uuids:
            frames = database.get_video_frames(vu)
            for f in frames:
                has_bboxes = bool((f.get('bboxes_text') or '').strip())
                has_ann_json = bool((f.get('annotations_json') or '').strip())
                if has_bboxes or has_ann_json:
                    all_frames.append(dict(f))

        if not all_frames:
            return set(), [], "No labeled bounding boxes or pose keypoint instances found in this dataset."

        for i, frame in enumerate(all_frames):
            instances_list = []

            if (frame.get('bboxes_text') or '').strip():
                rects, labels, _ = convert_text_to_rects_and_labels(frame['bboxes_text'])
                for j, r in enumerate(rects):
                    instances_list.append({'rect': r, 'label': labels[j], 'kpts': None})

            if (frame.get('annotations_json') or '').strip():
                try:
                    from annotation_model import AnnotationData
                    ann_data = AnnotationData.from_json(frame['annotations_json'])
                    for obj in ann_data.objects:
                        r = None
                        if obj.bbox and len(obj.bbox) == 4:
                            r = [float(obj.bbox[0]), float(obj.bbox[1]), float(obj.bbox[2]), float(obj.bbox[3])]
                        elif obj.keypoints:
                            valid_kps = [k for k in obj.keypoints if (k.get('v', 2) > 0)]
                            if valid_kps:
                                min_x = min(float(k['x']) for k in valid_kps)
                                min_y = min(float(k['y']) for k in valid_kps)
                                max_x = max(float(k['x']) for k in valid_kps)
                                max_y = max(float(k['y']) for k in valid_kps)
                                r = [min_x, min_y, max_x, max_y]

                        if r:
                            instances_list.append({
                                'rect': r,
                                'label': obj.label,
                                'kpts': obj.keypoints if obj.type == 'keypoint' else None
                            })
                except Exception as ex:
                    logging.warning(f"Error parsing annotations_json for quality scan: {ex}")

            img_path = file_storage.get_frame_path(frame['video_uuid'], frame['frame_number'])
            if not os.path.exists(img_path):
                continue
            img_bgr = cv2.imread(img_path)
            if img_bgr is None or img_bgr.size == 0:
                continue

            h_img, w_img = img_bgr.shape[:2]
            for inst in instances_list:
                label = inst['label']
                rect = inst['rect']
                kpts = inst['kpts']

                x1 = max(0, min(w_img - 1, int(rect[0])))
                y1 = max(0, min(h_img - 1, int(rect[1])))
                x2 = max(x1 + 1, min(w_img, int(rect[2])))
                y2 = max(y1 + 1, min(h_img, int(rect[3])))

                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
                    continue

                if use_clip:
                    try:
                        sem_vec = clip_model.clip_manager.extract_image_feature_vector(crop)
                    except Exception:
                        sem_vec = _extract_crop_color_hist(crop)
                else:
                    sem_vec = _extract_crop_color_hist(crop)

                sem_norm = np.linalg.norm(sem_vec)
                if sem_norm > 0:
                    sem_vec = sem_vec / sem_norm

                if enable_color_check:
                    color_vec = _extract_crop_color_hist(crop)
                    combined_vec = np.concatenate([sem_vec * 0.7, color_vec * 0.3])
                else:
                    combined_vec = sem_vec

                c_norm = np.linalg.norm(combined_vec)
                if c_norm > 0:
                    combined_vec = combined_vec / c_norm

                # 姿态关键点几何拓扑异常检查 (Keypoint Topology Anomaly Check)
                topology_anomaly = False
                if kpts:
                    for kp in kpts:
                        kx, ky = float(kp.get('x', 0)), float(kp.get('y', 0))
                        kv = int(kp.get('v', 2))
                        # 检查关节点是否严重越界（超出图像边界或超出包围框）
                        if kv > 0 and (kx < -50 or kx > w_img + 50 or ky < -50 or ky > h_img + 50):
                            topology_anomaly = True
                            break

                all_bboxes_info.append({
                    'image_index': i,
                    'rect': rect,
                    'label': label,
                    'video_uuid': frame['video_uuid'],
                    'frame_number': frame['frame_number'],
                    'vector': combined_vec,
                    'topology_anomaly': topology_anomaly
                })

        if not all_bboxes_info:
            return set(), all_bboxes_info, "No valid object crops found in this dataset."

        class_vectors = defaultdict(list)
        for bbox in all_bboxes_info:
            class_vectors[bbox['label']].append(bbox['vector'])

        class_centroids = {}
        for label, vecs in class_vectors.items():
            mean_v = np.mean(vecs, axis=0)
            norm_v = np.linalg.norm(mean_v)
            class_centroids[label] = mean_v / norm_v if norm_v > 0 else mean_v

        in_class_sims = defaultdict(list)
        for bbox in all_bboxes_info:
            label = bbox['label']
            sim = float(np.dot(bbox['vector'], class_centroids[label]))
            in_class_sims[label].append(sim)

        class_stats = {
            lbl: {'mean': float(np.mean(sims)), 'std': float(np.std(sims)) if len(sims) > 1 else 0.0}
            for lbl, sims in in_class_sims.items()
        }

        for bbox in all_bboxes_info:
            label = bbox['label']
            vec = bbox['vector']
            idx = bbox['image_index']
            own_sim = float(np.dot(vec, class_centroids[label]))
            stats = class_stats.get(label, {'mean': 0.8, 'std': 0.1})

            is_outlier = False
            thresh = max(0.48, stats['mean'] - 2.0 * stats['std'])
            if own_sim < thresh:
                is_outlier = True

            if bbox.get('topology_anomaly'):
                is_outlier = True

            for other_label, centroid in class_centroids.items():
                if other_label != label:
                    other_sim = float(np.dot(vec, centroid))
                    if other_sim > own_sim + 0.05 and other_sim > 0.65:
                        is_outlier = True
                        break

            if is_outlier:
                outlier_image_indices.add(idx)

    count = len(outlier_image_indices)
    mode_str = "CLS" if is_classification else "DET/SEG/POSE"
    color_str = " (Semantic + Color)" if enable_color_check else " (Semantic)"
    if count == 0:
        message = f"AI {mode_str} quality scan completed{color_str}. No obvious labeling confusion issues found."
    else:
        message = f"AI {mode_str} quality scan complete{color_str}. Found {count} images with potential labeling anomalies."

    return outlier_image_indices, all_bboxes_info, message
