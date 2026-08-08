import json
import logging
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


# --- 工具函数 ---

def get_retrieval_text_for_class(class_name):
    """获取类别用于 SAM3 检索的文本，未配置 prompt 则回退到类别名。"""
    try:
        prompt = database.get_class_sam3_prompt(class_name)
    except Exception as e:
        logging.warning(f"Failed to read sam3_prompt for class '{class_name}': {e}")
        prompt = None
    return prompt if prompt else class_name


def class_has_labeled_examples(class_name):
    """判断数据库中该类别是否有标注样本。"""
    try:
        sample_frames = database.get_all_frames_with_class(class_name)
        return bool(sample_frames)
    except Exception:
        return False


def postprocess_sam_results(results, nms_iou_threshold):
    """SAM 结果 NMS 后处理。"""
    DEVICE = settings_manager.get_device()
    if not results or not results[0].masks:
        return torch.empty(0, 4, device=DEVICE), torch.empty(0, 1, 1, device=DEVICE)
    all_boxes = results[0].boxes.xyxy.to(DEVICE)
    all_scores = results[0].boxes.conf.to(DEVICE)
    all_masks = results[0].masks.data.to(DEVICE)
    kept_indices = nms(all_boxes, all_scores, nms_iou_threshold)
    return all_boxes[kept_indices], all_masks[kept_indices]


def find_best_matching_masks_by_iou(reference_boxes_np, candidate_boxes_tensor):
    """根据 IoU 计算参考框与候选框的最佳匹配。"""
    DEVICE = settings_manager.get_device()
    if len(reference_boxes_np) == 0 or len(candidate_boxes_tensor) == 0:
        return torch.tensor([], dtype=torch.long, device=DEVICE)
    reference_boxes_tensor = torch.tensor(reference_boxes_np, dtype=torch.float32, device=DEVICE)
    iou_matrix = box_iou(reference_boxes_tensor, candidate_boxes_tensor)
    return torch.argmax(iou_matrix, dim=1)


def _best_iou_match(target_box, candidate_results, min_iou=0.1):
    """匹配与 target_box IoU 最高的候选框，低于 min_iou 时返回 (0.0, best_iou)。"""
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
    """计算区域中心 50% 范围的 HSV 颜色直方图。"""
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


# --- 模型加载与缓存管理 ---

def startup_ai_models():
    """按设置初始化 SAM2 / SAM3 模型。"""
    settings = settings_manager.load_settings()

    if settings.get('enable_sam_model', True):
        if sam_tasks:
            logging.info("正在检查 SAM2 环境...")
            try:
                sam_tasks.get_sam_model()
            except Exception as e:
                logging.error(f"SAM2 初始化检查失败: {e}")
        else:
            logging.warning("SAM 库未安装。")
    else:
        logging.warning("SAM 模型已禁用。")

    if settings.get('enable_feature_extractor', True):
        if sam_tasks:
            logging.info("正在预热 SAM3 检索引擎...")
            try:
                sam_tasks._load_sam3_models(mode="image")
            except Exception as e:
                logging.error(f"SAM3 检索引擎初始化失败: {e}")
    else:
        logging.warning("SAM3 检索功能已禁用。")


def clear_retrieval_engine_cache():
    """配置变更时清空 SAM3 帧级状态缓存。"""
    if sam_tasks:
        try:
            sam_tasks.clear_sam3_frame_state_cache()
        except Exception as e:
            logging.error(f"清理 SAM3 帧缓存失败: {e}")


def warm_frame_cache(video_uuid, frame_number):
    """预热 SAM3 帧缓存。"""
    if sam_tasks is None:
        raise RuntimeError("SAM 功能未安装。")
    return sam_tasks.warm_frame_cache(video_uuid, frame_number)


def is_frame_cached(video_uuid, frame_number):
    if sam_tasks is None:
        return False
    return sam_tasks.is_frame_cached(video_uuid, frame_number)


# --- 业务功能接口 ---

def lam_predict(video_uuid, frame_number, point_coords):
    """点击建议标签 (LAM)：SAM2 点选目标框，结合 SAM3 检索各类别匹配度。"""
    if sam_tasks is None:
        return None, "SAM 功能未安装。"

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
    """单样例智能选择：基于 positive/negative 框 prompt 进行 SAM3 框样例检索。"""
    if sam_tasks is None:
        raise RuntimeError("SAM 功能未安装。")

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
                    COLOR_VETO_L1_THRESHOLD = 1.0
                    for r in results:
                        cand_hist = _calculate_region_color_hist(image_bgr, r['box'])
                        if cand_hist is None or _color_hist_distance(cand_hist, pos_hist) < COLOR_VETO_L1_THRESHOLD:
                            filtered.append(r)
                    results = filtered

    return results


def predict_by_class_text(video_uuid, frame_number, class_name, confidence_threshold=0.5):
    """类别文本检索：使用类别的 SAM3 检索文本在指定帧查询。"""
    if sam_tasks is None:
        raise RuntimeError("SAM 功能未安装。")

    retrieval_text = get_retrieval_text_for_class(class_name)
    with AI_MODEL_LOCK:
        results = sam_tasks.sam3_query_frame(
            video_uuid, frame_number, text_prompt=retrieval_text, confidence=confidence_threshold
        )
    return results


def check_dataset_consistency(dataset_uuid, enable_color_check=True, semantic_threshold=None):
    """数据集一致性检查：基于 SAM3 开放词汇置信度与颜色特征筛选标注异常。"""
    if sam_tasks is None:
        raise RuntimeError("SAM 功能未安装。")

    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset:
        return None, None, "Dataset not found."

    video_uuids = json.loads(dataset.get('video_uuids') or '[]')
    all_frames = [
        dict(frame) for vu in video_uuids for frame in database.get_video_frames(vu)
        if (frame.get('bboxes_text') or '').strip()
    ]

    all_bboxes_info = []
    frames_to_process = defaultdict(list)
    for i, frame in enumerate(all_frames):
        rects, labels, _ = convert_text_to_rects_and_labels(frame['bboxes_text'])
        for j, rect in enumerate(rects):
            global_idx = len(all_bboxes_info)
            all_bboxes_info.append({
                'image_index': i, 'rect': rect, 'label': labels[j],
                'video_uuid': frame['video_uuid'], 'frame_number': frame['frame_number'],
                'color_hist': None,
            })
            frames_to_process[f"{frame['video_uuid']};{frame['frame_number']}"].append((global_idx, rect, labels[j]))

    if not all_bboxes_info:
        return set(), all_bboxes_info, "No labeled boxes found in this dataset."

    settings = settings_manager.load_settings()
    color_confusion_factor = float(settings.get('color_confusion_factor', 2.0))
    if semantic_threshold is not None:
        semantic_low_threshold = float(semantic_threshold)
    else:
        semantic_low_threshold = float(settings.get('consistency_semantic_threshold', 0.05))
    confusion_margin = float(settings.get('consistency_confusion_margin', 0.15))

    color_prototype_hists = defaultdict(list)
    outlier_image_indices = set()

    if enable_color_check:
        for frame_key, rect_data in frames_to_process.items():
            video_uuid, frame_number_str = frame_key.split(';')
            image_path = file_storage.get_frame_path(video_uuid, int(frame_number_str))
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                continue
            for global_idx, rect, label in rect_data:
                hist = _calculate_region_color_hist(image_bgr, rect)
                all_bboxes_info[global_idx]['color_hist'] = hist
                if hist is not None:
                    color_prototype_hists[label].append(hist)

        color_prototype_mean = {
            label: np.mean(np.array(hists), axis=0)
            for label, hists in color_prototype_hists.items() if hists
        }
    else:
        color_prototype_mean = {}

    with AI_MODEL_LOCK:
        for frame_key, rect_data in frames_to_process.items():
            video_uuid, frame_number_str = frame_key.split(';')
            frame_number = int(frame_number_str)
            labels_in_frame = sorted({label for _, _, label in rect_data})

            per_class_results = {}
            for label in labels_in_frame:
                retrieval_text = get_retrieval_text_for_class(label)
                try:
                    per_class_results[label] = sam_tasks.sam3_query_frame(
                        video_uuid, frame_number, text_prompt=retrieval_text, confidence=0.05
                    )
                except Exception as e:
                    logging.warning(f"Consistency check: SAM3 query failed for '{label}' in {frame_key}: {e}")
                    per_class_results[label] = []

            for global_idx, rect, label in rect_data:
                is_outlier = False
                own_score, _ = _best_iou_match(rect, per_class_results.get(label, []))

                if own_score < semantic_low_threshold:
                    is_outlier = True
                    logging.info(
                        f"[Consistency] SEMANTIC outlier: '{label}' own_score={own_score:.2f}. "
                        f"image_index={all_bboxes_info[global_idx]['image_index']}"
                    )
                else:
                    for other_label in labels_in_frame:
                        if other_label == label:
                            continue
                        other_score, _ = _best_iou_match(rect, per_class_results.get(other_label, []))
                        if other_score > own_score + confusion_margin:
                            is_outlier = True
                            logging.info(
                                f"[Consistency] SEMANTIC outlier: '{label}' (own={own_score:.2f}) looks more "
                                f"like '{other_label}' (other={other_score:.2f}). "
                                f"image_index={all_bboxes_info[global_idx]['image_index']}"
                            )
                            break

                if is_outlier:
                    outlier_image_indices.add(all_bboxes_info[global_idx]['image_index'])
                    continue

                if enable_color_check:
                    hist = all_bboxes_info[global_idx]['color_hist']
                    if hist is None or label not in color_prototype_mean or len(color_prototype_mean) < 2:
                        continue
                    dist_to_own = _color_hist_distance(hist, color_prototype_mean[label])
                    min_dist_other, closest_other = float('inf'), None
                    for other_label, other_hist in color_prototype_mean.items():
                        if other_label == label:
                            continue
                        dist = _color_hist_distance(hist, other_hist)
                        if dist < min_dist_other:
                            min_dist_other, closest_other = dist, other_label
                    if closest_other and min_dist_other * color_confusion_factor < dist_to_own:
                        logging.info(
                            f"[Consistency] COLOR outlier: '{label}' color profile closer to '{closest_other}'. "
                            f"image_index={all_bboxes_info[global_idx]['image_index']}"
                        )
                        outlier_image_indices.add(all_bboxes_info[global_idx]['image_index'])

    keyword = "**category or color**" if enable_color_check else "**category**"
    count = len(outlier_image_indices)
    if count == 0:
        message = "AI review completed. No obvious labeling confusion issues were found."
    elif count == 1:
        message = f"AI review complete. Found {count} image with potential instances of {keyword} confusion."
    else:
        message = f"AI review complete. Found {count} images with potential instances of {keyword} confusion."

    return outlier_image_indices, all_bboxes_info, message
