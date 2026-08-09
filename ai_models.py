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

# ==============================================================================
# 重构说明 (MobileNet -> SAM3 迁移)
# ==============================================================================
# 本文件原来还负责加载/运行一个独立的 MobileNetV3 (ImageNet 预训练、去分类头) 作为
# 通用语义特征提取器，支撑"类别原型库 (KMeans 聚类) + 余弦相似度"这一整套范式，用来
# 实现四个功能: LAM(点击建议标签)、智能选择(单样例/类别库)、批量应用到整段视频、
# 数据集一致性检查。
#
# 这套范式已经整体替换为基于 SAM3 开放词汇/框样例检索的新实现，核心原语是
# ultralytics_sam_tasks.sam3_query_frame()（同一帧的 backbone 只算一次，可以用文本
# 或框样例反复查询）。MobileNet 模型本身、原型库 (PROTOTYPE_CACHE)、按帧缓存的
# MobileNet 特征 (PREPROCESSED_DATA_CACHE)、KMeans 聚类、余弦相似度计算、磁盘持久化
# (prototype_library.pt / preprocessed_cache.pt) 等代码均已删除，不再需要 torchvision
# 的 MobileNet 权重和 scikit-learn 依赖。
#
# 类别现在通过 database.class_labels.sam3_prompt 字段维护各自的 SAM3 检索文本（见
# get_retrieval_text_for_class），不再需要"用已标注样本训练/构建原型"这个步骤。
# ==============================================================================

AI_MODEL_LOCK = threading.RLock()


# ==============================================================================
# 通用工具函数
# ==============================================================================

def get_retrieval_text_for_class(class_name):
    """
    取某个类别用于 SAM3 检索的文本：优先用用户在类别管理里填写的 sam3_prompt，
    没填就回退用类别名本身（保证没配置描述之前功能不是硬失败）。
    """
    try:
        prompt = database.get_class_sam3_prompt(class_name)
    except Exception as e:
        logging.warning(f"Failed to read sam3_prompt for class '{class_name}': {e}")
        prompt = None
    return prompt if prompt else class_name


def class_has_labeled_examples(class_name):
    """
    对应旧代码里 build_prototypes_for_class(...) is not None 的判断。
    新架构不需要真的"训练/构建原型"，这里只是给前端一个软提示用：
    这个类别在数据库里是否已经有标注样本（用于提示"还没有样本，建议先手动标几个再用批量功能"），
    不再是功能是否可用的硬性前置条件。
    """
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


def clear_retrieval_engine_cache():
    """
    对应旧的 clear_feature_extractor_cache()。设置里和 SAM3 检索相关的配置
    （设备、checkpoint 等）变更时调用，清空帧级 backbone 缓存，避免复用旧状态。
    """
    if sam_tasks:
        try:
            sam_tasks.clear_sam3_frame_state_cache()
        except Exception as e:
            logging.error(f"清理 SAM3 帧缓存失败: {e}")


def warm_frame_cache(video_uuid, frame_number):
    """给 /interactive_segment/preprocess、后台预处理路由用：提前跑掉最贵的 backbone 前向。"""
    if sam_tasks is None:
        raise RuntimeError("SAM 功能未安装。")
    return sam_tasks.warm_frame_cache(video_uuid, frame_number)


def is_frame_cached(video_uuid, frame_number):
    if sam_tasks is None:
        return False
    return sam_tasks.is_frame_cached(video_uuid, frame_number)


# ==============================================================================
# 功能 1: 点击建议标签 (LAM, Label Assignment Matching)
# ==============================================================================

def lam_predict(video_uuid, frame_number, point_coords):
    """
    第一步 (SAM2 点选出精确框) 和 MobileNet 无关，原样保留 —— 响应快、边缘精度高，
    SAM3 暂时替代不了这种亚秒级单点点选体验。
    第二步 (给这个框猜标签) 原来是 "MobileNet 特征 + 全部类别原型做余弦相似度"，
    现在改成: 对当前项目里每个类别都用 sam3_query_frame 在同一帧上查一次(同一帧
    backbone 只算一次，靠帧级缓存复用)，用 IoU 把 SAM3 各类别的检测结果和 SAM2
    给出的精确框做匹配，按匹配到的置信度对类别排序，取 Top-5。

    已知取舍: 类别数量较多时，这一步等价于同一帧要跑 N 次 SAM3 文本查询检测头
    (N=类别数，backbone 部分是共享的)。类别很多、或者在纯 CPU 环境下，单次点击的
    延迟会比旧的"点一下就出结果"明显变高，这是新架构相对旧架构的一个已知取舍，
    不是 bug。如果延迟成为问题，后续可以考虑把候选类别限制在"最近使用过的类别"
    等子集里，而不是每次都跑全量词表。
    """
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
                # 阈值刻意放得很低: 这里要问的是"这个具体位置有多像类别 X"，而不是
                # "SAM3 会不会主动把它检出来"——点击的位置已经确定有一个目标了，我们
                # 只是想知道各个类别的相对匹配程度，所以要拿到低置信度下的原始分数
                # 用于排序比较，而不是提前被阈值过滤掉。
                class_results = sam_tasks.sam3_query_frame(
                    video_uuid, frame_number, text_prompt=retrieval_text, confidence=0.05
                )
            except Exception as e:
                logging.warning(f"LAM: SAM3 query failed for class '{class_name}': {e}")
                continue

            score, iou = _best_iou_match(clicked_box, class_results)
            if iou > 0.3:  # 只有和点击框有实质几何重叠时，这个分数才有意义
                suggestions.append({"label": class_name, "score": round(score, 4)})

        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return {"bbox": bbox_dict, "suggestions": suggestions[:5]}, None


# ==============================================================================
# 功能 2: 智能选择 · 单样例 (One-shot Smart Select)
# ==============================================================================

def predict_from_one_shot(video_uuid, frame_number, positive_prompt_box, negative_prompt_boxes=None,
                           use_color=False):
    """
    旧实现: 全帧"分割一切"生成候选框 -> MobileNet 逐框提特征 -> 与正例框特征算余弦
           相似度 -> NMS。
    新实现: 直接把用户画的正例框(可选再加几个负例框)作为 SAM3 的框样例 prompt
           (add_geometric_prompt)，一次前向直接拿到本帧内的相似目标，不再需要
           "先枚举全部候选框、再逐个比对"这一步——这个能力(exemplar/框样例检索)
           是官方 SAM3 自带的，本仓库在这次重构之前完全没有用到过。

    use_color: 可选的颜色后过滤。SAM3 的框样例检索本身已经同时编码了外观和几何
        信息，一般不需要额外的颜色校验；但如果场景里有"形状几乎一样、只有颜色不同"
        的物体(比如不同颜色的同款游戏道具)，可以打开这个开关，用正例框的颜色直方图
        再筛一遍候选结果，颜色差异过大的会被剔除。
    """
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
                    # 归一化直方图的 L1 距离理论范围是 [0, 2]；这个 1.0 的阈值是一个
                    # 比较宽松的经验值(容忍光照/角度带来的颜色偏移，只挡明显不同色系
                    # 的候选)，没有做过大规模统计调参，如果发现挡太多/挡太少，
                    # 可以在这里调整。
                    COLOR_VETO_L1_THRESHOLD = 1.0
                    for r in results:
                        cand_hist = _calculate_region_color_hist(image_bgr, r['box'])
                        if cand_hist is None or _color_hist_distance(cand_hist, pos_hist) < COLOR_VETO_L1_THRESHOLD:
                            filtered.append(r)
                    results = filtered

    return results


# ==============================================================================
# 功能 3: 智能选择 · 类别库 / 批量应用到整段视频 (SAM3 文本检索)
# ==============================================================================

def predict_by_class_text(video_uuid, frame_number, class_name, confidence_threshold=0.5):
    """
    用某个类别的 SAM3 检索文本在指定帧上查询。取代旧的
    "build_prototypes_for_class + predict_with_prototypes" 两步。
    """
    if sam_tasks is None:
        raise RuntimeError("SAM 功能未安装。")

    retrieval_text = get_retrieval_text_for_class(class_name)
    with AI_MODEL_LOCK:
        results = sam_tasks.sam3_query_frame(
            video_uuid, frame_number, text_prompt=retrieval_text, confidence=confidence_threshold
        )
    return results


# ==============================================================================
# 功能 4: 数据集一致性检查 (Consistency Check / AI Quality Control)
# ==============================================================================
# 决策记录 (对应方案 §4.4 最终版):
# 旧实现: 已标注框 -> MobileNet 语义向量 + HSV 颜色直方图 -> 类内/类间余弦相似度
#        和颜色距离找离群点。这是一个"提取通用视觉 embedding 再做无监督聚类"的方案。
# 新实现: 核实过 SAM3 本身不提供任何"数据集标注质量审查"工具(sam3/eval/ 下都是
#        对齐 COCO/HOTA/TETA 这类学术基准的评测脚本，和审查用户自己的数据集是两回
#        事)。与其给 SAM3 的骨干网络硬套一个它不是为聚类/度量学习设计的向量空间,
#        不如换个角度: 不提取 embedding, 直接问 SAM3"这块区域符不符合这个类别的
#        描述", 把"一致性"重新定义成"SAM3 自己给出的开放词汇置信度分数",
#        而不是"和同类样本的向量距离"。
#        颜色直方图部分和 MobileNet 无关，原样保留。
# ==============================================================================

def check_dataset_consistency(dataset_uuid, enable_color_check=True):
    """
    返回: (outlier_image_indices: set[int], all_bboxes_info: list[dict], message: str)
    outlier_image_indices 里的下标对应 all_bboxes_info 里 'image_index' 字段的值，
    也就是"第几张涉及标注的图片"，和旧实现的返回结构保持一致，方便路由层复用。
    """
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
    frames_to_process = defaultdict(list)  # "video_uuid;frame_number" -> [(global_idx, rect, label), ...]
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
    semantic_low_threshold = float(settings.get('consistency_semantic_threshold', 0.3))
    confusion_margin = float(settings.get('consistency_confusion_margin', 0.15))

    color_prototype_hists = defaultdict(list)
    outlier_image_indices = set()

    # 第一遍: 颜色直方图 (纯 OpenCV，和 SAM3 无关，先算完不占 GPU/锁)
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

    # 第二遍: 逐帧跑 SAM3 语义一致性检查。同一帧涉及到的每个类别各查一次，
    # backbone 只算一次(帧级缓存)，一帧内该类别的所有框共用这一次查询结果。
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
                        f"[Consistency] SEMANTIC outlier: '{label}' own_score={own_score:.2f} "
                        f"(SAM3 认为这块区域不太像它自己的类别描述). image_index={all_bboxes_info[global_idx]['image_index']}"
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
