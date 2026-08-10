import logging
import os
import threading
import torch
import numpy as np
import cv2
import shutil
import gc
import uuid
from collections import OrderedDict
from PIL import Image

# ==============================================================================
# 1. 双引擎自适应导入 (Dual-Engine Import)
# ==============================================================================
# 基础引擎 (SAM 2)
try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    HAS_SAM2 = True
except ImportError:
    logging.warning("[SAM2] SAM 2 engine not found. Fast point clicks disabled.")
    HAS_SAM2 = False

# 高级引擎 (SAM 3.1)
try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor, build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    HAS_SAM3 = True
except ImportError:
    logging.warning("[SAM3] SAM 3.1 engine not found. Text prompt features disabled.")
    HAS_SAM3 = False

import config
import database
import file_storage
from bbox_writer import convert_text_to_rects_and_labels
import settings_manager

# 模型缓存管理
_sam_cache = {
    # SAM 2 缓存
    "sam2_video_predictor": None,
    "sam2_image_predictor": None,
    # SAM 3.1 缓存
    "multiplex_predictor": None,
    "image_model": None,
    "image_processor": None,
    # 按模式分别记录 checkpoint/device
    "image_checkpoint": None,
    "image_device": None,
    "video_checkpoint": None,
    "video_device": None,
}


# ==============================================================================
# 2. 引擎自适应加载层
# ==============================================================================

def _load_sam2_models(mode="image"):
    """加载高精度的 SAM 2 基础引擎"""
    if not HAS_SAM2: return None
    settings = settings_manager.load_settings()

    # 默认使用 sam2.1_b.pt 或 sam2.1_t.pt 满足毫秒级点选
    checkpoint_name = "sam2.1_b.pt"
    checkpoint_path = os.path.join(config.BASE_DIR, "checkpoints", checkpoint_name)
    device = settings_manager.get_device()

    if not os.path.exists(checkpoint_path):
        logging.warning(f"[SAM2] Checkpoint not found: {checkpoint_path}, falling back to Tiny")
        checkpoint_name = "sam2.1_t.pt"
        checkpoint_path = os.path.join(config.BASE_DIR, "checkpoints", checkpoint_name)

    # 动态确定配置文件
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml" if "_b.pt" in checkpoint_name else "configs/sam2.1/sam2.1_hiera_t.yaml"

    try:
        if mode == "image" and _sam_cache["sam2_image_predictor"] is None:
            logging.info(f"[SAM2] Loading Image Predictor on {device}...")
            _sam_cache["sam2_image_predictor"] = SAM2ImagePredictor(
                build_sam2(model_cfg, checkpoint_path, device=device)
            )
        elif mode == "video" and _sam_cache["sam2_video_predictor"] is None:
            logging.info(f"[SAM2] Loading Video Predictor on {device}...")
            _sam_cache["sam2_video_predictor"] = build_sam2_video_predictor(
                model_cfg, checkpoint_path, device=device
            )
    except Exception as e:
        logging.error(f"[SAM2] Load failed: {e}")
        return None

    return _sam_cache["sam2_image_predictor"] if mode == "image" else _sam_cache["sam2_video_predictor"]


def _load_sam3_models(mode="video"):
    """智能加载 SAM 3.1 模型：图像LAM和视频追踪双通道分离"""
    settings = settings_manager.load_settings()
    if not settings.get('enable_sam_model', True):
        return None
    if not HAS_SAM3:
        return None

    if mode == "image":
        checkpoint_name = os.path.join("sam3", "sam3.pt")
    else:
        checkpoint_name = os.path.join("sam3.1_multiplex", "sam3.1_multiplex.pt")

    checkpoint_path = os.path.abspath(os.path.join(config.BASE_DIR, "checkpoints", checkpoint_name))
    device = settings_manager.get_device()

    ckpt_key = f"{mode}_checkpoint"
    dev_key = f"{mode}_device"
    if (_sam_cache.get(ckpt_key) != checkpoint_path or str(_sam_cache.get(dev_key)) != str(device)):
        logging.info(f"[SAM3] Loading {mode} engine... Checkpoint: {checkpoint_name}")
        if mode == "video":
            _sam_cache["multiplex_predictor"] = None
        else:
            _sam_cache["image_model"] = None
            _sam_cache["image_processor"] = None
            clear_sam3_frame_state_cache()
        _sam_cache[ckpt_key] = checkpoint_path
        _sam_cache[dev_key] = device
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    try:
        if mode == "video" and _sam_cache["multiplex_predictor"] is None:
            _sam_cache["multiplex_predictor"] = build_sam3_multiplex_video_predictor(checkpoint_path=checkpoint_path, device=str(device))

        elif mode == "image" and _sam_cache["image_processor"] is None:
            _sam_cache["image_model"] = build_sam3_image_model(checkpoint_path=checkpoint_path, device=str(device))
            _sam_cache["image_processor"] = Sam3Processor(_sam_cache["image_model"], device=str(device))
    except Exception as e:
        logging.error(f"[SAM3] Error building model ({mode}): {e}")
        return None

    return _sam_cache["multiplex_predictor"] if mode == "video" else _sam_cache["image_processor"]


# --- SAM3 帧级 backbone 缓存与统一查询 ---

_sam3_frame_state_cache = OrderedDict()
_SAM3_QUERY_LOCK = threading.Lock()


def _sam3_frame_cache_put(key, value):
    _sam3_frame_state_cache[key] = value
    try:
        maxsize = int(settings_manager.load_settings().get('max_cache_size', 30))
    except Exception:
        maxsize = 30
    while len(_sam3_frame_state_cache) > maxsize:
        _sam3_frame_state_cache.popitem(last=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def clear_sam3_frame_state_cache():
    """在 SAM3 checkpoint / 设备变更时调用，避免复用到用旧模型算出来的 backbone 特征。"""
    _sam3_frame_state_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pixel_box_to_norm_cxcywh(box, img_w, img_h):
    """[x1,y1,x2,y2] 像素坐标 -> SAM3 geometric prompt 需要的 [cx,cy,w,h] 归一化坐标。"""
    x1, y1, x2, y2 = box
    return [
        (x1 + x2) / 2.0 / img_w,
        (y1 + y2) / 2.0 / img_h,
        max(1e-6, (x2 - x1) / img_w),
        max(1e-6, (y2 - y1) / img_h),
    ]


def _get_sam3_frame_state(processor, video_uuid, frame_number, image_path=None):
    """获取（或懒加载并缓存）某一帧的 SAM3 backbone state。命中缓存不会重跑 backbone。"""
    cache_key = f"{video_uuid}_{frame_number}"
    if cache_key in _sam3_frame_state_cache:
        state = _sam3_frame_state_cache.pop(cache_key)
        _sam3_frame_state_cache[cache_key] = state  # 移到 LRU 末尾
        return state

    if image_path is None:
        image_path = file_storage.get_frame_path(video_uuid, frame_number)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Frame image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device_type == "cuda" else torch.float32)
    with torch.autocast(device_type=device_type, dtype=dtype):
        state = processor.set_image(image)

    _sam3_frame_cache_put(cache_key, state)
    return state


def sam3_query_frame(video_uuid, frame_number, text_prompt=None, positive_boxes=None,
                      negative_boxes=None, confidence=0.25, image_path=None):
    """
    SAM3 统一查询入口：取代旧的"MobileNet 特征 + 候选框匹配"范式，是本次重构里
    智能选择(单样例/类别库)、LAM、批量应用、一致性检查共用的核心原语。

    - text_prompt: 开放词汇文本 query（比如某个类别的 SAM3 检索描述）
    - positive_boxes / negative_boxes: 像素坐标 [x1,y1,x2,y2] 的列表，框样例 query
      （对应官方 SAM3 的 exemplar/box-prompted 检索能力，即 add_geometric_prompt，
      本仓库在这次重构之前完全没有用到）
    - text_prompt 和 positive_boxes 可以同时提供（文本 + 框样例联合约束），至少要
      提供其中一个
    - 同一帧（同一个 video_uuid+frame_number）多次调用本函数只会跑一次图像 backbone

    返回: [{"box": [x1,y1,x2,y2], "score": float}, ...]，按 score 降序排列
    """
    if not text_prompt and not positive_boxes:
        raise ValueError("sam3_query_frame 至少需要 text_prompt 或 positive_boxes 中的一个。")

    processor = _load_sam3_models(mode="image")
    if processor is None:
        raise RuntimeError("SAM 3.1 Image Processor is not initialized (checkpoint 未加载或已被设置禁用)。")

    with _SAM3_QUERY_LOCK:
        state = _get_sam3_frame_state(processor, video_uuid, frame_number, image_path=image_path)

        # 每次独立查询前清空上一次查询残留的文本/框 prompt 和检测结果，避免不同查询
        # 之间互相串味；图像本身的 backbone_out（vision features）不受影响，继续复用。
        processor.reset_all_prompts(state)

        img_w = state["original_width"]
        img_h = state["original_height"]

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device_type == "cuda" else torch.float32)

        # confidence_threshold 是共享 processor 实例上的属性，查询期间临时改写，
        # 结束后还原，并且整个过程持有 _SAM3_QUERY_LOCK，避免并发请求互相冲突阈值。
        old_threshold = processor.confidence_threshold
        processor.confidence_threshold = confidence
        try:
            with torch.autocast(device_type=device_type, dtype=dtype):
                if text_prompt:
                    state = processor.set_text_prompt(prompt=text_prompt, state=state)
                if positive_boxes:
                    for box in positive_boxes:
                        norm_box = _pixel_box_to_norm_cxcywh(box, img_w, img_h)
                        state = processor.add_geometric_prompt(box=norm_box, label=True, state=state)
                if negative_boxes:
                    for box in negative_boxes:
                        norm_box = _pixel_box_to_norm_cxcywh(box, img_w, img_h)
                        state = processor.add_geometric_prompt(box=norm_box, label=False, state=state)
        finally:
            processor.confidence_threshold = old_threshold

        _sam3_frame_cache_put(f"{video_uuid}_{frame_number}", state)

        boxes = state.get("boxes", [])
        scores = state.get("scores", [])

    results = []
    sam2_pred = _load_sam2_models(mode="image")
    img_cv = None
    if sam2_pred is not None:
        try:
            img_path = file_storage.get_frame_path(video_uuid, frame_number)
            if os.path.exists(img_path):
                img_cv = cv2.imread(img_path)
                if img_cv is not None:
                    sam2_pred.set_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        except Exception:
            img_cv = None

    for i in range(len(boxes)):
        box_data = boxes[i]
        score = float(scores[i])
        if torch.is_tensor(box_data):
            box_data = box_data.detach().cpu().numpy().tolist()
        x1, y1, x2, y2 = map(int, box_data)
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        if sam2_pred is not None and img_cv is not None:
            try:
                m, _, _ = sam2_pred.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
                if m is not None and m.size > 0:
                    mask_np = m[0].astype(np.uint8)
                    cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        if sam2_pred is not None and img_cv is not None:
                            try:
                                m, _, _ = sam2_pred.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
                                if m is not None and m.size > 0:
                                    mask_np = m[0].astype(np.uint8)
                                    cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if cnts:
                                        c = max(cnts, key=cv2.contourArea)
                                        # 修复：将系数从 0.004 缩小到 0.0015，提取更精细、贴合物体真实轮廓的多边形
                                        eps = 0.0015 * cv2.arcLength(c, True)
                                        approx = cv2.approxPolyDP(c, eps, True)
                                        polygon = approx.reshape(-1, 2).tolist()
                            except Exception as e:
                                logging.error(f"[SAM2 Polygon Refinement Failed]: {e}")
                        approx = cv2.approxPolyDP(c, eps, True)
                        polygon = approx.reshape(-1, 2).tolist()
            except Exception:
                pass

        results.append({"box": [x1, y1, x2, y2], "polygon": polygon, "score": score})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def warm_frame_cache(video_uuid, frame_number, image_path=None):
    """
    预热某一帧的 SAM3 backbone 缓存（跑一次 set_image，不做任何查询）。
    对应旧的 "/interactive_segment/preprocess" 语义：在用户真正画框/点击之前，
    提前把最贵的 backbone 前向跑掉，之后的查询就只需要跑轻量的检测头。
    """
    processor = _load_sam3_models(mode="image")
    if processor is None:
        raise RuntimeError("SAM 3.1 Image Processor is not initialized.")
    with _SAM3_QUERY_LOCK:
        _get_sam3_frame_state(processor, video_uuid, frame_number, image_path=image_path)
    return True


def is_frame_cached(video_uuid, frame_number):
    return f"{video_uuid}_{frame_number}" in _sam3_frame_state_cache


def frame_cache_size():
    return len(_sam3_frame_state_cache)


# ==============================================================================
# 3. 图像交互逻辑的分流 (Core Feature Dispatch)
# ==============================================================================

def predict_box_from_point_ultralytics(image_path, point_coords):
    """
    点选交互 (SAM Point)：完全走 SAM 2 引擎。
    提供极致的亚毫秒级响应和像素级微小目标边缘咬合。
    """
    predictor = _load_sam2_models(mode="image")
    if predictor is None:
        logging.error("[Engine Switch] SAM 2 is offline, point click unavailable.")
        return None

    image = cv2.imread(image_path)
    if image is None: return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        with torch.inference_mode():
            predictor.set_image(image)
            input_point = np.array([point_coords])
            input_label = np.array([1])
            masks, scores, logits = predictor.predict(
                point_coords=input_point, point_labels=input_label, multimask_output=False
            )
            if masks is not None and masks.size > 0:
                mask = masks[0]
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if np.any(rows) and np.any(cols):
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    bbox = {'x1': int(x_min), 'y1': int(y_min), 'x2': int(x_max) + 1, 'y2': int(y_max) + 1}

                    mask_np = mask.astype(np.uint8)
                    cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polygon = []
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        epsilon = 0.004 * cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, epsilon, True)
                        polygon = approx.reshape(-1, 2).tolist()

                    bbox['polygon'] = polygon
                    return bbox
    except Exception as e:
        logging.error(f"[SAM2] Point predict failed: {e}")
    return None


# 注：原来这里有一个独立的 predict_boxes_from_text_sam3(image_path, text_prompt, ...)
# 实现，逻辑和上面 §2b 的 sam3_query_frame 高度重复（都是 set_image + set_text_prompt），
# 且不参与帧级 backbone 缓存。这次重构统一收敛到 sam3_query_frame，唯一的调用方
# app.py 的 /api/sam3_text_predict 路由已同步改为直接调用 sam3_query_frame（它本来
# 就有 video_uuid/frame_number，可以正常吃到缓存收益）。


# ==============================================================================
# 4. 视频追踪的分流 (目前走稳定且占用极低的 SAM 2 Tracking)
# ==============================================================================

def prepare_chunk_images(video_uuid, chunk_start, chunk_end, temp_dir, inference_size, session):
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    frame_files = []
    video_info = database.get_video_entity(video_uuid)
    orig_w, orig_h = video_info['width'], video_info['height']
    for i, frame_num in enumerate(range(chunk_start, chunk_end + 1)):
        if session.get('stop_requested', False): return None, None, None
        src_path = file_storage.get_frame_path(video_uuid, frame_num)
        if not os.path.exists(src_path): continue
        img = cv2.imread(src_path)
        if img is None: continue
        img_resized = cv2.resize(img, (inference_size, inference_size))
        cv2.imwrite(os.path.join(temp_dir, f"{i:05d}.jpg"), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_files.append(frame_num)
    return frame_files, orig_w, orig_h


def track_video_ultralytics(video_uuid, start_frame, end_frame, init_bboxes_text, session):
    """
    点选追踪 (SAM Tracking)：采用极度成熟、低延迟且支持分块内存释放的 SAM 2 视频推理器。
    """
    predictor = _load_sam2_models(mode="video")
    if predictor is None: raise RuntimeError("SAM 2 Video Predictor offline.")

    settings = settings_manager.load_settings()
    inference_size = 1024  # 采用标准高清分辨率进行追踪
    chunk_size = int(settings.get('batch_tracking_chunk_size', 200))

    init_rects, init_labels, init_ids = convert_text_to_rects_and_labels(init_bboxes_text)

    active_objects = OrderedDict()
    for i, rect in enumerate(init_rects):
        oid = init_ids[i] or str(uuid.uuid4())
        active_objects[oid] = {
            "label": init_labels[i],
            "last_box": rect,
            "internal_id": i + 1
        }

    session['results'][start_frame] = init_bboxes_text
    session['total'] = (end_frame - start_frame) + 1
    session['progress'] = 0

    current_start = start_frame
    base_temp_dir = os.path.join(config.STORAGE_DIR, "temp_sam2_tracking", str(uuid.uuid4()))

    try:
        while current_start <= end_frame:
            if session.get('stop_requested', False): break

            chunk_end = min(current_start + chunk_size - 1, end_frame)
            chunk_dir = os.path.join(base_temp_dir, f"chunk_{current_start}")

            frame_map, orig_w, orig_h = prepare_chunk_images(video_uuid, current_start, chunk_end, chunk_dir,
                                                             inference_size, session)
            if not frame_map or session.get('stop_requested', False): break

            inference_state = predictor.init_state(video_path=chunk_dir)

            scale_x = orig_w / inference_size
            scale_y = orig_h / inference_size

            for oid, obj_data in active_objects.items():
                box_orig = obj_data['last_box']
                box_resized = np.array([
                    box_orig[0] / scale_x,
                    box_orig[1] / scale_y,
                    box_orig[2] / scale_x,
                    box_orig[3] / scale_y
                ], dtype=np.float32)

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_data['internal_id'],
                    box=box_resized
                )

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if session.get('stop_requested', False): break
                if out_frame_idx >= len(frame_map): continue
                global_frame_num = frame_map[out_frame_idx]
                if global_frame_num == start_frame: continue

                current_frame_lines = []
                for i, out_obj_id in enumerate(out_obj_ids):
                    internal_id = int(out_obj_id)
                    target_oid = None
                    for oid, data in active_objects.items():
                        if data['internal_id'] == internal_id:
                            target_oid = oid
                            break
                    if not target_oid: continue

                    mask_np = (out_mask_logits[i] > 0.0).squeeze().cpu().numpy().astype(np.uint8)
                    cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(c)
                        x1, y1 = int(x * scale_x), int(y * scale_y)
                        x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)

                        active_objects[target_oid]['last_box'] = [x1, y1, x2, y2]
                        label = active_objects[target_oid]['label']
                        current_frame_lines.append(f"{x1},{y1},{x2},{y2},{label},{target_oid}")

                session['results'][global_frame_num] = "\n".join(current_frame_lines)
                session['progress'] = global_frame_num - start_frame

            predictor.reset_state(inference_state)
            if os.path.exists(chunk_dir): shutil.rmtree(chunk_dir)
            gc.collect()
            torch.cuda.empty_cache()

            current_start += chunk_size

    except Exception as e:
        logging.error(f"[SAM2] Video tracking failed: {e}", exc_info=True)
        session['status'] = 'FAILED'
        session['message'] = str(e)
    finally:
        if os.path.exists(base_temp_dir):
            shutil.rmtree(base_temp_dir, ignore_errors=True)
        if session.get('stop_requested', False):
            session['status'] = 'STOPPED'
        elif session['status'] in ['PROCESSING', 'BATCH_PROCESSING']:
            session['status'] = 'COMPLETED'


def track_pose_video_sam2(video_uuid, start_frame, end_frame, init_pose_objects, session):
    """
    基于 SAM2 视频推理器的点追踪 (Point Tracking) 姿态骨架跨帧传播引擎。
    传入 init_pose_objects (AnnotationObject 字典数组)。
    """
    predictor = _load_sam2_models(mode="video")
    if predictor is None: raise RuntimeError("SAM 2 Video Predictor offline.")

    settings = settings_manager.load_settings()
    inference_size = 1024
    chunk_size = int(settings.get('batch_tracking_chunk_size', 200))

    from annotation_model import AnnotationData, AnnotationObject

    keypoint_tracker_map = {}
    point_counter = 1

    for obj in init_pose_objects:
        obj_id = obj.get('id') or str(uuid.uuid4())
        label = obj.get('label') or 'person'
        for kp in obj.get('keypoints', []):
            name = kp.get('name', 'pt')
            kx = float(kp.get('x', 0))
            ky = float(kp.get('y', 0))
            v = int(kp.get('v', 2))
            if v > 0:
                keypoint_tracker_map[point_counter] = {
                    'instance_id': obj_id,
                    'label': label,
                    'name': name,
                    'x': kx,
                    'y': ky,
                    'v': v
                }
                point_counter += 1

    session['total'] = (end_frame - start_frame) + 1
    session['progress'] = 0

    current_start = start_frame
    base_temp_dir = os.path.join(config.STORAGE_DIR, "temp_sam2_pose_tracking", str(uuid.uuid4()))

    try:
        while current_start <= end_frame:
            if session.get('stop_requested', False): break

            chunk_end = min(current_start + chunk_size - 1, end_frame)
            chunk_dir = os.path.join(base_temp_dir, f"chunk_{current_start}")

            frame_map, orig_w, orig_h = prepare_chunk_images(video_uuid, current_start, chunk_end, chunk_dir,
                                                             inference_size, session)
            if not frame_map or session.get('stop_requested', False): break

            inference_state = predictor.init_state(video_path=chunk_dir)

            scale_x = orig_w / inference_size
            scale_y = orig_h / inference_size

            for pid, pt_info in keypoint_tracker_map.items():
                pt_resized = np.array([[pt_info['x'] / scale_x, pt_info['y'] / scale_y]], dtype=np.float32)
                labels_array = np.array([1], dtype=np.int32)

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=pid,
                    points=pt_resized,
                    labels=labels_array
                )

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if session.get('stop_requested', False): break
                if out_frame_idx >= len(frame_map): continue
                global_frame_num = frame_map[out_frame_idx]
                if global_frame_num == start_frame: continue

                tracked_instances = {}
                for obj in init_pose_objects:
                    obj_id = obj.get('id')
                    tracked_instances[obj_id] = {
                        'id': obj_id,
                        'type': 'keypoint',
                        'label': obj.get('label', 'person'),
                        'keypoints': []
                    }

                for i, out_obj_id in enumerate(out_obj_ids):
                    pid = int(out_obj_id)
                    if pid not in keypoint_tracker_map:
                        continue

                    pt_info = keypoint_tracker_map[pid]
                    inst_id = pt_info['instance_id']

                    mask_np = (out_mask_logits[i] > 0.0).squeeze().cpu().numpy().astype(np.uint8)
                    cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx = float(M["m10"] / M["m00"])
                            cy = float(M["m01"] / M["m00"])
                        else:
                            x, y, w, h = cv2.boundingRect(c)
                            cx, cy = x + w / 2.0, y + h / 2.0

                        real_x = round(cx * scale_x, 2)
                        real_y = round(cy * scale_y, 2)
                        pt_info['x'] = real_x
                        pt_info['y'] = real_y

                        tracked_instances[inst_id]['keypoints'].append({
                            'name': pt_info['name'],
                            'x': real_x,
                            'y': real_y,
                            'v': 2
                        })
                    else:
                        tracked_instances[inst_id]['keypoints'].append({
                            'name': pt_info['name'],
                            'x': pt_info['x'],
                            'y': pt_info['y'],
                            'v': 1
                        })

                updated_objects = []
                for inst_id, inst_data in tracked_instances.items():
                    kpts = inst_data['keypoints']
                    if kpts:
                        xs = [p['x'] for p in kpts]
                        ys = [p['y'] for p in kpts]
                        pad = 10
                        inst_data['bbox'] = [
                            max(0, min(xs) - pad),
                            max(0, min(ys) - pad),
                            min(orig_w, max(xs) + pad),
                            min(orig_h, max(ys) + pad)
                        ]
                        updated_objects.append(inst_data)

                ann_data = AnnotationData()
                for uo in updated_objects:
                    ann_data.objects.append(AnnotationObject(
                        id=uo['id'],
                        type='keypoint',
                        label=uo['label'],
                        bbox=uo.get('bbox'),
                        keypoints=uo['keypoints']
                    ))

                database.save_frame_annotations(video_uuid, global_frame_num, ann_data.to_json())
                session['results'][global_frame_num] = ann_data.to_json()
                session['progress'] = global_frame_num - start_frame

            predictor.reset_state(inference_state)
            if os.path.exists(chunk_dir): shutil.rmtree(chunk_dir)
            gc.collect()
            torch.cuda.empty_cache()

            current_start += chunk_size

    except Exception as e:
        logging.error(f"[SAM2 Pose] Tracking failed: {e}", exc_info=True)
        session['status'] = 'FAILED'
        session['message'] = str(e)
    finally:
        if os.path.exists(base_temp_dir):
            shutil.rmtree(base_temp_dir, ignore_errors=True)
        if session.get('stop_requested', False):
            session['status'] = 'STOPPED'
        elif session['status'] in ['PROCESSING', 'BATCH_PROCESSING']:
            session['status'] = 'COMPLETED'


def get_sam_model():
    settings = settings_manager.load_settings()
    return settings.get('enable_sam_model', True) and (HAS_SAM2 or HAS_SAM3)