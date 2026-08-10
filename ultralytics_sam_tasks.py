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
    工业级亚像素姿态零塌陷追帧引擎 (Sub-pixel LK Optical Flow + SAM2 BBox Kinematic Engine)：
    1. 使用 SAM2 跟踪目标的整体 BBox (Person Tracker)，锁死 Instance ID 永不混淆；
    2. 使用双向 PyLK 光流 (Forward-Backward Optical Flow) 进行姿态 0 维关键点亚像素追踪，彻底解决点塌陷/误吸到耳朵的现象；
    3. 利用 FB-Error (前向-反向重构误差) 自动鉴定遮挡（v=1）与可见（v=2）；
    4. 遮挡点融合刚体运动外推与解剖学骨骼拓扑约束，重现后自动复苏。
    """
    predictor = _load_sam2_models(mode="video")
    if predictor is None: raise RuntimeError("SAM 2 Video Predictor offline.")

    settings = settings_manager.load_settings()
    inference_size = 1024
    chunk_size = int(settings.get('batch_tracking_chunk_size', 200))

    from annotation_model import AnnotationData, AnnotationObject

    active_instances = {}
    for idx, obj in enumerate(init_pose_objects, start=1):
        obj_id = obj.get('id') or str(uuid.uuid4())
        label = obj.get('label') or 'person'
        kpts = obj.get('keypoints') or []

        bbox = obj.get('bbox')
        if not bbox or len(bbox) < 4:
            v_pts = [p for p in kpts if p.get('v', 2) > 0]
            pts = v_pts if v_pts else kpts
            if pts:
                pad = 15
                xs = [p['x'] for p in pts]
                ys = [p['y'] for p in pts]
                bbox = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]
            else:
                continue

        active_instances[idx] = {
            'instance_id': obj_id,
            'label': label,
            'last_bbox': [float(b) for b in bbox],
            'keypoints': [{
                'name': kp.get('name', f'pt_{k_idx}'),
                'x': float(kp.get('x', 0)),
                'y': float(kp.get('y', 0)),
                'v': int(kp.get('v', 2))
            } for k_idx, kp in enumerate(kpts)]
        }

    if not active_instances:
        session['status'] = 'COMPLETED'
        return

    # 从数据库获取该类别的骨骼连线图（用于遮挡时的拓扑约束）
    schema_edges = []
    try:
        first_label = list(active_instances.values())[0]['label']
        schema_raw = database.get_class_keypoint_schema(first_label)
        if schema_raw:
            schema_data = json.loads(schema_raw) if isinstance(schema_raw, str) else schema_raw
            schema_edges = schema_data.get('edges', [])
    except Exception as e:
        logging.warning(f"Failed to load keypoint schema edges for occlusion constraints: {e}")

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

            # 1. 注册 Top-Down 实例 BBox 给 SAM2 跟踪整体包围框
            for internal_id, inst_info in active_instances.items():
                b = inst_info['last_bbox']
                box_resized = np.array([
                    b[0] / scale_x, b[1] / scale_y, b[2] / scale_x, b[3] / scale_y
                ], dtype=np.float32)

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=internal_id,
                    box=box_resized
                )

            # 2. 获取 SAM2 的包围框序列，并配合 OpenCV 光流进行 亚像素点追踪
            # 按顺序加载图像帧用于双向光流计算
            sorted_frame_indices = sorted(frame_map.keys())
            prev_gray_img = None

            for local_idx in sorted_frame_indices:
                if session.get('stop_requested', False): break
                global_frame_num = frame_map[local_idx]

                frame_file = os.path.join(chunk_dir, f"{local_idx:05d}.jpg")
                curr_img_bgr = cv2.imread(frame_file)
                if curr_img_bgr is None: continue
                curr_gray_img = cv2.cvtColor(curr_img_bgr, cv2.COLOR_BGR2GRAY)

                if global_frame_num == start_frame:
                    prev_gray_img = curr_gray_img
                    continue

                # 查询 SAM2 在当前帧为每个实例预测的包围框
                sam2_bboxes = {}
                try:
                    out_frame_idx, out_obj_ids, out_mask_logits = predictor.propagate_in_video(
                        inference_state, start_frame_idx=local_idx, max_frame_num_to_track=1
                    ).__next__()
                    for i, out_obj_id in enumerate(out_obj_ids):
                        internal_id = int(out_obj_id)
                        mask_np = (out_mask_logits[i] > 0.0).squeeze().cpu().numpy().astype(np.uint8)
                        cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            sam2_bboxes[internal_id] = [
                                max(0, int(x * scale_x)),
                                max(0, int(y * scale_y)),
                                min(orig_w, int((x + w) * scale_x)),
                                min(orig_h, int((y + h) * scale_y))
                            ]
                except Exception:
                    pass

                frame_updated_objects = []

                for internal_id, inst_info in active_instances.items():
                    prev_bbox = inst_info['last_bbox']
                    prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2.0
                    prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2.0
                    prev_w = max(1.0, prev_bbox[2] - prev_bbox[0])
                    prev_h = max(1.0, prev_bbox[3] - prev_bbox[1])

                    # 目标的新包围框（优先 SAM2，备选基于前帧）
                    new_bbox = sam2_bboxes.get(internal_id, prev_bbox)
                    new_x1, new_y1, new_x2, new_y2 = new_bbox
                    new_cx = (new_x1 + new_x2) / 2.0
                    new_cy = (new_y1 + new_y2) / 2.0
                    new_w = max(1.0, float(new_x2 - new_x1))
                    new_h = max(1.0, float(new_y2 - new_y1))

                    dx = new_cx - prev_cx
                    dy = new_cy - prev_cy
                    scale_w = new_w / prev_w
                    scale_h = new_h / prev_h

                    # 构建当前实例的所有关键点 PyLK 光流输入
                    kpts_prev = inst_info['keypoints']
                    pts_prev_arr = np.array([[[kp['x'], kp['y']]] for kp in kpts_prev], dtype=np.float32)

                    # A. 前向与反向光流计算 (Forward-Backward LK Flow)
                    pts_next_arr, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray_img, curr_gray_img, pts_prev_arr, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                    )
                    pts_back_arr, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                        curr_gray_img, prev_gray_img, pts_next_arr, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                    )

                    # 计算 FB 误差
                    fb_err = np.linalg.norm(pts_prev_arr - pts_back_arr, axis=2).ravel()

                    updated_kpts = []
                    kp_dict_curr = {}

                    for k_idx, kp in enumerate(kpts_prev):
                        fwd_valid = (st_fwd[k_idx][0] == 1)
                        err_val = fb_err[k_idx]
                        nxt_x = float(pts_next_arr[k_idx][0][0])
                        nxt_y = float(pts_next_arr[k_idx][0][1])

                        # 光流跟踪有效性判断：前向成功、FB 误差 <= 3.5 像素，且落入包围框扩展区内
                        in_box = (new_x1 - 30 <= nxt_x <= new_x2 + 30) and (new_y1 - 30 <= nxt_y <= new_y2 + 30)
                        if fwd_valid and err_val <= 3.5 and in_box:
                            # 亚像素级精准无塌陷跟踪，标记 v=2 (Visible)
                            kp_entry = {
                                'name': kp['name'],
                                'x': round(nxt_x, 2),
                                'y': round(nxt_y, 2),
                                'v': 2
                            }
                        else:
                            # 遮挡/丢失状况：刚体运动向量外推，标记 v=1 (Occluded)
                            rel_x = kp['x'] - prev_cx
                            rel_y = kp['y'] - prev_cy
                            pred_x = round(new_cx + rel_x * scale_w, 2)
                            pred_y = round(new_cy + rel_y * scale_h, 2)
                            kp_entry = {
                                'name': kp['name'],
                                'x': pred_x,
                                'y': pred_y,
                                'v': 1
                            }

                        updated_kpts.append(kp_entry)
                        kp_dict_curr[kp['name']] = kp_entry

                    # B. 骨骼拓扑长度约束 (Kinematic Bone Constraint Adjustment)
                    if schema_edges:
                        prev_kp_dict = {p['name']: p for p in kpts_prev}
                        for edge in schema_edges:
                            if len(edge) >= 2:
                                p1_name, p2_name = edge[0], edge[1]
                                if p1_name in kp_dict_curr and p2_name in kp_dict_curr:
                                    k1 = kp_dict_curr[p1_name]
                                    k2 = kp_dict_curr[p2_name]

                                    if (k1['v'] == 2 and k2['v'] == 1) or (k1['v'] == 1 and k2['v'] == 2):
                                        vis_k = k1 if k1['v'] == 2 else k2
                                        occ_k = k2 if k1['v'] == 2 else k1

                                        if p1_name in prev_kp_dict and p2_name in prev_kp_dict:
                                            old_p1 = prev_kp_dict[p1_name]
                                            old_p2 = prev_kp_dict[p2_name]
                                            orig_len = np.sqrt((old_p1['x'] - old_p2['x'])**2 + (old_p1['y'] - old_p2['y'])**2)
                                            if orig_len > 2.0:
                                                dir_x = occ_k['x'] - vis_k['x']
                                                dir_y = occ_k['y'] - vis_k['y']
                                                curr_len = np.sqrt(dir_x**2 + dir_y**2)
                                                if curr_len > 0.1:
                                                    ratio = orig_len / curr_len
                                                    occ_k['x'] = round(vis_k['x'] + dir_x * ratio, 2)
                                                    occ_k['y'] = round(vis_k['y'] + dir_y * ratio, 2)

                    inst_info['last_bbox'] = new_bbox
                    inst_info['keypoints'] = updated_kpts

                    frame_updated_objects.append(AnnotationObject(
                        id=inst_info['instance_id'],
                        type='keypoint',
                        label=inst_info['label'],
                        bbox=new_bbox,
                        keypoints=updated_kpts
                    ))

                ann_data = AnnotationData()
                ann_data.objects = frame_updated_objects

                database.save_frame_annotations(video_uuid, global_frame_num, ann_data.to_json())
                session['results'][global_frame_num] = ann_data.to_json()
                session['progress'] = global_frame_num - start_frame

                prev_gray_img = curr_gray_img

            predictor.reset_state(inference_state)
            if os.path.exists(chunk_dir): shutil.rmtree(chunk_dir)
            gc.collect()
            torch.cuda.empty_cache()

            current_start += chunk_size

    except Exception as e:
        logging.error(f"[Subpixel Optical Flow Pose] Tracking failed: {e}", exc_info=True)
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