import logging
import os
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

# 双引擎缓存管理
_sam_cache = {
    # SAM 2 缓存
    "sam2_video_predictor": None,
    "sam2_image_predictor": None,
    # SAM 3.1 缓存
    "sam3_multiplex_predictor": None,
    "sam3_image_model": None,
    "sam3_image_processor": None,
    # 公共状态
    "checkpoint": None,
    "device": None
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
    """加载支持文本理解的 SAM 3.1 高级引擎"""
    if not HAS_SAM3: return None
    settings = settings_manager.load_settings()

    checkpoint_name = settings.get('sam_model_checkpoint', '')
    if "sam2" in checkpoint_name or not checkpoint_name.endswith("sam3.1_multiplex.pt"):
        checkpoint_name = os.path.join("sam3.1_multiplex", "sam3.1_multiplex.pt")

    checkpoint_path = os.path.abspath(os.path.join(config.BASE_DIR, "checkpoints", checkpoint_name))
    device = settings_manager.get_device()

    try:
        if mode == "video" and _sam_cache["sam3_multiplex_predictor"] is None:
            logging.info("[SAM3] Loading Multiplex Video Predictor...")
            _sam_cache["sam3_multiplex_predictor"] = build_sam3_multiplex_video_predictor(
                checkpoint_path=checkpoint_path
            )
        elif mode == "image" and _sam_cache["sam3_image_processor"] is None:
            logging.info("[SAM3] Loading Image Processor for LAM...")
            _sam_cache["sam3_image_model"] = build_sam3_image_model(checkpoint_path=checkpoint_path)
            _sam_cache["sam3_image_processor"] = Sam3Processor(_sam_cache["sam3_image_model"])
    except Exception as e:
        logging.error(f"[SAM3] Load failed: {e}", exc_info=True)
        return None

    return _sam_cache["sam3_multiplex_predictor"] if mode == "video" else _sam_cache["sam3_image_processor"]


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
                    return {'x1': int(x_min), 'y1': int(y_min), 'x2': int(x_max) + 1, 'y2': int(y_max) + 1}
    except Exception as e:
        logging.error(f"[SAM2] Point predict failed: {e}")
    return None


def predict_boxes_from_text_sam3(image_path, text_prompt):
    """
    文本分割 (LAM Text)：完全走 SAM 3.1 高级引擎。
    开放词汇，支持自然语言一键分割一切符合概念的物体。
    """
    processor = _load_sam3_models(mode="image")
    if processor is None:
        raise RuntimeError("SAM 3.1 Image Processor is offline, True LAM unavailable.")

    image = Image.open(image_path).convert("RGB")

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    with torch.autocast(device_type=device_type, dtype=dtype):
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )

    boxes = output.get("boxes", [])
    scores = output.get("scores", [])

    results = []
    for i in range(len(boxes)):
        box_data = boxes[i]
        score = float(scores[i])
        if torch.is_tensor(box_data):
            box_data = box_data.cpu().numpy().tolist()

        x1, y1, x2, y2 = map(int, box_data)
        results.append({"box": [x1, y1, x2, y2], "score": score})

    return results


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


def get_sam_model():
    settings = settings_manager.load_settings()
    # 只要拥有其中一个引擎，SAM 核心就视为可用
    return settings.get('enable_sam_model', True) and (HAS_SAM2 or HAS_SAM3)