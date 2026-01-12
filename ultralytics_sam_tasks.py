import logging
import os
import torch
import numpy as np
import cv2
import shutil
import gc
import uuid
import time
from collections import OrderedDict

# 引入官方 SAM2
try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    HAS_SAM2 = True
except ImportError:
    logging.critical("FATAL: 'sam2' library not found.")
    HAS_SAM2 = False

import config
import database
import file_storage
from bbox_writer import convert_text_to_rects_and_labels
import settings_manager

# ==============================================================================
# 配置复刻
# ==============================================================================
INFERENCE_SIZE = 512  # 建议提高到 512 以获得更好体验
CHUNK_SIZE = 200

# 全局缓存
_sam_cache = {
    "video_predictor": None,
    "image_predictor": None,
    "auto_mask_generator": None,
    "config": None,
    "checkpoint": None,
    "device": None
}


def _load_sam2_models(mode="video"):
    if not HAS_SAM2: return None

    settings = settings_manager.load_settings()
    model_cfg = settings.get('sam_model_config', 'configs/sam2.1/sam2.1_hiera_t.yaml')
    checkpoint_name = settings.get('sam_model_checkpoint', 'sam2.1_hiera_tiny.pt')
    checkpoint_path = os.path.join(config.BASE_DIR, "checkpoints", checkpoint_name)
    device = settings_manager.get_device()

    if (_sam_cache["config"] != model_cfg or
            _sam_cache["checkpoint"] != checkpoint_path or
            str(_sam_cache["device"]) != str(device)):
        logging.info(f"[SAM2] Reloading models... CFG: {model_cfg}, Device: {device}")
        _sam_cache["video_predictor"] = None
        _sam_cache["image_predictor"] = None
        _sam_cache["auto_mask_generator"] = None
        _sam_cache["config"] = model_cfg
        _sam_cache["checkpoint"] = checkpoint_path
        _sam_cache["device"] = device

    if not os.path.exists(checkpoint_path):
        logging.error(f"[SAM2] Checkpoint not found: {checkpoint_path}")
        return None

    try:
        inference_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        if mode == "video" and _sam_cache["video_predictor"] is None:
            logging.info(f"[SAM2] Building Video Predictor ({inference_dtype})...")
            _sam_cache["video_predictor"] = build_sam2_video_predictor(
                model_cfg, checkpoint_path, device=device
            )

        elif mode == "image" and _sam_cache["image_predictor"] is None:
            _sam_cache["image_predictor"] = SAM2ImagePredictor(
                build_sam2(model_cfg, checkpoint_path, device=device)
            )

        elif mode == "auto" and _sam_cache["auto_mask_generator"] is None:
            conf = float(settings.get('sam_mask_confidence', 0.7))
            iou = float(settings.get('nms_iou_threshold', 0.7))
            _sam_cache["auto_mask_generator"] = SAM2AutomaticMaskGenerator(
                build_sam2(model_cfg, checkpoint_path, device=device),
                points_per_side=32, pred_iou_thresh=conf, stability_score_thresh=iou
            )

    except Exception as e:
        logging.error(f"[SAM2] Error building model ({mode}): {e}", exc_info=True)
        return None

    if mode == "video": return _sam_cache["video_predictor"]
    if mode == "image": return _sam_cache["image_predictor"]
    if mode == "auto": return _sam_cache["auto_mask_generator"]
    return None


def _mask_to_bbox(mask):
    if mask is None: return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1]


# ==============================================================================
# 1. 核心视频追踪
# ==============================================================================

def prepare_chunk_images(video_uuid, chunk_start, chunk_end, temp_dir, inference_size, session):
    """
    修改点：增加了 session 参数，在处理每一张图片时都检查是否停止。
    """
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    frame_files = []

    video_info = database.get_video_entity(video_uuid)
    orig_w = video_info['width']
    orig_h = video_info['height']

    for i, frame_num in enumerate(range(chunk_start, chunk_end + 1)):
        # === 高频检查点 1：每处理一帧图片前都检查 ===
        if session.get('stop_requested', False):
            logging.info("[SAM2] Stop detected during image preparation.")
            return None, None, None
        # ==========================================

        src_path = file_storage.get_frame_path(video_uuid, frame_num)
        if not os.path.exists(src_path):
            continue

        img = cv2.imread(src_path)
        if img is None: continue

        img_resized = cv2.resize(img, (inference_size, inference_size))

        dst_name = f"{i:05d}.jpg"
        dst_path = os.path.join(temp_dir, dst_name)

        cv2.imwrite(dst_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_files.append(frame_num)

    return frame_files, orig_w, orig_h


def track_video_ultralytics(video_uuid, start_frame, end_frame, init_bboxes_text, session):
    predictor = _load_sam2_models(mode="video")
    if predictor is None:
        raise RuntimeError("SAM2 Video Predictor init failed.")

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

    inference_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    current_start = start_frame
    base_temp_dir = os.path.join(config.STORAGE_DIR, "temp_sam2", str(uuid.uuid4()))

    try:
        while current_start <= end_frame:
            # === 高频检查点 2：Chunk 开始前 ===
            if session.get('stop_requested', False):
                break

            chunk_end = min(current_start + CHUNK_SIZE - 1, end_frame)
            logging.info(f"[SAM2 Chunk] Processing frames {current_start} to {chunk_end}...")

            chunk_dir = os.path.join(base_temp_dir, f"chunk_{current_start}")

            # 2.1 准备数据 (传入 session 进行检查)
            frame_map, orig_w, orig_h = prepare_chunk_images(
                video_uuid, current_start, chunk_end, chunk_dir, INFERENCE_SIZE, session
            )

            # 如果 prepare 过程中被打断，frame_map 会是 None
            if not frame_map or session.get('stop_requested', False):
                logging.info("[SAM2] Stop detected after prepare_chunk_images.")
                break

            # === 高频检查点 3：初始化状态前 ===
            if session.get('stop_requested', False): break

            inference_state = predictor.init_state(video_path=chunk_dir)

            scale_x = orig_w / INFERENCE_SIZE
            scale_y = orig_h / INFERENCE_SIZE

            # === 高频检查点 4：添加 Prompt 前 ===
            if session.get('stop_requested', False): break

            with torch.autocast("cuda", dtype=inference_dtype):
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

            # === 高频检查点 5：开始推理前 ===
            if session.get('stop_requested', False): break

            with torch.inference_mode(), torch.autocast("cuda", dtype=inference_dtype):
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):

                    # === 高频检查点 6：每推理一帧都检查 ===
                    if session.get('stop_requested', False):
                        logging.info("[SAM2] Stop requested inside propagation loop.")
                        break

                    if out_frame_idx >= len(frame_map): continue
                    global_frame_num = frame_map[out_frame_idx]

                    if global_frame_num == start_frame:
                        continue

                    current_frame_lines = []
                    for i, out_obj_id in enumerate(out_obj_ids):
                        internal_id = int(out_obj_id)
                        target_oid = None
                        for oid, data in active_objects.items():
                            if data['internal_id'] == internal_id:
                                target_oid = oid
                                break

                        if not target_oid: continue

                        mask_tensor = (out_mask_logits[i] > 0.0)
                        mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

                        cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)

                            x1 = int(x * scale_x)
                            y1 = int(y * scale_y)
                            x2 = int((x + w) * scale_x)
                            y2 = int((y + h) * scale_y)

                            active_objects[target_oid]['last_box'] = [x1, y1, x2, y2]
                            label = active_objects[target_oid]['label']
                            current_frame_lines.append(f"{x1},{y1},{x2},{y2},{label},{target_oid}")

                    if current_frame_lines:
                        session['results'][global_frame_num] = "\n".join(current_frame_lines)
                    else:
                        session['results'][global_frame_num] = ""

                    session['progress'] = global_frame_num - start_frame

            # 清理当前块
            predictor.reset_state(inference_state)
            if os.path.exists(chunk_dir): shutil.rmtree(chunk_dir)
            gc.collect()
            torch.cuda.empty_cache()

            current_start += CHUNK_SIZE

    except Exception as e:
        logging.error(f"Tracking error: {e}", exc_info=True)
        session['status'] = 'FAILED'
        session['message'] = str(e)
    finally:
        if os.path.exists(base_temp_dir):
            shutil.rmtree(base_temp_dir, ignore_errors=True)

        if session.get('stop_requested', False):
            session['status'] = 'STOPPED'
        elif session['status'] == 'PROCESSING':
            session['status'] = 'COMPLETED'

        logging.info(f"[SAM2] Tracking process ended with status: {session['status']}")


# ==============================================================================
# 2. 自动掩码生成 - 不变
# ==============================================================================
def generate_masks_for_frame(video_uuid, frame_number):
    generator = _load_sam2_models(mode="auto")
    if generator is None: return None, None

    frame_path = file_storage.get_frame_path(video_uuid, frame_number)
    if not os.path.exists(frame_path): return None, None

    image = cv2.imread(frame_path)
    if image is None: return None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    inference_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    with torch.inference_mode(), torch.autocast("cuda", dtype=inference_dtype):
        masks_data = generator.generate(image)

    if not masks_data:
        dev = settings_manager.get_device()
        return torch.empty(0, 4, device=dev), torch.empty(0, 0, 0, device=dev)

    boxes = []
    masks_list = []
    for m in masks_data:
        x, y, w, h = m['bbox']
        boxes.append([x, y, x + w, y + h])
        masks_list.append(m['segmentation'])

    dev = settings_manager.get_device()
    boxes_t = torch.tensor(boxes, dtype=torch.float32, device=dev)
    masks_np = np.array(masks_list, dtype=np.bool_)
    masks_t = torch.tensor(masks_np, dtype=torch.float32, device=dev).unsqueeze(1)
    return boxes_t, masks_t


# ==============================================================================
# 3. 单图预测 - 不变
# ==============================================================================
def predict_box_from_point_ultralytics(image_path, point_coords):
    predictor = _load_sam2_models(mode="image")
    if predictor is None: return None

    image = cv2.imread(image_path)
    if image is None: return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    inference_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    with torch.inference_mode(), torch.autocast("cuda", dtype=inference_dtype):
        predictor.set_image(image)
        input_point = np.array([point_coords])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=False
        )
        if masks is not None and masks.size > 0:
            mask = masks[0]
            bbox = _mask_to_bbox(mask)
            if bbox:
                return {'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]}
    return None


def get_sam_model():
    return HAS_SAM2