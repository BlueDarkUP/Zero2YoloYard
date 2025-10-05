import logging
import os
import torch
import numpy as np
import cv2
import uuid

try:
    from ultralytics import SAM
    from ultralytics.models.sam import SAM2VideoPredictor
    from ultralytics.engine.results import Results
except ImportError:
    logging.critical("FATAL: ultralytics library is not installed. Please run 'pip install ultralytics'.")
    SAM = None
    SAM2VideoPredictor = None
    Results = None

import config
import database
import file_storage
from bbox_writer import convert_text_to_rects_and_labels
import settings_manager

_sam_model_cache = {"model": None, "path": None}


def get_sam_model():
    global _sam_model_cache
    DEVICE = settings_manager.get_device()

    settings = settings_manager.load_settings()
    checkpoint_filename = settings.get('sam_model_checkpoint', 'sam2.1_t.pt')
    sam_checkpoint_path = os.path.join(config.BASE_DIR, "checkpoints", checkpoint_filename)

    if _sam_model_cache["path"] != sam_checkpoint_path or \
            (_sam_model_cache["model"] is not None and str(_sam_model_cache["model"].device) != str(DEVICE)):
        logging.info(f"Model/device change detected. Reloading SAM model to {DEVICE}. New model: {checkpoint_filename}")
        _sam_model_cache["model"] = None
        _sam_model_cache["path"] = None

    if _sam_model_cache["model"] is not None:
        return _sam_model_cache["model"]

    if SAM is None:
        logging.error("Ultralytics SAM class is not available due to import error.")
        return None

    if not os.path.exists(sam_checkpoint_path):
        logging.error(f"Ultralytics SAM checkpoint not found at {sam_checkpoint_path}. All SAM features are disabled.")
        return None

    try:
        logging.info(f"Loading Ultralytics SAM model ('{checkpoint_filename}') to device '{DEVICE}'...")
        model = SAM(sam_checkpoint_path)
        model.to(DEVICE)
        _sam_model_cache["model"] = model
        _sam_model_cache["path"] = sam_checkpoint_path
        logging.info("Ultralytics SAM model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Ultralytics SAM model: {e}", exc_info=True)
        return None

    return _sam_model_cache["model"]


def predict_box_from_point_ultralytics(image_path, point_coords):
    model = get_sam_model()
    if model is None:
        raise RuntimeError("Ultralytics SAM model is not available.")
    results = model(image_path, points=[point_coords], labels=[1])
    if results and results[0].boxes and results[0].boxes.xyxy.numel() > 0:
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    return None


def _get_bbox_from_mask(mask_data, original_width, original_height):
    if mask_data is None:
        return None
    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.cpu().numpy()
    mask = (mask_data * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(original_width, x + w)
        y2 = min(original_height, y + h)
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
    return None


def track_video_ultralytics(video_uuid, start_frame, end_frame, init_bboxes_text, session):
    model = get_sam_model()
    if model is None:
        raise RuntimeError("Ultralytics SAM model is not available for tracking.")
    video_info = database.get_video_entity(video_uuid)
    original_width = video_info['width']
    original_height = video_info['height']
    init_rects, init_labels, _ = convert_text_to_rects_and_labels(init_bboxes_text)
    if not init_rects:
        raise ValueError("No initial bounding boxes provided for tracking.")
    tracked_objects = {}
    first_frame_path = file_storage.get_frame_path(video_uuid, start_frame)
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"Initial frame not found at {first_frame_path}")
    results = model(first_frame_path, bboxes=init_rects)
    if isinstance(results, Results):
        results = [results]
    if results and results[0].masks:
        initial_masks = results[0].masks.data
        for i, mask in enumerate(initial_masks):
            bbox = _get_bbox_from_mask(mask, original_width, original_height)
            if bbox:
                tracked_objects[i] = {"label": init_labels[i], "bbox": bbox}
    session['results'][start_frame] = init_bboxes_text
    session['progress'] = 1
    for current_frame_num in range(start_frame + 1, end_frame + 1):
        if session.get('stop_requested', False):
            logging.info(f"Tracking for {video_uuid} stopped by user request.")
            session['status'] = 'STOPPED'
            break
        frame_path = file_storage.get_frame_path(video_uuid, current_frame_num)
        if not os.path.exists(frame_path):
            logging.warning(f"Frame {current_frame_num} not found, skipping.")
            continue
        current_frame_bboxes_text_lines = []
        new_tracked_objects = {}
        prompts_bboxes = [obj_data['bbox'] for obj_id, obj_data in tracked_objects.items()]
        original_ids = list(tracked_objects.keys())
        if not prompts_bboxes:
            logging.warning(f"Lost all objects at frame {current_frame_num}. Stopping tracking.")
            break
        results = model(frame_path, bboxes=prompts_bboxes)
        if isinstance(results, Results):
            results = [results]
        if results and results[0].masks:
            new_masks = results[0].masks.data
            for i, new_mask in enumerate(new_masks):
                new_bbox = _get_bbox_from_mask(new_mask, original_width, original_height)
                if new_bbox:
                    original_id = original_ids[i]
                    label = tracked_objects[original_id]['label']
                    x1, y1, x2, y2 = new_bbox
                    current_frame_bboxes_text_lines.append(f"{x1},{y1},{x2},{y2},{label}")
                    new_tracked_objects[original_id] = {"label": label, "bbox": new_bbox}
        tracked_objects = new_tracked_objects
        session['results'][current_frame_num] = "\n".join(current_frame_bboxes_text_lines)
        session['progress'] = (current_frame_num - start_frame) + 1
    if 'status' not in session or session['status'] == 'PROCESSING':
        session['status'] = 'COMPLETED'


def run_batch_tracking_with_predictor(video_uuid, start_frame, end_frame, init_bboxes_text, session):
    if SAM2VideoPredictor is None:
        raise ImportError("SAM2VideoPredictor could not be imported. Please check your ultralytics installation.")
    if get_sam_model() is None:
        raise RuntimeError("SAM model is not available for batch tracking.")

    settings = settings_manager.load_settings()
    model_checkpoint_filename = settings.get('sam_model_checkpoint', 'sam2.1_t.pt')
    model_absolute_path = os.path.join(config.BASE_DIR, "checkpoints", model_checkpoint_filename)
    if not os.path.exists(model_absolute_path):
        raise FileNotFoundError(f"Batch tracking model not found at path: {model_absolute_path}")

    video_info = database.get_video_entity(video_uuid)
    width, height, fps = video_info['width'], video_info['height'], video_info['fps']
    if not fps or fps <= 0:
        fps = 30
        logging.warning(f"Video {video_uuid} has invalid FPS, falling back to {fps}.")

    init_rects, init_labels, _ = convert_text_to_rects_and_labels(init_bboxes_text)
    if not init_rects:
        raise ValueError("No initial bounding boxes provided for tracking.")

    all_frame_results = {start_frame: init_bboxes_text}
    last_known_rects = init_rects

    total_frames_to_process = end_frame - start_frame

    imgsz = settings.get('batch_tracking_imgsz', 1024)
    conf = settings.get('batch_tracking_conf', 0.30)
    chunk_size = settings.get('batch_tracking_chunk_size', 10)
    device = str(settings_manager.get_device())

    for i in range(0, total_frames_to_process, chunk_size):
        chunk_start_frame = start_frame + i
        chunk_end_frame = min(start_frame + i + chunk_size - 1, end_frame)

        if chunk_start_frame > end_frame:
            break

        logging.info(f"Processing chunk: frames {chunk_start_frame} to {chunk_end_frame}")

        temp_video_filename = f"temp_chunk_{uuid.uuid4().hex}.mp4"
        temp_video_path = os.path.join(config.STORAGE_DIR, 'videos', temp_video_filename)
        os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)

        video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not video_writer.isOpened():
            raise IOError(f"Failed to create temporary video writer for chunk {chunk_start_frame}-{chunk_end_frame}.")

        predictor = None
        try:
            for frame_num in range(chunk_start_frame, chunk_end_frame + 1):
                frame_path = file_storage.get_frame_path(video_uuid, frame_num)
                if os.path.exists(frame_path):
                    img = cv2.imread(frame_path)
                    if img.shape[1] != width or img.shape[0] != height:
                        img = cv2.resize(img, (width, height))
                    video_writer.write(img)
            video_writer.release()

            current_prompts = [[int((r[0] + r[2]) / 2), int((r[1] + r[3]) / 2)] for r in last_known_rects]
            labels_prompt = [1] * len(current_prompts)

            overrides = dict(
                conf=conf,
                task="segment",
                mode="predict",
                imgsz=imgsz,
                model=model_absolute_path,
                device=device
            )
            predictor = SAM2VideoPredictor(overrides=overrides)

            results_generator = predictor(source=temp_video_path, points=current_prompts, labels=labels_prompt,
                                          stream=True)

            latest_rects_in_chunk = None

            for frame_idx, results in enumerate(results_generator):
                actual_frame_num = chunk_start_frame + frame_idx
                session['progress'] = (actual_frame_num - start_frame)
                session['message'] = f'Processing frame {actual_frame_num}'

                if not results.masks:
                    all_frame_results[actual_frame_num] = ""
                    latest_rects_in_chunk = []
                    continue

                masks = results.masks.data
                bboxes_text_lines = []
                current_frame_rects = []

                for obj_idx in range(len(init_labels)):
                    if obj_idx < len(masks):
                        mask_data = masks[obj_idx]
                        bbox = _get_bbox_from_mask(mask_data, width, height)
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            bboxes_text_lines.append(f"{x1},{y1},{x2},{y2},{init_labels[obj_idx]}")
                            current_frame_rects.append(bbox)

                all_frame_results[actual_frame_num] = "\n".join(bboxes_text_lines)
                latest_rects_in_chunk = current_frame_rects

            if latest_rects_in_chunk and len(latest_rects_in_chunk) > 0:
                last_known_rects = latest_rects_in_chunk
            else:
                logging.warning("Lost all objects in chunk. Stopping batch processing.")
                break

        finally:
            if predictor is not None:
                del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            logging.info(f"Finished chunk, cleared predictor, cache, and temp file.")

    return all_frame_results