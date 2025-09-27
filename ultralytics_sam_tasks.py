# ultralytics_sam_tasks.py

import logging
import os
import torch
import numpy as np
import cv2

try:
    from ultralytics import SAM
    from ultralytics.engine.results import Results
except ImportError:
    logging.critical("FATAL: ultralytics library is not installed. Please run 'pip install ultralytics'.")
    SAM = None
    Results = None

import config
import database
import file_storage
from bbox_writer import parse_bboxes_text
import settings_manager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sam_model_cache = {"model": None, "path": None}


def get_sam_model():
    global _sam_model_cache

    settings = settings_manager.load_settings()
    checkpoint_filename = settings.get('sam_model_checkpoint', 'sam2.1_t.pt')
    sam_checkpoint_path = os.path.join(config.BASE_DIR, "checkpoints", checkpoint_filename)

    if _sam_model_cache["path"] != sam_checkpoint_path:
        logging.info(f"Model change detected. Reloading SAM model. New model: {checkpoint_filename}")
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

    mask = mask_data.cpu().numpy().astype(np.uint8) * 255
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

    init_rects, init_labels = convert_text_to_rects_and_labels(init_bboxes_text)
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
                tracked_objects[i] = {
                    "label": init_labels[i],
                    "bbox": bbox
                }

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

                    new_tracked_objects[original_id] = {
                        "label": label,
                        "bbox": new_bbox
                    }

        tracked_objects = new_tracked_objects
        session['results'][current_frame_num] = "\n".join(current_frame_bboxes_text_lines)
        session['progress'] = (current_frame_num - start_frame) + 1

    if 'status' not in session or session['status'] == 'PROCESSING':
        session['status'] = 'COMPLETED'