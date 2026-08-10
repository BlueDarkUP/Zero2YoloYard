import logging
import os
import random
import shutil
import time
import traceback
import uuid
import zipfile

import cv2
import numpy as np
import tensorflow as tf
import torch
import yaml

import ai_models
import config
import database
import file_storage
import settings_manager
from bbox_writer import extract_labels
from multiprocessing import Pool, cpu_count

try:
    import ultralytics_sam_tasks
except ImportError:
    ultralytics_sam_tasks = None

try:
    import albumentations as A


    class BboxSafeCoarseDropout(A.CoarseDropout):
        def apply_to_bbox(self, bbox, **params):
            return bbox

except ImportError:
    logging.warning(
        "albumentations library not found. Data augmentation will be disabled. Run 'pip install albumentations opencv-python-headless'")
    A = None

active_tasks = {}
tracking_sessions = {}
# 批量应用任务进度表（按 task_uuid 索引）
batch_apply_sessions = {}


def apply_class_to_videos_task(task_uuid, video_uuids, class_name, confidence_threshold, app_context, process_all_frames=True):
    """把某个类别的 SAM3 检索文本应用到一批视频的帧上，生成建议框。"""
    session = {
        'status': 'RUNNING',
        'video_uuids': video_uuids,
        'total_videos': len(video_uuids),
        'videos_done': 0,
        'current_video_uuid': None,
        'class_name': class_name,
        'message': f"Preparing to apply '{class_name}' across {len(video_uuids)} video(s)...",
        'error': None,
    }
    batch_apply_sessions[task_uuid] = session

    busy_videos = [vu for vu in video_uuids if active_tasks.get(vu)]
    if busy_videos:
        session['status'] = 'FAILED'
        session['message'] = f"以下视频当前有其它任务在运行，无法开始: {', '.join(v[:8] for v in busy_videos)}"
        return

    for vu in video_uuids:
        active_tasks[vu] = task_uuid

    logging.info(f"[{task_uuid}] Starting to apply '{class_name}' across {len(video_uuids)} video(s), "
                 f"threshold={confidence_threshold}, process_all_frames={process_all_frames}")

    try:
        with app_context:
            for v_idx, video_uuid in enumerate(video_uuids):
                session['current_video_uuid'] = video_uuid
                database.update_video_status(video_uuid, 'APPLYING_CLASS',
                                             f"Applying '{class_name}' ({v_idx + 1}/{len(video_uuids)} videos)...")

                all_frames = database.get_video_frames(video_uuid)
                if process_all_frames:
                    target_frames = all_frames
                else:
                    def _is_unlabeled(f):
                        if (f.get('bboxes_text') or '').strip():
                            return False
                        ann_json_str = (f.get('annotations_json') or '').strip()
                        if not ann_json_str:
                            return True
                        try:
                            from annotation_model import AnnotationData
                            ann_data = AnnotationData.from_json(ann_json_str)
                            if not ann_data.objects and not ann_data.classifications:
                                return True
                        except Exception:
                            pass
                        return False

                    target_frames = [f for f in all_frames if _is_unlabeled(f)]

                total_frames = len(target_frames)
                logging.info(f"[{task_uuid}] Video {video_uuid}: {total_frames} frames to process (all_frames={process_all_frames}).")

                for i, frame_info in enumerate(target_frames):
                    frame_number = frame_info['frame_number']

                    current_video_status = database.get_video_entity(video_uuid)['status']
                    if current_video_status == 'CANCELLING':
                        logging.info(f"[{task_uuid}] Cancelled by user while processing video {video_uuid}.")
                        database.update_video_status(video_uuid, 'READY', 'Task was cancelled.')
                        session['status'] = 'CANCELLED'
                        session['message'] = f"Cancelled while processing video {video_uuid[:8]}."
                        return

                    session['message'] = (
                        f"[{v_idx + 1}/{len(video_uuids)}] video {video_uuid[:8]}: frame {i + 1}/{total_frames}")
                    if i % 10 == 0 or i == total_frames - 1:
                        database.update_video_status(video_uuid, 'APPLYING_CLASS', session['message'])

                    try:
                        predictions = ai_models.predict_by_class_text(
                            video_uuid, frame_number, class_name, confidence_threshold=confidence_threshold
                        )
                        if predictions:
                            annotation_type = database.get_video_annotation_type(video_uuid)
                            if annotation_type == 'segmentation':
                                from annotation_model import AnnotationData, AnnotationObject
                                existing_ann_dict = database.get_frame_annotations(video_uuid, frame_number)
                                if existing_ann_dict:
                                    ann_data = AnnotationData.from_dict(existing_ann_dict)
                                else:
                                    ann_data = AnnotationData()

                                for j, p in enumerate(predictions):
                                    poly = p.get('polygon', [])
                                    if poly and len(poly) >= 3:
                                        pts = [[float(pt[0]), float(pt[1])] for pt in poly]
                                    else:
                                        box = p['box']
                                        x1, y1, x2, y2 = box
                                        pts = [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]

                                    obj_id = f"poly_{int(time.time()*1000)}_{j}"
                                    ann_data.objects.append(AnnotationObject(id=obj_id, type='polygon', label=class_name, points=pts))

                                database.save_frame_annotations(video_uuid, frame_number, ann_data.to_json())
                            elif annotation_type == 'classification':
                                from annotation_model import AnnotationData
                                existing_ann_dict = database.get_frame_annotations(video_uuid, frame_number)
                                if existing_ann_dict:
                                    ann_data = AnnotationData.from_dict(existing_ann_dict)
                                else:
                                    ann_data = AnnotationData()

                                if class_name not in ann_data.classifications:
                                    ann_data.classifications.append(class_name)

                                database.save_frame_annotations(video_uuid, frame_number, ann_data.to_json())
                            elif annotation_type == 'pose':
                                from annotation_model import AnnotationData, AnnotationObject
                                import gkdt_tasks
                                existing_ann_dict = database.get_frame_annotations(video_uuid, frame_number)
                                if existing_ann_dict:
                                    ann_data = AnnotationData.from_dict(existing_ann_dict)
                                else:
                                    ann_data = AnnotationData()

                                try:
                                    pose_objects = gkdt_tasks.predict_sam3_gkdt_batch_pose(
                                        video_uuid=video_uuid,
                                        frame_number=int(frame_number),
                                        class_label=class_name,
                                        confidence=confidence_threshold
                                    )
                                    if pose_objects:
                                        for p_obj in pose_objects:
                                            if isinstance(p_obj, dict):
                                                ann_data.objects.append(AnnotationObject.from_dict(p_obj))
                                            else:
                                                ann_data.objects.append(p_obj)
                                        database.save_frame_annotations(video_uuid, frame_number, ann_data.to_json())
                                except Exception as pose_e:
                                    logging.warning(f"[{task_uuid}] Failed to generate pose for frame {frame_number} of {video_uuid[:8]}: {pose_e}")
                            else:
                                lines = []
                                for p in predictions:
                                    box_str = f"{int(p['box'][0])},{int(p['box'][1])},{int(p['box'][2])},{int(p['box'][3])}"
                                    lines.append(f"{box_str},{class_name}")

                                existing_frame = database.get_frame_bboxes(video_uuid, frame_number)
                                if existing_frame and existing_frame.get('bboxes_text') and existing_frame['bboxes_text'].strip():
                                    combined = existing_frame['bboxes_text'].strip() + "\n" + "\n".join(lines)
                                else:
                                    combined = "\n".join(lines)

                                database.save_frame_bboxes(video_uuid, frame_number, combined)

                    except Exception as frame_e:
                        logging.error(f"[{task_uuid}] Failed to process frame {frame_number} of {video_uuid}: {frame_e}")

                database.update_video_status(video_uuid, 'READY',
                                             f"Finished applying '{class_name}'. Formal annotations generated.")
                if active_tasks.get(video_uuid) == task_uuid:
                    del active_tasks[video_uuid]
                session['videos_done'] += 1

            session['status'] = 'COMPLETED'
            session['message'] = f"Finished applying '{class_name}' across {len(video_uuids)} video(s)."
            logging.info(f"[{task_uuid}] Completed successfully.")

    except Exception as e:
        error_message = f"Failed to apply class '{class_name}' to videos"
        logging.error(f"[{task_uuid}] {error_message}: {e}")
        logging.error(traceback.format_exc())
        session['status'] = 'FAILED'
        session['error'] = str(e)
        session['message'] = str(e)
        for vu in video_uuids:
            try:
                database.update_video_status(vu, status="FAILED", message=str(e))
            except Exception:
                pass
    finally:
        for vu in video_uuids:
            if active_tasks.get(vu) == task_uuid:
                del active_tasks[vu]


def start_sam2_tracking_task(video_uuid, tracker_uuid, start_frame, end_frame, init_bboxes_text):
    if active_tasks.get(video_uuid):
        logging.warning(f"A task is already running for video {video_uuid}.")
        tracking_sessions[tracker_uuid] = {'status': 'FAILED', 'message': 'Another task is active.'}
        return

    if ultralytics_sam_tasks is None:
        logging.error("Ultralytics SAM Tasks module not available.")
        tracking_sessions[tracker_uuid] = {'status': 'FAILED',
                                           'message': 'Ultralytics library not installed or configured on server.'}
        return

    active_tasks[video_uuid] = tracker_uuid
    session = {
        'status': 'STARTING',
        'progress': 0,
        'total': (end_frame - start_frame) + 1,
        'results': {},
        'stop_requested': False,
        'message': ''
    }
    tracking_sessions[tracker_uuid] = session

    try:
        logging.info(
            f"Starting INTERACTIVE SAM tracking for video {video_uuid} from frame {start_frame} to {end_frame}")
        session['status'] = 'PROCESSING'

        ultralytics_sam_tasks.track_video_ultralytics(
            video_uuid,
            start_frame,
            end_frame,
            init_bboxes_text,
            session
        )

        final_status = session.get('status', 'COMPLETED')
        logging.info(f"Interactive SAM tracking for {tracker_uuid} finished with status: {final_status}.")

    except Exception as e:
        logging.error(f"Error during Interactive SAM tracking for {video_uuid}: {e}\n{traceback.format_exc()}")
        session['status'] = 'FAILED'
        session['message'] = str(e)
    finally:
        logging.info(f"Cleaning up resources for Interactive SAM tracking task {tracker_uuid}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Emptied PyTorch CUDA cache.")

        if active_tasks.get(video_uuid) == tracker_uuid:
            del active_tasks[video_uuid]

        logging.info(f"Resource cleanup for task {tracker_uuid} complete.")




def start_sam2_batch_tracking_task(video_uuid, tracker_uuid, start_frame, end_frame, init_bboxes_text):
    if active_tasks.get(video_uuid):
        logging.warning(f"A task is already running for video {video_uuid}.")
        tracking_sessions[tracker_uuid] = {'status': 'FAILED', 'message': 'Another task is active.'}
        return

    if ultralytics_sam_tasks is None:
        logging.error("Ultralytics SAM Tasks module not available for batch tracking.")
        tracking_sessions[tracker_uuid] = {'status': 'FAILED',
                                           'message': 'Ultralytics library not installed or configured on server.'}
        return

    active_tasks[video_uuid] = tracker_uuid
    session = {
        'status': 'BATCH_PROCESSING',
        'progress': 0,
        'total': (end_frame - start_frame) + 1,
        'results': {},
        'stop_requested': False,
        'message': 'Preparing temporary video clip...'
    }
    tracking_sessions[tracker_uuid] = session

    try:
        logging.info(
            f"Starting BATCH SAM tracking for video {video_uuid} from frame {start_frame} to {end_frame}")

        all_results = ultralytics_sam_tasks.run_batch_tracking_with_predictor(
            video_uuid,
            start_frame,
            end_frame,
            init_bboxes_text,
            session
        )

        session['results'] = all_results
        session['progress'] = session['total']
        session['status'] = 'COMPLETED'
        session['message'] = 'Batch processing complete. Ready for review.'
        logging.info(f"Batch SAM tracking for {tracker_uuid} finished successfully.")

    except Exception as e:
        logging.error(f"Error during Batch SAM tracking for {video_uuid}: {e}\n{traceback.format_exc()}")
        session['status'] = 'FAILED'
        session['message'] = str(e)
    finally:
        if active_tasks.get(video_uuid) == tracker_uuid:
            del active_tasks[video_uuid]
        logging.info(f"Batch tracking task for {tracker_uuid} cleaned up.")

def extract_frames_task(video_uuid, frame_interval=1):
    if active_tasks.get(video_uuid) == 'EXTRACTING':
        logging.warning(f"Extraction for {video_uuid} is already running.")
        return

    active_tasks[video_uuid] = 'EXTRACTING'
    logging.info(f"Starting frame extraction for {video_uuid} (Frame Interval: {frame_interval})")
    video_path = file_storage.get_video_path(video_uuid)

    try:
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Cannot open video file")

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        native_fps = vid.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0:
            native_fps = 30.0
        total_native_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_native_frames <= 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            total_native_frames = 0
            while vid.grab():
                total_native_frames += 1
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        interval = float(frame_interval) if frame_interval and float(frame_interval) >= 1.0 else 1.0
        skip_interval = interval
        actual_fps = native_fps / skip_interval

        frames_to_extract_indices = set()
        next_target = 0.0
        for i in range(total_native_frames):
            if i >= next_target:
                frames_to_extract_indices.add(i)
                next_target += skip_interval
        exact_frame_count = len(frames_to_extract_indices)
        if exact_frame_count > config.MAX_FRAMES_PER_VIDEO:
            raise ValueError(
                f"Extracted frames ({exact_frame_count}) exceed limit ({config.MAX_FRAMES_PER_VIDEO}). Please increase extraction interval.")
        database.update_video_after_extraction_start(video_uuid, width, height, actual_fps, exact_frame_count)
        settings = settings_manager.load_settings()
        jpeg_quality = int(settings.get('frame_extraction_jpeg_quality', 75))

        native_count = 0
        extracted_index = 0

        while True:
            success, frame = vid.read()
            if not success:
                break

            if native_count in frames_to_extract_indices:
                success_enc, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if success_enc:
                    file_storage.save_frame_image(video_uuid, extracted_index, buffer.tobytes())
                    if (extracted_index + 1) % 20 == 0 or (extracted_index + 1) == exact_frame_count:
                        database.update_extracted_frame_count(video_uuid, extracted_index + 1)
                    extracted_index += 1

            native_count += 1
        vid.release()
        database.update_video_status(video_uuid, 'READY')
        logging.info(
            f"Frame extraction for {video_uuid} completed. Kept {exact_frame_count} out of {total_native_frames} frames.")

    except Exception as e:
        logging.error(f"Error extracting frames for {video_uuid}: {e}")
        database.update_video_status(video_uuid, 'FAILED', str(e))
    finally:
        if active_tasks.get(video_uuid) == 'EXTRACTING':
            del active_tasks[video_uuid]


def pre_annotate_video_task(video_uuid, model_uuid, options):
    if active_tasks.get(video_uuid):
        logging.warning(f"Cannot start pre-annotation for {video_uuid}, another task is active.")
        return

    active_tasks[video_uuid] = 'PRE_ANNOTATING'
    logging.info(f"Starting pre-annotation for video {video_uuid} with options: {options}")

    try:
        confidence_threshold = options['confidence']
        start_frame = options['start_frame']
        end_frame = options['end_frame']
        merge_strategy = options['merge_strategy']

        video = database.get_video_entity(video_uuid)
        model_info = database.get_model_entity(model_uuid)
        model_type = model_info['model_type']  # float32, float16, int8, uint8 等

        database.update_video_status(video_uuid, 'PRE_ANNOTATING', f"Using model: {model_info['description']}")
        database.update_pre_annotation_info(video_uuid, model_uuid, model_info['description'])

        model_path = file_storage.get_model_path(model_uuid)
        label_path = file_storage.get_label_file_path(model_uuid)

        is_pt = model_path.endswith('.pt')
        is_tflite = model_path.endswith('.tflite')

        if not is_pt and not is_tflite:
            raise ValueError("Unsupported model file extension. Must be .tflite or .pt")

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]

        yolo_model = None
        interpreter = None
        input_details = None
        output_details = None
        height, width = 640, 640

        DEVICE = settings_manager.get_device()

        if is_pt:
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(model_path)

                yolo_model.to(DEVICE)

                if model_type == 'float16':
                    yolo_model.half()
                    logging.info("YOLO .pt model switched to FP16 (Half Precision) mode.")
                elif model_type in ['int8', 'uint8']:
                    logging.info("INT8 precision selected for YOLO .pt model.")

                if not labels and hasattr(yolo_model, 'names'):
                    labels = list(yolo_model.names.values())
                    logging.info(f"Auto-extracted {len(labels)} classes from YOLO .pt model metadata.")

            except Exception as e:
                raise ValueError(
                    f"Failed to load YOLO .pt model. Ensure it is a valid YOLOv8/v11 weights file. Details: {e}")

        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            height = input_details[0]['shape'][1]
            width = input_details[0]['shape'][2]
            if len(output_details) < 4:
                raise ValueError(
                    f"Unsupported TFLite format (Found {len(output_details)} outputs). "
                    "Standard TF Object Detection API models (4 outputs) are required for .tflite."
                )

            if model_type == 'float16':
                logging.info("TFLite model configured for FP16 inference.")

        annotation_type = video.get('annotation_type') or 'detection'

        all_frames = database.get_video_frames(video_uuid)
        frames_to_process = []
        for frame_info in all_frames:
            if start_frame <= frame_info['frame_number'] <= end_frame:
                if merge_strategy == 'skip_labeled':
                    if annotation_type == 'segmentation':
                        if frame_info.get('annotations_json') and frame_info['annotations_json'].strip():
                            continue
                    else:
                        if frame_info.get('bboxes_text') and frame_info['bboxes_text'].strip():
                            continue
                frames_to_process.append(frame_info)

        total_frames_to_process = len(frames_to_process)
        logging.info(f"Total frames to process after filtering: {total_frames_to_process}")

        from annotation_model import AnnotationData, AnnotationObject
        import time

        for i, frame_info in enumerate(frames_to_process):
            if i % 10 == 0:
                current_status = database.get_video_entity(video_uuid)['status']
                if current_status == 'CANCELLING':
                    logging.info(f"Pre-annotation for {video_uuid} cancelled by user.")
                    database.update_video_status(video_uuid, 'READY', 'Task was cancelled.')
                    return

            if (i + 1) % 20 == 0:
                progress_msg = f"Processed {i + 1}/{total_frames_to_process} frames"
                database.update_video_status(video_uuid, 'PRE_ANNOTATING', progress_msg)

            frame_path = file_storage.get_frame_path(video_uuid, frame_info['frame_number'])
            if not os.path.exists(frame_path):
                continue

            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                continue
            imH, imW, _ = frame_img.shape
            frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

            bboxes_text_lines = []
            ann_data = AnnotationData()
            has_masks = False

            if is_pt:
                results = yolo_model(frame_rgb, conf=confidence_threshold, verbose=False)
                if results and len(results) > 0:
                    res0 = results[0]
                    boxes_data = getattr(res0, 'boxes', None)
                    masks_data = getattr(res0, 'masks', None)
                    kpts_data = getattr(res0, 'keypoints', None)

                    has_masks = masks_data is not None and getattr(masks_data, 'xy', None) is not None and len(masks_data.xy) > 0
                    has_kpts = kpts_data is not None and getattr(kpts_data, 'xy', None) is not None and len(kpts_data.xy) > 0

                    if boxes_data is not None and len(boxes_data) > 0:
                        xyxy = boxes_data.xyxy.cpu().numpy()
                        conf = boxes_data.conf.cpu().numpy()
                        cls = boxes_data.cls.cpu().numpy()

                        kpts_xy = kpts_data.xy.cpu().numpy() if has_kpts else None
                        kpts_conf = getattr(kpts_data, 'conf', None) if has_kpts else None
                        kpts_conf_np = kpts_conf.cpu().numpy() if kpts_conf is not None and hasattr(kpts_conf, 'cpu') else None

                        coco_names = [
                            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist", "left_hip", "right_hip",
                            "left_knee", "right_knee", "left_ankle", "right_ankle"
                        ]

                        for j in range(len(xyxy)):
                            xmin = int(max(0, xyxy[j][0]))
                            ymin = int(max(0, xyxy[j][1]))
                            xmax = int(min(imW, xyxy[j][2]))
                            ymax = int(min(imH, xyxy[j][3]))
                            score = float(conf[j])
                            class_id = int(cls[j])
                            object_name = labels[class_id] if class_id < len(labels) else f"class_{class_id}"

                            bboxes_text_lines.append(f"{xmin},{ymin},{xmax},{ymax},{object_name}")

                            if has_kpts and j < len(kpts_xy):
                                instance_kpts = kpts_xy[j]
                                kpt_objects = []
                                for k_idx, pt in enumerate(instance_kpts):
                                    kx, ky = float(pt[0]), float(pt[1])
                                    kp_name = coco_names[k_idx] if k_idx < len(coco_names) else f"point_{k_idx+1}"
                                    kp_v = 2
                                    if kx == 0 and ky == 0:
                                        kp_v = 0
                                    elif kpts_conf_np is not None and k_idx < len(kpts_conf_np[j]):
                                        if float(kpts_conf_np[j][k_idx]) < 0.3:
                                            kp_v = 1
                                    kpt_objects.append({'name': kp_name, 'x': round(kx, 2), 'y': round(ky, 2), 'v': kp_v})

                                ann_data.objects.append(AnnotationObject(
                                    id=f"pose_{int(time.time()*1000)}_{j}",
                                    type='keypoint',
                                    label=object_name,
                                    bbox=[xmin, ymin, xmax, ymax],
                                    keypoints=kpt_objects
                                ))
                            elif has_masks and j < len(masks_data.xy):
                                pts = masks_data.xy[j]
                                poly_pts = []
                                if len(pts) >= 3:
                                    poly_pts = [[float(max(0, min(imW, pt[0]))), float(max(0, min(imH, pt[1])))] for pt in pts]
                                else:
                                    poly_pts = [[float(xmin), float(ymin)], [float(xmax), float(ymin)], [float(xmax), float(ymax)], [float(xmin), float(ymax)]]
                                ann_data.objects.append(AnnotationObject(
                                    id=f"poly_{int(time.time()*1000)}_{j}",
                                    type='polygon',
                                    label=object_name,
                                    points=poly_pts
                                ))
                            else:
                                ann_data.objects.append(AnnotationObject(
                                    id=f"bbox_{int(time.time()*1000)}_{j}",
                                    type='bbox',
                                    label=object_name,
                                    bbox=[xmin, ymin, xmax, ymax]
                                ))

            else:
                image_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(image_resized, axis=0)

                if model_type in ['float32', 'float16']:
                    input_data = np.float32(input_data) / 255.0

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                idx_scores, idx_boxes, idx_classes = 0, 1, 3
                for idx, detail in enumerate(output_details):
                    if len(detail['shape']) == 3 and detail['shape'][2] == 4:
                        idx_boxes = idx

                scores_raw = interpreter.get_tensor(output_details[idx_scores]['index'])[0]
                boxes_raw = interpreter.get_tensor(output_details[idx_boxes]['index'])[0]
                classes_raw = interpreter.get_tensor(output_details[idx_classes]['index'])[0]

                scores = (np.float32(scores_raw) - output_details[idx_scores]['quantization'][1]) * \
                         output_details[idx_scores]['quantization'][0] if output_details[idx_scores][
                                                                              'dtype'] == np.uint8 and output_details[
                                                                              idx_scores].get(
                    'quantization') else scores_raw
                boxes = (np.float32(boxes_raw) - output_details[idx_boxes]['quantization'][1]) * \
                        output_details[idx_boxes]['quantization'][0] if output_details[idx_boxes][
                                                                            'dtype'] == np.uint8 and output_details[
                                                                            idx_boxes].get(
                    'quantization') else boxes_raw
                classes = classes_raw

                for j in range(len(scores)):
                    if scores[j] > confidence_threshold:
                        ymin = int(max(0, boxes[j][0] * imH))
                        xmin = int(max(0, boxes[j][1] * imW))
                        ymax = int(min(imH, boxes[j][2] * imH))
                        xmax = int(min(imW, boxes[j][3] * imW))

                        object_id = int(classes[j])
                        if object_id < len(labels):
                            object_name = labels[object_id]
                            bboxes_text_lines.append(f"{xmin},{ymin},{xmax},{ymax},{object_name}")

                            poly_pts = [[float(xmin), float(ymin)], [float(xmax), float(ymin)], [float(xmax), float(ymax)], [float(xmin), float(ymax)]]
                            obj_id = f"poly_{int(time.time()*1000)}_{j}"
                            ann_data.objects.append(AnnotationObject(
                                id=obj_id,
                                type='polygon',
                                label=object_name,
                                points=poly_pts
                            ))

            if annotation_type == 'segmentation' or has_masks:
                json_str = ann_data.to_json()
                database.save_frame_annotations(video_uuid, frame_info['frame_number'], json_str)
            else:
                final_bboxes_text = "\n".join(bboxes_text_lines)
                database.save_frame_bboxes(video_uuid, frame_info['frame_number'], final_bboxes_text)

        database.update_video_status(video_uuid, 'READY', "Pre-annotation complete")
        logging.info(f"Pre-annotation for {video_uuid} completed successfully.")

    except ValueError as ve:
        logging.warning(f"Pre-annotation validation failed: {ve}")
        database.update_video_status(video_uuid, 'READY', f"FAILED: {str(ve)}")
    except Exception as e:
        logging.error(f"Error during pre-annotation for {video_uuid}: {e}", exc_info=True)
        database.update_video_status(video_uuid, 'READY', f"Pre-annotation failed: {e}")
    finally:
        if active_tasks.get(video_uuid) == 'PRE_ANNOTATING':
            del active_tasks[video_uuid]


def start_tracking_task(video_uuid, tracker_uuid, tracker_name, scale, init_frame_number, init_bboxes_text):
    if active_tasks.get(video_uuid):
        logging.warning(f"A task (tracking/extraction) is already running for video {video_uuid}.")
        tracking_sessions[tracker_uuid] = {'status': 'FAILED', 'message': 'Another task is active.'}
        return

    active_tasks[video_uuid] = tracker_uuid
    video_path = file_storage.get_video_path(video_uuid)
    video_info = database.get_video_entity(video_uuid)

    tracker_fns = {
        'CSRT': cv2.legacy.TrackerCSRT_create, 'MedianFlow': cv2.legacy.TrackerMedianFlow_create,
        'MIL': cv2.legacy.TrackerMIL_create, 'MOSSE': cv2.legacy.TrackerMOSSE_create,
        'TLD': cv2.legacy.TrackerTLD_create, 'KCF': cv2.legacy.TrackerKCF_create,
        'Boosting': cv2.legacy.TrackerBoosting_create,
    }

    try:
        logging.info(f"Starting tracking for video {video_uuid} with tracker {tracker_name}")
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened(): raise IOError("Cannot open video file")
        vid.set(cv2.CAP_PROP_POS_FRAMES, init_frame_number)
        session = {'status': 'RUNNING', 'current_frame': init_frame_number, 'bboxes_text': init_bboxes_text,
                   'last_client_update': time.time(), 'stop_requested': False}
        tracking_sessions[tracker_uuid] = session
        frame_number = init_frame_number
        trackers = None
        while not session['stop_requested']:
            success, frame = vid.read()
            if not success:
                session['status'] = 'COMPLETED'
                break
            if trackers is None or session['current_frame'] == frame_number:
                from bbox_writer import parse_bboxes_text
                bboxes, classes = parse_bboxes_text(session['bboxes_text'], scale)
                tracker_fn = tracker_fns[tracker_name]
                trackers = []
                for bbox in bboxes:
                    tracker = tracker_fn()
                    tracker.init(frame, tuple(bbox))
                    trackers.append(tracker)
            new_bboxes = []
            for tracker in trackers:
                ok, bbox = tracker.update(frame)
                new_bboxes.append(np.array(bbox) if ok else None)
            from bbox_writer import format_bboxes_text
            session['bboxes_text'] = format_bboxes_text(new_bboxes, classes, scale, video_info['width'],
                                                        video_info['height'])
            session['current_frame'] = frame_number
            while session['current_frame'] == frame_number and not session['stop_requested']:
                time.sleep(0.1)
                if time.time() - session['last_client_update'] > 60:
                    logging.warning(f"Tracking session {tracker_uuid} timed out.")
                    session['status'] = 'TIMED OUT'
                    session['stop_requested'] = True
            frame_number += 1
        vid.release()
    except Exception as e:
        logging.error(f"Error during tracking for {video_uuid}: {e}\n{traceback.format_exc()}")
        if tracker_uuid in tracking_sessions:
            tracking_sessions[tracker_uuid]['status'] = 'FAILED'
            tracking_sessions[tracker_uuid]['message'] = str(e)
    finally:
        if active_tasks.get(video_uuid) == tracker_uuid: del active_tasks[video_uuid]
        if tracker_uuid in tracking_sessions and tracking_sessions[tracker_uuid]['status'] == 'RUNNING':
            tracking_sessions[tracker_uuid]['status'] = 'STOPPED'
        logging.info(
            f"Tracking task for {video_uuid} finished with status: {tracking_sessions.get(tracker_uuid, {}).get('status')}")


# Note: Augmentation and process_frame_worker logic has been moved to exporters/detection/yolo_detect.py
from exporters.detection.yolo_detect import build_augmentation_pipeline, build_augmentation_pipeline_for_keypoints


def create_dataset_task(dataset_uuid, video_uuids, eval_percent, test_percent, export_format="yolo_v8_detect", augmentation_options=None, export_options=None):
    from exporters import ExporterRegistry
    import json
    from annotation_model import AnnotationData

    if augmentation_options is None:
        augmentation_options = {}
    if export_options is None:
        export_options = {}

    logging.info(f"Starting dataset creation task for UUID: {dataset_uuid} with format: {export_format}, augmentations: {augmentation_options}, export_options: {export_options}")
    try:
        if eval_percent is None: eval_percent = 20.0
        if test_percent is None: test_percent = 10.0
        if eval_percent + test_percent >= 100.0:
            raise ValueError(
                f"The sum of validation ({eval_percent}%) and test ({test_percent}%) percentages must be less than 100.")

        database.update_dataset_status(dataset_uuid, status="PROCESSING", message="Gathering labeled frames...")

        frames_to_include = []
        all_labels = set()
        logging.info(f"Gathering frames from {len(video_uuids)} selected video(s)...")
        for video_uuid in video_uuids:
            video = database.get_video_entity(video_uuid)
            all_video_frames = database.get_annotated_video_frames(video_uuid)
            for frame in all_video_frames:
                ann_data = None
                if frame.get('annotations_json'):
                    ann_data = AnnotationData.from_json(frame['annotations_json'])
                elif frame.get('bboxes_text'):
                    # Fallback for old data during transition
                    from annotation_model import AnnotationObject
                    from bbox_writer import parse_bboxes_text
                    ann_data = AnnotationData()
                    bboxes, classes = parse_bboxes_text(frame['bboxes_text'], 1.0)
                    for bbox, cls in zip(bboxes, classes):
                        ann_data.objects.append(AnnotationObject(id=str(uuid.uuid4()), type="bbox", label=cls, bbox=bbox))
                
                if ann_data and (ann_data.objects or ann_data.classifications):
                    frames_to_include.append({
                        "video_uuid": video_uuid, "frame_number": frame['frame_number'],
                        "annotations": ann_data, "width": video['width'], "height": video['height']
                    })
                    for label in ann_data.get_unique_labels():
                        all_labels.add(label)

        if not frames_to_include:
            raise ValueError("No labeled frames were found in the selected videos.")

        sorted_labels = sorted(list(all_labels))
        logging.info(f"Dataset classes (sorted): {sorted_labels}")

        dataset_dir = file_storage.get_dataset_dir(dataset_uuid)
        
        exporter = ExporterRegistry.get(export_format)
        exporter.export(
            export_dir=dataset_dir,
            frames_data=frames_to_include,
            class_list=sorted_labels,
            dataset_uuid=dataset_uuid,
            eval_percent=eval_percent,
            test_percent=test_percent,
            augmentation_options=augmentation_options,
            export_options=export_options,
            multilabel_strategy=export_options.get('multilabel_strategy', 'first')
        )

        database.update_dataset_status(dataset_uuid, status="PROCESSING", message="Creating ZIP archive...")
        zip_path_base = os.path.join(config.STORAGE_DIR, 'datasets', dataset_uuid)
        zip_path = f"{zip_path_base}.zip"
        
        settings = settings_manager.load_settings()
        zip_setting = settings.get('zip_compression', 'standard')
        compress_type = zipfile.ZIP_STORED if zip_setting == 'fast' else zipfile.ZIP_DEFLATED
        compress_level = 1 if zip_setting == 'fast' else (9 if zip_setting == 'max' else 6)

        with zipfile.ZipFile(zip_path, 'w', compression=compress_type, compresslevel=compress_level) as zipf:
            for root, _, files in os.walk(dataset_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dataset_dir)
                    zipf.write(file_path, arcname)

        shutil.rmtree(dataset_dir)

        # Execute auto-cleanup of orphan frames
        if settings.get('auto_cleanup_frames', False):
            database.update_dataset_status(dataset_uuid, status="PROCESSING", message="Cleaning up orphan frames...")
            cleaned_count = 0
            for v_uuid in video_uuids:
                all_v_frames = database.get_video_frames(v_uuid)
                for f_info in all_v_frames:
                    if not f_info.get('annotations_json') and not f_info.get('bboxes_text'):
                        f_path = file_storage.get_frame_path(v_uuid, f_info['frame_number'])
                        if os.path.exists(f_path):
                            os.remove(f_path)
                            cleaned_count += 1
            logging.info(f"Auto-cleanup finished. Deleted {cleaned_count} unlabeled frame files to free up space.")

        logging.info(f"ZIP archive created at: {zip_path}")
        database.update_dataset_status(dataset_uuid, status="READY", zip_path=zip_path, sorted_label_list=sorted_labels)
        logging.info(f"Dataset {dataset_uuid} task completed successfully.")

    except Exception as e:
        error_message = f"Failed to create dataset {dataset_uuid}"
        logging.error(f"{error_message}: {e}")
        logging.error(traceback.format_exc())
        database.update_dataset_status(dataset_uuid, status="FAILED", message=str(e))


def apply_pose_class_to_videos_task(task_uuid, video_uuids, class_name, confidence_threshold, app_context, process_all_frames=True):
    """把某个类别的 SAM3 + GKDT 开放姿态大模型推导应用到整套数据集的所有视频帧上"""
    session = {
        'status': 'RUNNING',
        'video_uuids': video_uuids,
        'total_videos': len(video_uuids),
        'videos_done': 0,
        'current_video_uuid': None,
        'class_name': class_name,
        'message': f"Preparing to apply TrueLAM Pose '{class_name}' across {len(video_uuids)} video(s)...",
        'error': None,
    }
    batch_apply_sessions[task_uuid] = session

    busy_videos = [vu for vu in video_uuids if active_tasks.get(vu)]
    if busy_videos:
        session['status'] = 'FAILED'
        session['message'] = f"以下视频当前有其它任务在运行，无法开始: {', '.join(v[:8] for v in busy_videos)}"
        return

    for vu in video_uuids:
        active_tasks[vu] = task_uuid

    logging.info(f"[{task_uuid}] Starting Pose SAM3+GKDT auto-labeling for '{class_name}' across {len(video_uuids)} video(s)...")

    try:
        with app_context:
            import gkdt_tasks
            from annotation_model import AnnotationData

            for v_idx, video_uuid in enumerate(video_uuids):
                session['current_video_uuid'] = video_uuid
                database.update_video_status(video_uuid, 'APPLYING_CLASS',
                                             f"Applying Pose '{class_name}' ({v_idx + 1}/{len(video_uuids)} videos)...")

                all_frames = database.get_video_frames(video_uuid)
                target_frames = all_frames if process_all_frames else [f for f in all_frames if not (f.get('annotations_json') or '').strip()]
                total_frames = len(target_frames)

                for i, frame_info in enumerate(target_frames):
                    frame_number = frame_info['frame_number']
                    session['message'] = f"[{v_idx + 1}/{len(video_uuids)}] Pose {video_uuid[:8]}: frame {i + 1}/{total_frames}"

                    if i % 5 == 0 or i == total_frames - 1:
                        database.update_video_status(video_uuid, 'APPLYING_CLASS', session['message'])

                    try:
                        pose_objects = gkdt_tasks.predict_sam3_gkdt_batch_pose(
                            video_uuid=video_uuid,
                            frame_number=int(frame_number),
                            class_label=class_name,
                            confidence=confidence_threshold
                        )

                        if pose_objects:
                            existing_ann_dict = database.get_frame_annotations(video_uuid, frame_number)
                            if existing_ann_dict:
                                ann_data = AnnotationData.from_dict(existing_ann_dict)
                            else:
                                ann_data = AnnotationData()

                            from annotation_model import AnnotationObject
                            for p_obj in pose_objects:
                                if isinstance(p_obj, dict):
                                    ann_data.objects.append(AnnotationObject.from_dict(p_obj))
                                else:
                                    ann_data.objects.append(p_obj)

                            database.save_frame_annotations(video_uuid, frame_number, ann_data.to_json())

                    except Exception as e:
                        logging.warning(f"[{task_uuid}] Pose auto-label error on frame {frame_number} of {video_uuid[:8]}: {e}")

                session['videos_done'] += 1
                database.update_video_status(video_uuid, 'READY', f"Pose '{class_name}' auto-labeling complete.")

        session['status'] = 'COMPLETED'
        session['message'] = f"Successfully auto-labeled pose '{class_name}' across all {len(video_uuids)} video(s)."

    except Exception as e:
        logging.error(f"[{task_uuid}] Pose batch apply failed: {e}", exc_info=True)
        session['status'] = 'FAILED'
        session['error'] = str(e)
        session['message'] = f"Pose auto-labeling failed: {e}"
    finally:
        for vu in video_uuids:
            if active_tasks.get(vu) == task_uuid:
                del active_tasks[vu]