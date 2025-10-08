import logging
import os
import random
import shutil
import time
import traceback

import cv2
import numpy as np
import tensorflow as tf
import torch
import yaml

import ai_models
import config
import database
import file_storage
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


def apply_prototypes_to_video_task(video_uuid, class_name, negative_samples, confidence_threshold, app_context):
    """
    (最终优化版)
    在后台为视频应用原型，生成 AI 建议。
    - 接受用户自定义的置信度阈值。
    - 将带有分数的预测结果保存到 'suggested_bboxes_text' 数据库字段。
    """
    if active_tasks.get(video_uuid):
        logging.warning(f"Cannot start applying prototypes for {video_uuid}, another task is active.")
        return

    active_tasks[video_uuid] = 'APPLYING_PROTOTYPES'
    logging.info(
        f"Starting to apply suggestions for class '{class_name}' to video {video_uuid} with threshold {confidence_threshold}")

    try:
        with app_context:
            database.update_video_status(video_uuid, 'APPLYING_PROTOTYPES', f"Initializing for '{class_name}'...")

            database.update_video_status(video_uuid, 'APPLYING_PROTOTYPES',
                                         f"Building positive prototypes for '{class_name}'...")
            positive_prototypes = ai_models.build_prototypes_for_class(class_name)
            if positive_prototypes is None or len(positive_prototypes) == 0:
                raise ValueError(f"Could not build positive prototypes for class '{class_name}'.")
            logging.info(f"Successfully built {len(positive_prototypes)} positive prototypes for '{class_name}'.")

            negative_prototypes = None
            if negative_samples:
                database.update_video_status(video_uuid, 'APPLYING_PROTOTYPES', "Building negative prototypes...")
                negative_prototypes = ai_models.get_prototypes_from_drawn_boxes(negative_samples)
                if negative_prototypes is not None and len(negative_prototypes) > 0:
                    logging.info(
                        f"Successfully built {len(negative_prototypes)} negative prototypes from user samples.")
                else:
                    logging.warning("User provided negative samples, but failed to build prototypes from them.")

            all_frames = database.get_video_frames(video_uuid)
            unlabeled_frames = [f for f in all_frames if not f['bboxes_text'].strip()]
            total_frames = len(unlabeled_frames)
            logging.info(f"Found {total_frames} unlabeled frames to process in video {video_uuid}.")

            for i, frame_info in enumerate(unlabeled_frames):
                frame_number = frame_info['frame_number']
                current_status = database.get_video_entity(video_uuid)['status']
                if current_status == 'CANCELLING':
                    logging.info(f"Task for {video_uuid} cancelled by user.")
                    database.update_video_status(video_uuid, 'READY', 'Task was cancelled.')
                    return

                database.update_video_status(video_uuid, 'APPLYING_PROTOTYPES',
                                             f"Processing frame {i + 1}/{total_frames}")

                try:
                    # <<<<<<<<<<<<<<< 核心修改 1: 传递阈值给 AI 模型 >>>>>>>>>>>>>>>>>
                    predictions = ai_models.predict_with_prototypes(
                        video_uuid, frame_number, positive_prototypes,
                        negative_prototypes=negative_prototypes,
                        confidence_threshold=confidence_threshold
                    )

                    # <<<<<<<<<<<<<<< 核心修改 2: 保存带有分数的结果 >>>>>>>>>>>>>>>>>
                    # AI 模型返回的结果已经是被阈值过滤过的，直接使用即可
                    if predictions:
                        # 新格式: x1,y1,x2,y2,label,score
                        suggested_text = "\n".join(
                            [
                                f"{int(p['box'][0])},{int(p['box'][1])},{int(p['box'][2])},{int(p['box'][3])},{class_name},{p['score']:.4f}"
                                for p in predictions])
                        # 调用数据库函数将建议保存到 dedicated 字段
                        database.save_frame_suggestions(video_uuid, frame_number, suggested_text)

                except Exception as frame_e:
                    logging.error(f"Failed to process frame {frame_number} for {video_uuid}: {frame_e}")

                cache_key = f"{video_uuid}_{frame_number}"
                if cache_key in ai_models.PREPROCESSED_DATA_CACHE:
                    del ai_models.PREPROCESSED_DATA_CACHE[cache_key]

            # <<<<<<<<<<<<<<< 核心修改 3: 更新最终状态信息 >>>>>>>>>>>>>>>>>
            database.update_video_status(video_uuid, 'READY',
                                         f"Finished applying '{class_name}' suggestions. Review suggestions.")
            logging.info(f"Task for {video_uuid} completed successfully.")

    except Exception as e:
        error_message = f"Failed to apply prototypes to video {video_uuid}"
        logging.error(f"{error_message}: {e}")
        logging.error(traceback.format_exc())
        database.update_video_status(video_uuid, status="FAILED", message=str(e))
    finally:
        if active_tasks.get(video_uuid) == 'APPLYING_PROTOTYPES':
            del active_tasks[video_uuid]


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


def extract_frames_task(video_uuid):
    if active_tasks.get(video_uuid) == 'EXTRACTING':
        logging.warning(f"Extraction for {video_uuid} is already running.")
        return

    active_tasks[video_uuid] = 'EXTRACTING'
    logging.info(f"Starting frame extraction for {video_uuid}")
    video_path = file_storage.get_video_path(video_uuid)

    try:
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Cannot open video file")

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid.get(cv2.CAP_PROP_FPS)

        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0 or frame_count > config.MAX_FRAMES_PER_VIDEO:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            while vid.grab():
                frame_count += 1
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if frame_count > config.MAX_FRAMES_PER_VIDEO:
            raise ValueError(f"Video has more than {config.MAX_FRAMES_PER_VIDEO} frames.")

        database.update_video_after_extraction_start(video_uuid, width, height, fps, frame_count)

        count = 0
        while True:
            success, frame = vid.read()
            if not success:
                break

            success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if success:
                file_storage.save_frame_image(video_uuid, count, buffer.tobytes())
                database.update_extracted_frame_count(video_uuid, count + 1)

            count += 1

        vid.release()
        database.update_video_status(video_uuid, 'READY')
        logging.info(f"Frame extraction for {video_uuid} completed successfully.")

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
        model_type = model_info['model_type']

        database.update_video_status(video_uuid, 'PRE_ANNOTATING', f"Using model: {model_info['description']}")
        database.update_pre_annotation_info(video_uuid, model_uuid, model_info['description'])

        model_path = file_storage.get_model_path(model_uuid)
        label_path = file_storage.get_label_file_path(model_uuid)

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        all_frames = database.get_video_frames(video_uuid)
        frames_to_process = []
        for frame_info in all_frames:
            if start_frame <= frame_info['frame_number'] <= end_frame:
                if merge_strategy == 'skip_labeled' and frame_info.get('bboxes_text', '').strip():
                    continue
                frames_to_process.append(frame_info)

        total_frames_to_process = len(frames_to_process)
        logging.info(f"Total frames to process after filtering: {total_frames_to_process}")

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
            imH, imW, _ = frame_img.shape
            frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)

            if model_type == 'float32':
                input_data = np.float32(input_data) / 255.0

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            scores_raw = interpreter.get_tensor(output_details[0]['index'])[0]
            boxes_raw = interpreter.get_tensor(output_details[1]['index'])[0]
            classes_raw = interpreter.get_tensor(output_details[3]['index'])[0]

            scores_details = output_details[0]
            if scores_details['dtype'] == np.uint8 and scores_details.get('quantization'):
                scale, zero_point = scores_details['quantization']
                scores = (np.float32(scores_raw) - zero_point) * scale
            else:
                scores = scores_raw

            boxes_details = output_details[1]
            if boxes_details['dtype'] == np.uint8 and boxes_details.get('quantization'):
                scale, zero_point = boxes_details['quantization']
                boxes = (np.float32(boxes_raw) - zero_point) * scale
            else:
                boxes = boxes_raw
            classes = classes_raw

            bboxes_text_lines = []
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

            final_bboxes_text = "\n".join(bboxes_text_lines)
            database.save_frame_bboxes(video_uuid, frame_info['frame_number'], final_bboxes_text)

        database.update_video_status(video_uuid, 'READY', "Pre-annotation complete")
        logging.info(f"Pre-annotation for {video_uuid} completed successfully.")

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


def build_augmentation_pipeline(options):
    if A is None: return None
    transforms = []
    if options.get('hflip', {}).get('enabled'):
        transforms.append(A.HorizontalFlip(p=options['hflip']['p']))
    if options.get('vflip', {}).get('enabled'):
        transforms.append(A.VerticalFlip(p=options['vflip']['p']))
    if options.get('rotate90', {}).get('enabled'):
        transforms.append(A.RandomRotate90(p=options['rotate90']['p']))
    if options.get('rotate', {}).get('enabled'):
        transforms.append(
            A.Rotate(limit=options['rotate']['limit'], p=options['rotate']['p'], border_mode=cv2.BORDER_CONSTANT,
                     value=0))
    if options.get('ssr', {}).get('enabled'):
        transforms.append(A.ShiftScaleRotate(shift_limit=options['ssr']['shift'], scale_limit=options['ssr']['scale'],
                                             rotate_limit=options['ssr']['rotate'], p=options['ssr']['p'],
                                             border_mode=cv2.BORDER_CONSTANT, value=0))
    if options.get('affine', {}).get('enabled'):
        limit = options['affine']['shear']
        transforms.append(
            A.Affine(shear={'x': (-limit, limit), 'y': (-limit, limit)}, p=options['affine']['p'], cval=0))
    if options.get('crop', {}).get('enabled'):
        transforms.append(A.RandomSizedBBoxSafeCrop(height=1024, width=1024, erosion_rate=0.2,
                                                    p=options['crop']['p']))

    if options.get('grayscale', {}).get('enabled'):
        transforms.append(A.ToGray(p=options['grayscale']['p']))
    if options.get('hsv', {}).get('enabled'):
        transforms.append(A.HueSaturationValue(hue_shift_limit=options['hsv']['h'], sat_shift_limit=options['hsv']['s'],
                                               val_shift_limit=options['hsv']['v'], p=options['hsv']['p']))
    if options.get('bc', {}).get('enabled'):
        transforms.append(
            A.RandomBrightnessContrast(brightness_limit=options['bc']['b'], contrast_limit=options['bc']['c'],
                                       p=options['bc']['p']))

    if options.get('blur', {}).get('enabled'):
        transforms.append(A.GaussianBlur(blur_limit=(3, options['blur']['limit']), p=options['blur']['p']))
    if options.get('noise', {}).get('enabled'):
        transforms.append(A.GaussNoise(var_limit=(10.0, options['noise']['limit']), p=options['noise']['p']))

    if options.get('cutout', {}).get('enabled'):
        transforms.append(
            BboxSafeCoarseDropout(max_holes=options['cutout']['holes'], max_height=options['cutout']['size'],
                                  max_width=options['cutout']['size'], fill_value=0, p=options['cutout']['p']))

    if not transforms: return None
    return A.Compose(transforms,
                     bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))


def process_frame_worker(args):
    frame_info, target_img_dir, target_lbl_dir, class_map, augmentation_options = args

    augment_pipeline = None
    is_augmented = frame_info.get("type") == "augmented"
    if is_augmented and augmentation_options and augmentation_options.get("enabled", False):
        augment_pipeline = build_augmentation_pipeline(augmentation_options)

    try:
        if is_augmented:
            base_filename = frame_info["augmented_id"]
        else:
            base_filename = f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"

        src_img_path = file_storage.get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])
        if not os.path.exists(src_img_path):
            logging.warning(f"源文件未找到，跳过: {src_img_path}")
            return None

        image = cv2.imread(src_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo_bboxes, class_indices = file_storage.get_yolo_bboxes(
            frame_info['bboxes_text'], frame_info['width'], frame_info['height'], class_map
        )

        if not yolo_bboxes:
            return None

        if is_augmented and augment_pipeline:
            transformed = augment_pipeline(image=image, bboxes=yolo_bboxes, class_labels=class_indices)
            image_aug_rgb = transformed['image']
            bboxes_aug_yolo_tuples = transformed['bboxes']
            labels_aug_indices = transformed['class_labels']
            bboxes_aug_yolo = [(labels_aug_indices[i], *box) for i, box in enumerate(bboxes_aug_yolo_tuples)]
        else:
            image_aug_rgb = image
            bboxes_aug_yolo = [(class_indices[i], *box) for i, box in enumerate(yolo_bboxes)]

        final_image_bgr = cv2.cvtColor(image_aug_rgb, cv2.COLOR_RGB2BGR)
        output_image_path = os.path.join(target_img_dir, base_filename + '.jpg')
        cv2.imwrite(output_image_path, final_image_bgr)

        yolo_content_lines = [f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for class_id, x, y, w, h in bboxes_aug_yolo]
        output_label_path = os.path.join(target_lbl_dir, base_filename + '.txt')
        with open(output_label_path, 'w') as f:
            f.write("\n".join(yolo_content_lines))

        return output_image_path

    except Exception as e:
        logging.error(f"处理帧 {frame_info.get('augmented_id') or frame_info.get('frame_number')} 时发生错误: {e}")
        logging.error(traceback.format_exc())
        return None


def create_dataset_task(dataset_uuid, video_uuids, eval_percent, test_percent, augmentation_options=None):
    if augmentation_options is None:
        augmentation_options = {}

    logging.info(f"Starting dataset creation task for UUID: {dataset_uuid} with augmentations: {augmentation_options}")
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
            all_video_frames = database.get_video_frames(video_uuid)
            for frame in all_video_frames:
                if frame.get('bboxes_text') and frame['bboxes_text'].strip():
                    frames_to_include.append({
                        "video_uuid": video_uuid, "frame_number": frame['frame_number'],
                        "bboxes_text": frame['bboxes_text'], "width": video['width'], "height": video['height']
                    })
                    labels_in_frame = extract_labels(frame['bboxes_text'])
                    for label in labels_in_frame: all_labels.add(label)

        if not frames_to_include:
            raise ValueError("No labeled frames with valid bounding boxes were found in the selected videos.")

        sorted_labels = sorted(list(all_labels))
        class_map = {name: i for i, name in enumerate(sorted_labels)}
        logging.info(f"Dataset classes (sorted): {sorted_labels}")

        is_aug_enabled = A is not None and augmentation_options.get("enabled", False)
        multiplication_factor = int(augmentation_options.get("multiply_factor", 1)) if is_aug_enabled else 1
        final_frames_to_process = []
        if is_aug_enabled and multiplication_factor > 1:
            for frame_info in frames_to_include:
                final_frames_to_process.append({"type": "original", **frame_info})
                for i in range(multiplication_factor - 1):
                    aug_id = f"aug_{i}_{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"
                    final_frames_to_process.append({"type": "augmented", "augmented_id": aug_id, **frame_info})
        else:
            final_frames_to_process = [{"type": "original", **frame_info} for frame_info in frames_to_include]

        random.shuffle(final_frames_to_process)
        total_count = len(final_frames_to_process)
        val_count = int(total_count * eval_percent / 100.0)
        test_count = int(total_count * test_percent / 100.0)

        val_data = final_frames_to_process[:val_count]
        test_data = final_frames_to_process[val_count:val_count + test_count]
        train_data = final_frames_to_process[val_count + test_count:]

        dataset_dir = file_storage.get_dataset_dir(dataset_uuid)
        if os.path.exists(dataset_dir): shutil.rmtree(dataset_dir)
        dir_map = {
            'train': (os.path.join(dataset_dir, 'images', 'train'), os.path.join(dataset_dir, 'labels', 'train')),
            'val': (os.path.join(dataset_dir, 'images', 'val'), os.path.join(dataset_dir, 'labels', 'val')),
            'test': (os.path.join(dataset_dir, 'images', 'test'), os.path.join(dataset_dir, 'labels', 'test')),
        }
        for img_dir, lbl_dir in dir_map.values():
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
        all_tasks = []
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            img_dir, lbl_dir = dir_map[split_name]
            for frame_info in split_data:
                all_tasks.append((frame_info, img_dir, lbl_dir, class_map, augmentation_options))

        database.update_dataset_status(dataset_uuid, status="PROCESSING",
                                       message=f"Processing {len(all_tasks)} images across {cpu_count()} CPU cores...")
        logging.info(f"Starting parallel processing of {len(all_tasks)} images using up to {cpu_count()} cores.")

        processed_count = 0
        with Pool(processes=cpu_count()) as pool:
            for result in pool.imap_unordered(process_frame_worker, all_tasks):
                if result:
                    processed_count += 1
                    if processed_count % 50 == 0:
                        progress_msg = f"Processed {processed_count}/{len(all_tasks)} images..."
                        database.update_dataset_status(dataset_uuid, status="PROCESSING", message=progress_msg)

        logging.info(f"Parallel processing finished. Processed {processed_count} images successfully.")

        if yaml:
            yaml_content = {'path': f"../datasets/{dataset_uuid}", 'train': 'images/train', 'val': 'images/val',
                            'test': 'images/test', 'nc': len(sorted_labels), 'names': sorted_labels}
            with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
                yaml.dump(yaml_content, f, sort_keys=False)
        else:
            logging.error("PyYAML is not installed! Cannot create data.yaml for the dataset.")

        database.update_dataset_status(dataset_uuid, status="PROCESSING", message="Creating ZIP archive...")
        zip_path_base = os.path.join(config.STORAGE_DIR, 'datasets', dataset_uuid)
        zip_path = shutil.make_archive(zip_path_base, 'zip', dataset_dir)
        shutil.rmtree(dataset_dir)

        logging.info(f"ZIP archive created at: {zip_path}")
        database.update_dataset_status(dataset_uuid, status="READY", zip_path=zip_path, sorted_label_list=sorted_labels)
        logging.info(f"Dataset {dataset_uuid} task completed successfully.")

    except Exception as e:
        error_message = f"Failed to create dataset {dataset_uuid}"
        logging.error(f"{error_message}: {e}")
        logging.error(traceback.format_exc())
        database.update_dataset_status(dataset_uuid, status="FAILED", message=str(e))