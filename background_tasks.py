import cv2
import time
import logging
from datetime import datetime, timedelta, timezone
import os
import uuid
import numpy as np
import traceback
import tempfile
import shutil
import torch

import config
import database
import file_storage
from bbox_writer import extract_labels

try:
    import ultralytics_sam_tasks
except ImportError:
    ultralytics_sam_tasks = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

active_tasks = {}
tracking_sessions = {}


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
            f"Starting ULTRALYTICS SAM tracking for video {video_uuid} from frame {start_frame} to {end_frame}")
        session['status'] = 'PROCESSING'

        ultralytics_sam_tasks.track_video_ultralytics(
            video_uuid,
            start_frame,
            end_frame,
            init_bboxes_text,
            session
        )

        final_status = session.get('status', 'COMPLETED')
        logging.info(f"Ultralytics SAM tracking for {tracker_uuid} finished with status: {final_status}.")

    except Exception as e:
        logging.error(f"Error during Ultralytics SAM tracking for {video_uuid}: {e}\n{traceback.format_exc()}")
        session['status'] = 'FAILED'
        session['message'] = str(e)
    finally:
        logging.info(f"Cleaning up resources for Ultralytics SAM tracking task {tracker_uuid}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Emptied PyTorch CUDA cache.")

        if active_tasks.get(video_uuid) == tracker_uuid:
            del active_tasks[video_uuid]

        logging.info(f"Resource cleanup for task {tracker_uuid} complete.")


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


def start_tracking_task(video_uuid, tracker_uuid, tracker_name, scale, init_frame_number, init_bboxes_text):
    if active_tasks.get(video_uuid):
        logging.warning(f"A task (tracking/extraction) is already running for video {video_uuid}.")
        tracking_sessions[tracker_uuid] = {'status': 'FAILED', 'message': 'Another task is active.'}
        return

    active_tasks[video_uuid] = tracker_uuid
    video_path = file_storage.get_video_path(video_uuid)
    video_info = database.get_video_entity(video_uuid)

    tracker_fns = {
        'CSRT': cv2.legacy.TrackerCSRT_create,
        'MedianFlow': cv2.legacy.TrackerMedianFlow_create,
        'MIL': cv2.legacy.TrackerMIL_create,
        'MOSSE': cv2.legacy.TrackerMOSSE_create,
        'TLD': cv2.legacy.TrackerTLD_create,
        'KCF': cv2.legacy.TrackerKCF_create,
        'Boosting': cv2.legacy.TrackerBoosting_create,
    }

    try:
        logging.info(f"Starting tracking for video {video_uuid} with tracker {tracker_name}")
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Cannot open video file")

        vid.set(cv2.CAP_PROP_POS_FRAMES, init_frame_number)

        session = {
            'status': 'RUNNING',
            'current_frame': init_frame_number,
            'bboxes_text': init_bboxes_text,
            'last_client_update': time.time(),
            'stop_requested': False
        }
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
            session['bboxes_text'] = format_bboxes_text(
                new_bboxes, classes, scale, video_info['width'], video_info['height']
            )
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
        if active_tasks.get(video_uuid) == tracker_uuid:
            del active_tasks[video_uuid]
        if tracker_uuid in tracking_sessions and tracking_sessions[tracker_uuid]['status'] == 'RUNNING':
            tracking_sessions[tracker_uuid]['status'] = 'STOPPED'
        logging.info(
            f"Tracking task for {video_uuid} finished with status: {tracking_sessions.get(tracker_uuid, {}).get('status')}")


def create_dataset_task(dataset_uuid, video_uuids, eval_percent):
    logging.info(f"Starting dataset creation task for UUID: {dataset_uuid}")
    try:
        database.update_dataset_status(dataset_uuid, status="PROCESSING")

        frames_to_include = []
        all_labels = set()

        logging.info(f"Gathering frames from {len(video_uuids)} selected video(s)...")
        for video_uuid in video_uuids:
            logging.info(f"Processing video: {video_uuid}")
            video = database.get_video_entity(video_uuid)
            all_video_frames = database.get_video_frames(video_uuid)
            logging.info(f"Retrieved {len(all_video_frames)} total frame records from DB for this video.")

            labeled_frames_found_in_video = 0
            for frame in all_video_frames:
                if frame.get('bboxes_text') and frame['bboxes_text'].strip():
                    labeled_frames_found_in_video += 1
                    frames_to_include.append((
                        video_uuid,
                        frame['frame_number'],
                        frame['bboxes_text'],
                        video['width'],
                        video['height']
                    ))
                    labels_in_frame = extract_labels(frame['bboxes_text'])
                    for label in labels_in_frame:
                        all_labels.add(label)
            logging.info(
                f"Found and added {labeled_frames_found_in_video} labeled frames from this video to the dataset compilation.")

        logging.info(f"Total frames to be included in the final dataset: {len(frames_to_include)}.")
        if not frames_to_include:
            raise ValueError("No labeled frames with valid bounding boxes were found in the selected videos.")

        sorted_labels = sorted(list(all_labels))
        logging.info(f"Dataset classes (sorted): {sorted_labels}")

        logging.info(f"Creating YOLO ZIP archive for dataset {dataset_uuid}...")
        zip_path = file_storage.create_yolo_dataset_zip(
            dataset_uuid, frames_to_include, sorted_labels, eval_percent
        )
        logging.info(f"ZIP archive created at: {zip_path}")

        database.update_dataset_status(
            dataset_uuid, status="READY", zip_path=zip_path, sorted_label_list=sorted_labels
        )
        logging.info(f"Dataset {dataset_uuid} task completed successfully.")

    except Exception as e:
        error_message = f"Failed to create dataset {dataset_uuid}"
        logging.error(f"{error_message}: {e}")
        logging.error(traceback.format_exc())
        database.update_dataset_status(dataset_uuid, status="FAILED", message=str(e))