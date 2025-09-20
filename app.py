import flask
from flask import Response, request, jsonify, render_template, send_from_directory, send_file
import os
import time
import json
import threading
import uuid
import logging
import cv2
import numpy as np
import io
from collections import Counter
from skimage import io as skio
from skimage.color import rgb2gray
import itertools  # Added for finding combinations of boxes

import settings_manager
import config
import database
import file_storage
import background_tasks
from bbox_writer import validate_bboxes_text, count_boxes, convert_text_to_rects_and_labels, extract_labels

try:
    import ultralytics_sam_tasks as sam_tasks
except ImportError:
    logging.warning("ultralytics_sam_tasks.py not found or failed to import. All SAM features will be disabled.")
    sam_tasks = None

app = flask.Flask(__name__)
app.secret_key = os.urandom(24)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

try:
    import yaml
except ImportError:
    logging.error("PyYAML is not installed! Dataset export will fail. Please run 'pip install pyyaml'.")
    yaml = None

with app.app_context():
    database.init_db()
    database.migrate_db()
    file_storage.init_storage()


def validate_description(desc, existing_descriptions):
    if not (1 <= len(desc) <= config.MAX_DESCRIPTION_LENGTH):
        return False, "Description must be between 1 and 30 characters."
    if desc in existing_descriptions:
        return False, "Description is a duplicate."
    return True, ""


def sanitize_dict(d):
    return d


def string_to_color_bgr(s):
    hash_val = 0
    for char in s:
        hash_val = ord(char) + ((hash_val << 5) - hash_val)
    hue = hash_val % 360
    color_hsv = np.uint8([[[hue, 200, 200]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, color_bgr))


def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


@app.route('/')
def index():
    return render_template('root.html',
                           limit_data=config.get_limit_data_for_render_template(),
                           tracker_fns=config.TRACKER_FNS)


@app.route('/labelVideo')
def label_video():
    task_uuid = request.args.get('task_uuid')
    if not task_uuid:
        return "Task UUID is required.", 400

    task_entity = database.get_task_entity(task_uuid)
    if not task_entity:
        return "Annotation task not found.", 404

    if task_entity['status'] == 'PENDING':
        database.update_task_status(task_uuid, 'IN_PROGRESS')
        task_entity = database.get_task_entity(task_uuid)

    video_entity = database.get_video_entity(task_entity['video_uuid'])
    if not video_entity:
        return "Associated video not found.", 404

    first_frame_url = f"/media/frames/{video_entity['video_uuid']}/frame_{task_entity['start_frame']:05d}.jpg"

    return render_template('labelVideo.html',
                           task_entity=sanitize_dict(task_entity),
                           video_entity=sanitize_dict(video_entity),
                           first_frame_url=first_frame_url,
                           limit_data=config.get_limit_data_for_render_template())


@app.route('/media/<path:path>')
def send_media(path):
    return send_from_directory(config.STORAGE_DIR, path)


@app.route('/media/annotated_frame/<video_uuid>/<int:frame_number>.jpg')
def serve_annotated_frame(video_uuid, frame_number):
    try:
        frame_path = file_storage.get_frame_path(video_uuid, frame_number)
        if not os.path.exists(frame_path):
            return "Frame not found", 404
        image = cv2.imread(frame_path)
        if image is None:
            return "Could not read frame image", 500

        conn = database.get_db_connection()
        frame_data = conn.execute(
            'SELECT bboxes_text FROM video_frames WHERE video_uuid = ? AND frame_number = ?',
            (video_uuid, frame_number)
        ).fetchone()
        conn.close()

        bboxes_text = frame_data['bboxes_text'] if frame_data else None

        if bboxes_text and bboxes_text.strip():
            rects, labels = convert_text_to_rects_and_labels(bboxes_text)
            for i, rect in enumerate(rects):
                label = labels[i]
                color = string_to_color_bgr(label)
                x1, y1, x2, y2 = rect
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return "Failed to encode image", 500

        return Response(buffer.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        logging.error(f"Error generating annotated frame for {video_uuid}/{frame_number}: {e}")
        return "Internal server error", 500


@app.route('/listVideos', methods=['GET'])
def list_videos():
    all_videos = database.get_all_video_list()
    ready_videos = database.get_ready_videos_with_labels()
    return jsonify({
        'all_videos': [sanitize_dict(v) for v in all_videos],
        'ready_videos_for_dataset': [sanitize_dict(v) for v in ready_videos]
    })


@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    desc = request.form.get('description')
    video_file = request.files.get('video_file')

    is_valid, message = validate_description(desc, [v['description'] for v in database.get_all_video_list()])
    if not is_valid:
        return jsonify({'success': False, 'message': message}), 400
    if not video_file:
        return jsonify({'success': False, 'message': 'No video file provided.'}), 400

    create_time_ms = int(time.time() * 1000)
    video_uuid = database.create_video_entry(desc, video_file.filename, 0, create_time_ms)
    file_storage.save_uploaded_video(video_file, video_uuid)

    threading.Thread(target=background_tasks.extract_frames_task, args=(video_uuid,),
                     name=f"Extractor-{video_uuid[:6]}").start()

    return jsonify({'success': True, 'video_uuid': video_uuid})


@app.route('/importFrames', methods=['POST'])
def import_frames():
    video_uuid = request.form.get('video_uuid')
    frame_files = request.files.getlist('frame_files')

    if not video_uuid or not frame_files:
        return jsonify({'success': False, 'message': 'Missing video UUID or frame files.'}), 400

    video = database.get_video_entity(video_uuid)
    if not video:
        return jsonify({'success': False, 'message': 'Video not found.'}), 404

    try:
        imported_count = database.add_frames_from_upload(video_uuid, frame_files)
        return jsonify({'success': True, 'imported_count': imported_count})
    except Exception as e:
        logging.error(f"Failed to import frames for {video_uuid}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/retrieveVideoEntity', methods=['POST'])
def retrieve_video_entity():
    video_uuid = request.json.get('video_uuid')
    entity = database.get_video_entity(video_uuid)
    if entity:
        return jsonify({'success': True, 'video_entity': sanitize_dict(entity)})
    return jsonify({'success': False, 'message': 'Video not found.'})


@app.route('/deleteVideo', methods=['POST'])
def delete_video():
    video_uuid = request.json.get('video_uuid')
    database.delete_video(video_uuid)
    file_storage.delete_video_file(video_uuid)
    file_storage.delete_frames_for_video(video_uuid)
    return jsonify({'success': True})


@app.route('/retrieveVideoFrames', methods=['POST'])
def retrieve_video_frames():
    video_uuid = request.json.get('video_uuid')
    frames = database.get_video_frames(video_uuid)
    for frame in frames:
        frame['image_url'] = f"/media/frames/{video_uuid}/frame_{frame['frame_number']:05d}.jpg"
    return jsonify({'success': True, 'frames': [sanitize_dict(f) for f in frames]})


@app.route('/storeVideoFrameBboxesText', methods=['POST'])
def store_video_frame_bboxes_text():
    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = int(data.get('frame_number'))
    bboxes_text = validate_bboxes_text(data.get('bboxes_text'))
    database.save_frame_bboxes(video_uuid, frame_number, bboxes_text)
    return jsonify({'success': True})


@app.route('/listTasks', methods=['GET'])
def list_tasks():
    video_uuid = request.args.get('video_uuid')
    if not video_uuid:
        return jsonify({'success': False, 'message': 'Video UUID is required.'}), 400
    tasks = database.get_tasks_for_video(video_uuid)
    return jsonify({'success': True, 'tasks': [sanitize_dict(t) for t in tasks]})


@app.route('/createTask', methods=['POST'])
def create_task():
    data = request.json
    video_uuid = data.get('video_uuid')
    assigned_to = data.get('assigned_to')
    description = data.get('description', '')
    start_frame = data.get('start_frame')
    end_frame = data.get('end_frame')

    if not all([video_uuid, assigned_to, start_frame is not None, end_frame is not None]):
        return jsonify({'success': False, 'message': 'Missing required fields.'}), 400

    try:
        start_frame, end_frame = int(start_frame), int(end_frame)
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'Frame numbers must be integers.'}), 400

    video = database.get_video_entity(video_uuid)
    if not video:
        return jsonify({'success': False, 'message': 'Video not found.'}), 404

    if not (0 <= start_frame < end_frame < video['frame_count']):
        return jsonify({'success': False,
                        'message': f'Invalid frame range. Must be within 0 and {video["frame_count"] - 1}.'}), 400

    try:
        task_uuid = database.create_annotation_task(video_uuid, assigned_to, description, start_frame, end_frame)
        return jsonify({'success': True, 'task_uuid': task_uuid})
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/deleteTask', methods=['POST'])
def delete_task():
    task_uuid = request.json.get('task_uuid')
    database.delete_task(task_uuid)
    return jsonify({'success': True})


@app.route('/updateTaskStatus', methods=['POST'])
def update_task_status():
    data = request.json
    task_uuid = data.get('task_uuid')
    status = data.get('status')
    if not task_uuid or status not in ['PENDING', 'IN_PROGRESS', 'COMPLETED']:
        return jsonify({'success': False, 'message': 'Invalid task UUID or status.'}), 400
    database.update_task_status(task_uuid, status)
    return jsonify({'success': True})


@app.route('/listClasses', methods=['GET'])
def list_classes():
    labels = database.get_all_class_labels()
    return jsonify({'success': True, 'labels': labels})


@app.route('/addClass', methods=['POST'])
def add_class():
    data = request.json
    label_name = data.get('label_name', '').strip()
    if not label_name:
        return jsonify({'success': False, 'message': 'Label name cannot be empty.'}), 400
    database.add_class_label(label_name)
    return jsonify({'success': True})


@app.route('/deleteClass', methods=['POST'])
def delete_class():
    data = request.json
    label_name = data.get('label_name')
    if not label_name:
        return jsonify({'success': False, 'message': 'Label name is required.'}), 400
    database.delete_class_label(label_name)
    return jsonify({'success': True})


@app.route('/api/settings', methods=['GET'])
def get_settings():
    settings = settings_manager.load_settings()
    return jsonify({'success': True, 'settings': settings})


@app.route('/api/settings', methods=['POST'])
def save_settings():
    new_settings = request.json
    if not new_settings:
        return jsonify({'success': False, 'message': 'No settings data provided.'}), 400

    current_settings = settings_manager.load_settings()

    sam_model_changed = current_settings.get('sam_model_checkpoint') != new_settings.get('sam_model_checkpoint')

    if settings_manager.save_settings(new_settings):
        if sam_model_changed:
            logging.info("SAM model setting changed. The model will be reloaded on the next request.")
            if sam_tasks:
                sam_tasks._sam_model_cache["model"] = None
                sam_tasks._sam_model_cache["path"] = None

        return jsonify({
            'success': True,
            'message': 'Settings saved successfully!',
            'restart_required': sam_model_changed
        })
    else:
        return jsonify({'success': False, 'message': 'Failed to save settings to file.'}), 500


@app.route('/samPredict', methods=['POST'])
def sam_predict():
    if not sam_tasks or not sam_tasks.get_sam_model():
        return jsonify(
            {'success': False, 'message': 'Ultralytics SAM model is not available or not installed on server.'}), 501

    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = data.get('frame_number')
    point_coords = data.get('point')

    if not all([video_uuid, frame_number is not None, point_coords]):
        return jsonify({'success': False, 'message': 'Missing required data (video_uuid, frame_number, point).'}), 400

    try:
        image_path = file_storage.get_frame_path(video_uuid, int(frame_number))
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'message': 'Frame image not found on server.'}), 404

        coords_tuple = (int(point_coords['x']), int(point_coords['y']))

        bbox = sam_tasks.predict_box_from_point_ultralytics(image_path, coords_tuple)

        if bbox:
            return jsonify({'success': True, 'bbox': bbox})
        else:
            return jsonify({'success': False, 'message': 'No object found at the specified point.'})

    except Exception as e:
        logging.error(f"SAM prediction failed: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/startSam2Tracking', methods=['POST'])
def start_sam2_tracking():
    if not sam_tasks:
        return jsonify(
            {'success': False, 'message': 'SAM video tracking feature is not available on the server.'}), 501

    data = request.json
    video_uuid = data.get('video_uuid')
    start_frame = int(data.get('start_frame'))
    end_frame = int(data.get('end_frame'))
    init_bboxes_text = data.get('init_bboxes_text')

    if not all([video_uuid, start_frame is not None, end_frame is not None, init_bboxes_text]):
        return jsonify({'success': False, 'message': 'Missing required data for tracking.'}), 400

    if background_tasks.active_tasks.get(video_uuid):
        return jsonify(
            {'success': False, 'message': 'Another task (extraction or tracking) is already running for this video.'})

    tracker_uuid = str(uuid.uuid4().hex)

    threading.Thread(target=background_tasks.start_sam2_tracking_task, args=(
        video_uuid, tracker_uuid, start_frame, end_frame, init_bboxes_text
    ), name=f"SAM-Tracker-{video_uuid[:6]}").start()

    return jsonify({'success': True, 'tracker_uuid': tracker_uuid})


@app.route('/streamSam2Tracking/<tracker_uuid>')
def stream_sam2_tracking(tracker_uuid):
    def generate_events():
        while tracker_uuid not in background_tasks.tracking_sessions:
            time.sleep(0.1)

        session = background_tasks.tracking_sessions.get(tracker_uuid)
        if not session:
            error_event = {"event": "error", "message": "Tracking session not found or failed to start."}
            yield f"data: {json.dumps(error_event)}\n\n"
            return

        last_sent_frame = -1
        try:
            while True:
                status = session.get('status', 'STARTING')
                sorted_frames = sorted([k for k in session.get('results', {}).keys() if k > last_sent_frame])

                for frame_num in sorted_frames:
                    result_data = {
                        "event": "update",
                        "frame_number": frame_num,
                        "bboxes_text": session['results'][frame_num],
                        "progress": session.get('progress', 0),
                        "total": session.get('total', 0)
                    }
                    yield f"data: {json.dumps(result_data)}\n\n"
                    last_sent_frame = frame_num

                if status in ['COMPLETED', 'STOPPED', 'FAILED']:
                    final_event = {"event": status.lower(), "message": session.get('message', '')}
                    yield f"data: {json.dumps(final_event)}\n\n"
                    break

                time.sleep(0.2)
        except GeneratorExit:
            logging.info(f"Client disconnected from SSE stream for tracker {tracker_uuid}")
        finally:
            logging.info(f"SSE stream for tracker {tracker_uuid} is closing.")

    return Response(generate_events(), mimetype='text/event-stream')


@app.route('/stopSam2Tracking', methods=['POST'])
def stop_sam2_tracking():
    tracker_uuid = request.json.get('tracker_uuid')
    if tracker_uuid in background_tasks.tracking_sessions:
        session = background_tasks.tracking_sessions[tracker_uuid]
        session['stop_requested'] = True
        logging.info(f"Stop request received for SAM tracking session {tracker_uuid}")
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Tracker not found.'})


@app.route('/prepareToStartTracking', methods=['POST'])
def prepare_to_start_tracking():
    data = request.json
    video_uuid = data.get('video_uuid')
    if background_tasks.active_tasks.get(video_uuid):
        return jsonify(
            {'success': False, 'message': 'Another task (extraction or tracking) is already running for this video.'})
    tracker_uuid = str(uuid.uuid4().hex)
    threading.Thread(target=background_tasks.start_tracking_task, args=(
        video_uuid, tracker_uuid, data.get('tracker_name'),
        float(data.get('scale')), int(data.get('init_frame_number')),
        data.get('init_bboxes_text'),
    ), name=f"Tracker-{video_uuid[:6]}").start()
    return jsonify({'success': True, 'tracker_uuid': tracker_uuid})


@app.route('/retrieveTrackedBboxes', methods=['POST'])
def retrieve_tracked_bboxes():
    tracker_uuid = request.json.get('tracker_uuid')
    session = background_tasks.tracking_sessions.get(tracker_uuid)
    if session:
        session['last_client_update'] = time.time()
        return jsonify({
            'success': True, 'tracker_failed': session['status'] in ['FAILED', 'TIMED OUT'],
            'frame_number': session.get('current_frame'), 'bboxes_text': session.get('bboxes_text'),
        })
    return jsonify({'success': False, 'tracker_failed': True})


@app.route('/continueTracking', methods=['POST'])
def continue_tracking():
    data = request.json
    tracker_uuid = data.get('tracker_uuid')
    session = background_tasks.tracking_sessions.get(tracker_uuid)
    if session and session['status'] == 'RUNNING':
        session['last_client_update'] = time.time()
        session['bboxes_text'] = data.get('bboxes_text')
        session['current_frame'] = int(data.get('frame_number')) + 1
        database.save_frame_bboxes(data.get('video_uuid'), int(data.get('frame_number')), data.get('bboxes_text'))
        return jsonify({'success': True})
    return jsonify({'success': False})


@app.route('/stopTracking', methods=['POST'])
def stop_tracking():
    tracker_uuid = request.json.get('tracker_uuid')
    if tracker_uuid in background_tasks.tracking_sessions:
        background_tasks.tracking_sessions[tracker_uuid]['stop_requested'] = True
    return jsonify({'success': True})


@app.route('/listDatasets', methods=['GET'])
def list_datasets():
    datasets = database.get_dataset_list()
    return jsonify({'datasets': [sanitize_dict(d) for d in datasets]})


@app.route('/createDataset', methods=['POST'])
def create_dataset():
    data = request.json
    desc = data.get('description')
    video_uuids = data.get('video_uuids')
    eval_percent = float(data.get('eval_percent', 20.0))
    test_percent = float(data.get('test_percent', 10.0))

    is_valid, message = validate_description(desc, [d['description'] for d in database.get_dataset_list()])
    if not is_valid:
        return jsonify({'success': False, 'message': message}), 400
    if not video_uuids:
        return jsonify({'success': False, 'message': 'Please select at least one video.'}), 400

    create_time = int(time.time() * 1000)
    dataset_uuid = database.create_dataset_entry(desc, video_uuids, create_time, eval_percent, test_percent)

    threading.Thread(target=background_tasks.create_dataset_task, args=(
        dataset_uuid, video_uuids, eval_percent, test_percent
    ), name=f"Dataset-{dataset_uuid[:6]}").start()

    return jsonify({'success': True, 'dataset_uuid': dataset_uuid})


@app.route('/regenerateDataset', methods=['POST'])
def regenerate_dataset():
    dataset_uuid = request.json.get('dataset_uuid')
    if not dataset_uuid:
        return jsonify({'success': False, 'message': 'Dataset UUID is required.'}), 400

    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': 'Dataset not found.'}), 404

    file_storage.delete_dataset_files(dataset_uuid)
    database.update_dataset_status(dataset_uuid, 'PENDING')
    video_uuids = json.loads(dataset['video_uuids'])
    eval_percent = dataset.get('eval_percent')
    test_percent = dataset.get('test_percent')

    threading.Thread(target=background_tasks.create_dataset_task, args=(
        dataset_uuid, video_uuids, eval_percent, test_percent
    ), name=f"Dataset-Regen-{dataset_uuid[:6]}").start()

    return jsonify({'success': True, 'message': 'Dataset regeneration started.'})


@app.route('/downloadDataset/<dataset_uuid>')
def download_dataset(dataset_uuid):
    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset or dataset['status'] != 'READY' or not dataset['zip_path']:
        return "Dataset not found or not ready.", 404

    try:
        return send_file(dataset['zip_path'], as_attachment=True)
    except Exception as e:
        logging.error(f"Could not send file: {e}")
        return "Error downloading file.", 500


@app.route('/deleteDataset', methods=['POST'])
def delete_dataset():
    dataset_uuid = request.json.get('dataset_uuid')
    database.delete_dataset(dataset_uuid)
    file_storage.delete_dataset_files(dataset_uuid)
    return jsonify({'success': True})


@app.route('/listModels', methods=['GET'])
def list_models():
    models = database.get_model_list()
    return jsonify({'models': [sanitize_dict(m) for m in models]})


@app.route('/importModel', methods=['POST'])
def import_model():
    desc = request.form.get('description')
    model_file = request.files.get('model_file')
    is_valid, message = validate_description(desc, [m['description'] for m in database.get_model_list()])
    if not is_valid:
        return jsonify({'success': False, 'message': message}), 400
    if not model_file or not model_file.filename.endswith('.tflite'):
        return jsonify({'success': False, 'message': 'Please provide a .tflite file.'}), 400
    create_time = int(time.time() * 1000)
    model_uuid = database.import_model_metadata(desc, create_time)
    file_storage.save_imported_model(model_file, model_uuid)
    return jsonify({'success': True, 'model_uuid': model_uuid})


@app.route('/deleteModel', methods=['POST'])
def delete_model():
    model_uuid = request.json.get('model_uuid')
    database.delete_model(model_uuid)
    file_storage.delete_model_file(model_uuid)
    return jsonify({'success': True})


@app.route('/datasetAnalysis/<dataset_uuid>')
def dataset_analysis(dataset_uuid):
    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset:
        return "Dataset not found", 404
    return render_template('dataset_analysis.html', dataset=sanitize_dict(dataset))


@app.route('/api/datasetAnalysis/<dataset_uuid>', methods=['GET'])
def get_dataset_analysis_data(dataset_uuid):
    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': 'Dataset not found.'}), 404

    video_uuids = json.loads(dataset.get('video_uuids', '[]'))

    tasks_by_video = {vu: database.get_tasks_for_video(vu) for vu in video_uuids}
    video_info_cache = {vu: database.get_video_entity(vu) for vu in video_uuids}

    def get_task_for_frame(video_uuid, frame_number):
        for task in tasks_by_video.get(video_uuid, []):
            if task['start_frame'] <= frame_number <= task['end_frame']:
                return task['task_uuid']
        return None

    all_frames = [
        {**dict(frame), 'video_uuid': vu, 'video_description': video_info_cache[vu].get('description')}
        for vu in video_uuids
        for frame in database.get_video_frames(vu)
        if frame.get('bboxes_text', '').strip()
    ]

    class_counts = Counter()
    aspect_ratios, objects_per_image, center_points, brightness_levels = [], [], [], []
    all_bboxes_for_outliers = []
    suspicious_pairs = []
    image_class_map = {}

    for i, frame in enumerate(all_frames):
        video_uuid, frame_number, bboxes_text = frame['video_uuid'], frame['frame_number'], frame['bboxes_text']
        rects, labels = convert_text_to_rects_and_labels(bboxes_text)

        image_class_map[i] = list(set(labels))
        objects_per_image.append(len(labels))

        if len(rects) > 1:
            for (idx1, rect1), (idx2, rect2) in itertools.combinations(enumerate(rects), 2):
                iou = calculate_iou(rect1, rect2)
                if iou > 0.95:
                    suspicious_pairs.append({
                        'image_index': i, 'iou': iou,
                        'box1_label': labels[idx1], 'box2_label': labels[idx2]
                    })
        try:
            image_gray = skio.imread(file_storage.get_frame_path(video_uuid, frame_number), as_gray=True)
            brightness_levels.append(np.mean(image_gray) * 255)
        except Exception:
            pass

        for j, rect in enumerate(rects):
            class_counts[labels[j]] += 1
            width, height = int(rect[2] - rect[0]), int(rect[3] - rect[1])

            if width > 0 and height > 0:
                aspect_ratios.append(width / height)
                video_info = video_info_cache[video_uuid]
                if video_info and video_info['width'] > 0 and video_info['height'] > 0:
                    center_x = (float(rect[0]) + float(rect[2])) / 2.0 / float(video_info['width'])
                    center_y = (float(rect[1]) + float(rect[3])) / 2.0 / float(video_info['height'])
                    center_points.append({'x': center_x, 'y': center_y})
                all_bboxes_for_outliers.append(
                    {'id': f'{video_uuid}_{frame_number}_{j}', 'image_index': i, 'area': width * height,
                     'aspect_ratio': width / height})

    annotator_stats = {}
    all_tasks = [task for vid_tasks in tasks_by_video.values() for task in vid_tasks]
    for task in all_tasks:
        user = task['assigned_to']
        if user not in annotator_stats:
            annotator_stats[user] = {'image_count': 0, 'class_counts': Counter()}

    user_frame_sets = {user: set() for user in annotator_stats.keys()}
    for frame in all_frames:
        for task in all_tasks:
            if task['video_uuid'] == frame['video_uuid'] and task['start_frame'] <= frame['frame_number'] <= task[
                'end_frame']:
                user = task['assigned_to']
                user_frame_sets[user].add(f"{frame['video_uuid']}_{frame['frame_number']}")
                annotator_stats[user]['class_counts'].update(extract_labels(frame['bboxes_text']))
    for user, frame_set in user_frame_sets.items():
        annotator_stats[user]['image_count'] = len(frame_set)

    gallery_images = [{
        'original_url': f"/media/frames/{f['video_uuid']}/frame_{f['frame_number']:05d}.jpg",
        'video': f['video_description'], 'frame': f['frame_number'], 'video_uuid': f['video_uuid'],
        'task_uuid': get_task_for_frame(f['video_uuid'], f['frame_number'])
    } for f in all_frames]

    # Generate Summary and Warnings
    warnings = []
    total_instances = sum(class_counts.values())
    if class_counts:
        avg_instances = total_instances / len(class_counts)
        for class_name, count in class_counts.items():
            if count < 10 or count < avg_instances * 0.1:
                warnings.append(
                    f"<b>Class Imbalance:</b> Class '{class_name}' has very few instances ({count}), which may lead to poor model performance.")

    small_object_threshold = 100
    small_object_count = sum(1 for bbox in all_bboxes_for_outliers if bbox['area'] < small_object_threshold)
    if small_object_count > 0:
        warnings.append(
            f"<b>Small Objects:</b> Found {small_object_count} objects with an area smaller than {small_object_threshold} pixels. Please verify they are not annotation errors.")

    if suspicious_pairs:
        warnings.append(
            f"<b>Potential Duplicates:</b> Found {len(suspicious_pairs)} pairs of bounding boxes with very high overlap (IoU > 0.95), suggesting possible duplicate annotations.")

    summary_text = f"This dataset contains <strong>{len(class_counts)}</strong> classes with a total of <strong>{total_instances}</strong> instances across <strong>{len(all_frames)}</strong> annotated images."

    return jsonify({
        'success': True,
        'summary_text': summary_text,
        'warnings': warnings,
        'class_counts': dict(class_counts),
        'aspect_ratios': aspect_ratios,
        'objects_per_image': objects_per_image,
        'center_points': center_points,
        'brightness_levels': brightness_levels,
        'annotator_stats': {u: {'image_count': d['image_count'], 'class_counts': dict(d['class_counts'])} for u, d in
                            annotator_stats.items()},
        'all_bboxes': all_bboxes_for_outliers,
        'suspicious_pairs': suspicious_pairs,
        'image_class_map': image_class_map,
        'gallery_images': gallery_images
    })


def startup_ai_models():
    if sam_tasks:
        logging.info("Starting initial AI model load based on settings...")
        sam_tasks.get_sam_model()


if __name__ == '__main__':
    from waitress import serve

    logging.info("Starting AI model initialization in a background thread...")
    threading.Thread(target=startup_ai_models, name="AI-Initializers").start()
    print("=" * 60)
    print("FTC-ML Server is running.")
    print("Open your web browser and go to http://127.0.0.1:5000")
    print("=" * 60)
    serve(app, host='0.0.0.0', port=5000)