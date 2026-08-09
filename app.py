import os
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
import base64
from collections import Counter, defaultdict
from skimage import io as skio
import itertools
import random
import torch
import torch.nn.functional as F
import atexit
import settings_manager
import config
import database
import file_storage
import background_tasks
import ai_models
from bbox_writer import validate_bboxes_text, convert_text_to_rects_and_labels, extract_labels
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Style, init
import webview
from threading import Thread
import multiprocessing


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'TIMESTAMP': Fore.WHITE + Style.DIM,
        'THREAD': Fore.CYAN,
        'LEVEL_DEFAULT': Fore.WHITE,
        'MESSAGE_DEFAULT': Fore.WHITE + Style.NORMAL,

        logging.DEBUG: {'level': Fore.MAGENTA, 'message': Fore.MAGENTA},
        logging.INFO: {'level': Fore.GREEN + Style.BRIGHT, 'message': Fore.WHITE},
        logging.WARNING: {'level': Fore.YELLOW + Style.BRIGHT, 'message': Fore.YELLOW},
        logging.ERROR: {'level': Fore.RED + Style.BRIGHT, 'message': Fore.RED},
        logging.CRITICAL: {'level': Fore.RED + Style.BRIGHT, 'message': Fore.RED + Style.BRIGHT},
    }

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        level_colors = self.COLORS.get(record.levelno, {
            'level': self.COLORS['LEVEL_DEFAULT'],
            'message': self.COLORS['MESSAGE_DEFAULT']
        })

        asctime = self.formatTime(record, self.datefmt)
        colored_asctime = f"{self.COLORS['TIMESTAMP']}{asctime}{Style.RESET_ALL}"

        colored_levelname = f"{level_colors['level']}{record.levelname:<8}{Style.RESET_ALL}"

        colored_threadname = f"{self.COLORS['THREAD']}[{record.threadName}]{Style.RESET_ALL}"

        message = record.getMessage()
        colored_message = f"{level_colors['message']}{message}{Style.RESET_ALL}"

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            colored_message += f"\n{level_colors['message']}{exc_text}{Style.RESET_ALL}"
        return f"{colored_asctime} - {colored_levelname} - {colored_threadname} - {colored_message}"


try:
    import yaml
except ImportError:
    logging.error("PyYAML is not installed! Dataset export will fail. Please run 'pip install pyyaml'.")
    yaml = None

# ----------------- 动态获取线程数 -----------------
initial_settings = settings_manager.load_settings()
max_workers_setting = initial_settings.get('max_workers', 8)

if str(max_workers_setting).lower() == 'auto':
    try:
        max_workers_setting = multiprocessing.cpu_count()
    except NotImplementedError:
        max_workers_setting = 8
else:
    max_workers_setting = int(max_workers_setting)

app = flask.Flask(__name__)
app.secret_key = os.urandom(24)
APP_BOOT_ID = uuid.uuid4().hex
# 注: 原来这里有一个 prototype_executor (ThreadPoolExecutor)，专给"重建类别原型"
# (MobileNet+KMeans，比较慢，需要后台跑) 用。SAM3 迁移后，"类别检索文本"只是一次
# DB 写入 (见 /api/setClassSam3Prompt)，同步返回即可，已确认全仓库没有其它地方在用
# 这个线程池，一并删除。
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

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
    hue = hash_val % 180
    color_hsv = np.uint8([[[hue, 200, 200]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, color_bgr))


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = float(boxAArea + boxBArea - interArea)
    if union <= 0:
        return 0.0

    iou = interArea / union
    return iou


def generate_mosaic_previews(sample_pool, selected_video_uuid, selected_frame_number):
    if len(sample_pool) < 4:
        sample_pool.extend(sample_pool * (4 - len(sample_pool)))

    all_labels = database.get_all_class_labels()
    class_map = {name: i for i, name in enumerate(all_labels)}

    image_infos = []
    for sample in sample_pool:
        video_info = database.get_video_entity(sample['video_uuid'])
        frame_info = database.get_frame_bboxes(sample['video_uuid'], sample['frame_number'])

        if video_info and frame_info and frame_info['bboxes_text']:
            image_infos.append({
                "video_uuid": sample['video_uuid'],
                "frame_number": sample['frame_number'],
                "bboxes_text": frame_info['bboxes_text'],
                "width": video_info['width'],
                "height": video_info['height']
            })

    if len(image_infos) < 4:
        return None, 'Not enough labeled images in the sample pool to generate a mosaic preview.'

    previews = []
    selected_image_info = next((info for info in image_infos if info['video_uuid'] == selected_video_uuid and info[
        'frame_number'] == selected_frame_number), None)

    for _ in range(6):
        other_images = [info for info in image_infos if info != selected_image_info]
        random.shuffle(other_images)

        mosaic_set = [selected_image_info] + other_images[:3] if selected_image_info else other_images[:4]
        random.shuffle(mosaic_set)

        mosaic_img, final_bboxes = file_storage.create_mosaic_image(mosaic_set, class_map)

        h, w, _ = mosaic_img.shape
        vis_image = mosaic_img.copy()
        for bbox_data in final_bboxes:
            class_index, x_center, y_center, width_norm, height_norm = bbox_data
            class_name = all_labels[class_index]
            color = string_to_color_bgr(class_name)

            x1 = int((x_center - width_norm / 2) * w)
            y1 = int((y_center - height_norm / 2) * h)
            x2 = int((x_center + width_norm / 2) * w)
            y2 = int((y_center + height_norm / 2) * h)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        _, buffer = cv2.imencode('.jpg', vis_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        previews.append(f"data:image/jpeg;base64,{img_base64}")

    return previews, None


@app.route('/')
def index():
    settings = settings_manager.load_settings()
    if not settings.get('initial_setup_done', False):
        return flask.redirect(flask.url_for('setup_wizard'))

    return render_template('root.html',
                           limit_data=config.get_limit_data_for_render_template(),
                           tracker_fns=config.TRACKER_FNS,
                           server_boot_id=APP_BOOT_ID)


@app.route('/setup')
def setup_wizard():
    return render_template('setup.html')


@app.route('/api/detect_hardware', methods=['GET'])
def detect_hardware():
    """检测本机硬件配置并返回"""
    cpu_cores = os.cpu_count() or 4

    # 检测内存 (如果没有 psutil 则默认返回 0)
    ram_gb = 0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    # 检测显卡与显存
    has_cuda = torch.cuda.is_available()
    gpu_name = "None"
    vram_gb = 0

    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)

    # 简单的分级策略 (low, mid, high)
    tier = "low"
    if has_cuda:
        if vram_gb >= 8:
            tier = "high"
        elif vram_gb >= 4:
            tier = "mid"

    return jsonify({
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "has_cuda": has_cuda,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "tier": tier
    })


@app.route('/api/complete_setup', methods=['POST'])
def complete_setup():
    """保存向导配置并唤醒 AI 模型加载"""
    data = request.json
    settings = settings_manager.load_settings()

    # 覆盖所有传过来的设置
    for key, value in data.items():
        settings[key] = value

    settings['initial_setup_done'] = True

    if settings_manager.save_settings(settings):
        # 更新设备状态
        settings_manager.update_device()

        # 核心：使用独立线程启动 AI 模型加载，防止前端请求超时
        logging.info("配置完成，正在后台唤醒 AI 模型引擎...")
        threading.Thread(target=ai_models.startup_ai_models, name="Delayed-AI-Startup").start()

        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Failed to save settings."}), 500

@app.route('/labelVideo')
def label_video():
    task_uuid = request.args.get('task_uuid')
    video_uuid = request.args.get('video_uuid')

    if not task_uuid and video_uuid:
        tasks = database.get_tasks_for_video(video_uuid)
        if tasks:
            task_uuid = tasks[0]['task_uuid']
        else:
            video = database.get_video_entity(video_uuid)
            if video:
                task_uuid = database.create_annotation_task(
                    video_uuid, "admin", video['description'], 0, max(0, (video.get('frame_count') or 1) - 1)
                )

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
    settings = settings_manager.load_settings()

    return render_template('labelVideo.html',
                           task_entity=sanitize_dict(task_entity),
                           video_entity=sanitize_dict(video_entity),
                           first_frame_url=first_frame_url,
                           settings=settings,
                           limit_data=config.get_limit_data_for_render_template(),
                           is_sam_enabled=settings.get('enable_sam_model', True),
                           is_feature_extractor_enabled=settings.get('enable_feature_extractor', True))


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

        # 获取指定帧的数据库记录
        frames = database.get_video_frames(video_uuid)
        frame_item = next((f for f in frames if f['frame_number'] == frame_number), None)

        if frame_item:
            # 1. 优先绘制高级 JSON 标注 (适用于分割多边形、姿态、高级框)
            if frame_item.get('annotations_json') and frame_item['annotations_json'].strip():
                from annotation_model import AnnotationData
                ann_data = AnnotationData.from_json(frame_item['annotations_json'])
                for obj in ann_data.objects:
                    color = string_to_color_bgr(obj.label)

                    # 绘制分割多边形与半透明色彩掩码
                    if obj.type == 'polygon' and obj.polygon:
                        pts = np.array(obj.polygon, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

                        overlay = image.copy()
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

                        # 绘制类别标签
                        x_min = int(np.min(pts[:, 0, 0]))
                        y_min = int(np.min(pts[:, 0, 1]))
                        (text_w, text_h), _ = cv2.getTextSize(obj.label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(image, (x_min, max(0, y_min - text_h - 5)), (x_min + text_w, y_min), color, -1)
                        cv2.putText(image, obj.label, (x_min, max(text_h, y_min - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)

                    # 绘制矩形框
                    elif obj.type == 'bbox' and obj.bbox:
                        x1, y1, x2, y2 = map(int, obj.bbox)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        (text_w, text_h), _ = cv2.getTextSize(obj.label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(image, (x1, max(0, y1 - text_h - 5)), (x1 + text_w, y1), color, -1)
                        cv2.putText(image, obj.label, (x1, max(text_h, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)

            # 2. 兼容传统文本检测框 (bboxes_text)
            elif frame_item.get('bboxes_text') and frame_item['bboxes_text'].strip():
                rects, labels, _ = convert_text_to_rects_and_labels(frame_item['bboxes_text'])
                for i, rect in enumerate(rects):
                    label = labels[i]
                    color = string_to_color_bgr(label)
                    x1, y1, x2, y2 = rect
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(image, (x1, max(0, y1 - text_h - 5)), (x1 + text_w, y1), color, -1)
                    cv2.putText(image, label, (x1, max(text_h, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)

        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return "Failed to encode image", 500

        return Response(buffer.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        logging.error(f"Error generating annotated frame for {video_uuid}/{frame_number}: {e}", exc_info=True)
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
    try:
        raw_val = request.form.get('frame_interval') or request.form.get('target_fps') or '5'
        frame_interval = int(float(raw_val))
        if frame_interval < 1:
            frame_interval = 1
    except ValueError:
        frame_interval = 5

    is_valid, message = validate_description(desc, [v['description'] for v in database.get_all_video_list()])
    if not is_valid:
        return jsonify({'success': False, 'message': message}), 400
    if not video_file:
        return jsonify({'success': False, 'message': 'No video file provided.'}), 400

    create_time_ms = int(time.time() * 1000)
    annotation_type = request.form.get('annotation_type', 'detection')
    video_uuid = database.create_video_entry(desc, video_file.filename, 0, create_time_ms, annotation_type=annotation_type)
    file_storage.save_uploaded_video(video_file, video_uuid)

    threading.Thread(target=background_tasks.extract_frames_task, args=(video_uuid, frame_interval),
                     name=f"Extractor-{video_uuid[:6]}").start()

    return jsonify({'success': True, 'video_uuid': video_uuid})


@app.route('/importFrames', methods=['POST'])
def import_frames():
    video_uuid = request.form.get('video_uuid')
    uploaded_files = request.files.getlist('frame_files')

    if not video_uuid or not uploaded_files:
        return jsonify({'success': False, 'message': 'Missing video UUID or files.'}), 400

    video = database.get_video_entity(video_uuid)
    if not video:
        return jsonify({'success': False, 'message': 'Video not found.'}), 404

    total_imported = 0

    try:
        # 分离图片和视频处理
        image_bytes_list = []

        for file in uploaded_files:
            filename = file.filename.lower()

            # 1. 如果是视频文件 -> 抽帧
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                # 保存临时文件以便 OpenCV 读取
                temp_video_path = os.path.join(config.STORAGE_DIR, f"temp_import_{uuid.uuid4().hex}.mp4")
                file.save(temp_video_path)

                try:
                    cap = cv2.VideoCapture(temp_video_path)
                    while True:
                        ret, frame = cap.read()
                        if not ret: break

                        # 编码为 JPG binary
                        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if success:
                            image_bytes_list.append(buffer.tobytes())
                    cap.release()
                finally:
                    # 清理临时视频
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)

            # 2. 如果是图片文件 -> 直接读取
            elif filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                image_bytes_list.append(file.read())

        # 3. 统一写入数据库 (使用 database.py 中新写的安全函数)
        if image_bytes_list:
            total_imported = database.add_frames_to_video(video_uuid, image_bytes_list)

        return jsonify({'success': True, 'imported_count': total_imported})

    except Exception as e:
        logging.error(f"Failed to import frames for {video_uuid}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f"Import failed: {str(e)}"}), 500


@app.route('/retrieveVideoEntity', methods=['POST'])
def retrieve_video_entity():
    video_uuid = (request.json or {}).get('video_uuid')
    entity = database.get_video_entity(video_uuid)
    if entity:
        return jsonify({'success': True, 'video_entity': sanitize_dict(entity)})
    return jsonify({'success': False, 'message': 'Video not found.'})


@app.route('/deleteVideo', methods=['POST'])
def delete_video():
    video_uuid = (request.json or {}).get('video_uuid')
    database.delete_video(video_uuid)
    file_storage.delete_video_file(video_uuid)
    file_storage.delete_frames_for_video(video_uuid)
    return jsonify({'success': True})


@app.route('/retrieveVideoFrames', methods=['POST'])
def retrieve_video_frames():
    video_uuid = (request.json or {}).get('video_uuid')
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


@app.route('/saveFrameAnnotations', methods=['POST'])
def save_frame_annotations():
    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = int(data.get('frame_number'))
    annotations_json = data.get('annotations_json')

    if not video_uuid or frame_number is None:
        return jsonify({'success': False, 'message': 'Missing video_uuid or frame_number'}), 400

    database.save_frame_annotations(video_uuid, frame_number, annotations_json)
    return jsonify({'success': True})


@app.route('/getFrameAnnotations', methods=['GET'])
def get_frame_annotations():
    video_uuid = request.args.get('video_uuid')
    frame_number = request.args.get('frame_number', type=int)

    if not video_uuid or frame_number is None:
        return jsonify({'success': False, 'message': 'Missing video_uuid or frame_number'}), 400

    frame_data = database.get_frame_annotations(video_uuid, frame_number)
    return jsonify({'success': True, 'annotations': frame_data})


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
    task_uuid = (request.json or {}).get('task_uuid')
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


@app.route('/api/setClassSam3Prompt', methods=['POST'])
def set_class_sam3_prompt_route():
    """
    设置/查看某个类别用于 SAM3 检索的描述文本。取代旧的 /api/rebuild_prototypes。
    旧接口是"用已标注样本训练/重建一个 embedding 原型"（需要后台异步任务），
    新架构下这只是一次 DB 文本写入，同步返回即可，不再需要 prototype_executor。
    """
    data = request.json
    class_name = data.get('class_name')
    if not class_name:
        return jsonify({'success': False, 'message': 'class_name is required.'}), 400

    if 'sam3_prompt' in data:
        database.set_class_sam3_prompt(class_name, data.get('sam3_prompt'))

    return jsonify({
        'success': True,
        'class_name': class_name,
        'sam3_prompt': database.get_class_sam3_prompt(class_name),
        'has_labeled_examples': ai_models.class_has_labeled_examples(class_name),
    })


@app.route('/api/listClassSam3Prompts', methods=['GET'])
def list_class_sam3_prompts_route():
    return jsonify({'success': True, 'classes': database.get_all_class_labels_with_prompts()})


@app.route('/api/interpolateBboxes', methods=['POST'])
def interpolate_bboxes():
    data = request.json
    video_uuid = data.get('video_uuid')
    object_id = data.get('object_id')
    start_frame_data = data.get('start_frame')
    end_frame_data = data.get('end_frame')

    if not all([video_uuid, object_id, start_frame_data, end_frame_data]):
        return jsonify({'success': False, 'message': 'Missing required data.'}), 400

    try:
        start_frame_num = int(start_frame_data['frame_number'])
        end_frame_num = int(end_frame_data['frame_number'])
        start_bbox = start_frame_data['bbox']
        end_bbox = end_frame_data['bbox']
        label = start_bbox['label']

        if start_frame_num >= end_frame_num:
            start_frame_num, end_frame_num = end_frame_num, start_frame_num
            start_bbox, end_bbox = end_bbox, start_bbox

        total_steps = end_frame_num - start_frame_num
        if total_steps <= 1:
            return jsonify({'success': True, 'message': 'No frames to interpolate.'})
        for i in range(1, total_steps):
            current_frame_num = start_frame_num + i
            t = i / float(total_steps)
            interp_x1 = int(start_bbox['x1'] + (end_bbox['x1'] - start_bbox['x1']) * t)
            interp_y1 = int(start_bbox['y1'] + (end_bbox['y1'] - start_bbox['y1']) * t)
            interp_x2 = int(start_bbox['x2'] + (end_bbox['x2'] - start_bbox['x2']) * t)
            interp_y2 = int(start_bbox['y2'] + (end_bbox['y2'] - start_bbox['y2']) * t)

            new_bbox_line = f"{interp_x1},{interp_y1},{interp_x2},{interp_y2},{label},{object_id}"

            # 替换为使用新的通用查询方法
            frame_db = database.get_frame_bboxes(video_uuid, current_frame_num)

            existing_bboxes = frame_db['bboxes_text'] if frame_db else ''
            lines = existing_bboxes.split('\n') if existing_bboxes else []
            updated_lines = []
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 6 and parts[5] == object_id:
                    continue
                updated_lines.append(line)

            updated_lines.append(new_bbox_line)
            final_bboxes_text = '\n'.join(filter(None, updated_lines))

            database.save_frame_bboxes(video_uuid, current_frame_num, final_bboxes_text)

        return jsonify({'success': True, 'message': f'Interpolated {total_steps - 1} frames successfully.'})

    except Exception as e:
        logging.error(f"Interpolation failed: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500


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


@app.route('/api/checkClassUsage', methods=['POST'])


def check_class_usage():
    label_name = (request.json or {}).get('label_name')
    if not label_name:
        return jsonify({'success': False, 'usage_count': 0})

    try:
        with database.engine.connect() as conn:
            count = conn.execute(
                database.text('SELECT COUNT(DISTINCT frame_id) FROM frame_labels WHERE label_name = :ln'),
                {"ln": label_name}
            ).scalar()
        return jsonify({'success': True, 'usage_count': count or 0})
    except Exception as e:
        logging.error(f"Failed to check class usage: {e}")
        return jsonify({'success': False, 'usage_count': 0})

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
    device_changed = current_settings.get('gpu_device') != new_settings.get('gpu_device')
    max_workers_changed = str(current_settings.get('max_workers')) != str(new_settings.get('max_workers'))

    restart_required = sam_model_changed or device_changed or max_workers_changed

    # --- THE FIX: Merge the new settings into the current ones ---
    current_settings.update(new_settings)

    # --- Pass the merged 'current_settings' instead of 'new_settings' ---
    if settings_manager.save_settings(current_settings):
        if sam_model_changed or device_changed:
            logging.info("SAM model or device setting changed. Clearing SAM2/SAM3 cache.")
            try:
                # 修复: 原代码这里 `from ultralytics_sam_tasks import _sam_model_cache`
                # 引用的是一个不存在的变量名（实际缓存字典叫 _sam_cache），一直被
                # except (ImportError, AttributeError) 静默吞掉，"切换模型后清缓存"
                # 这个动作过去其实从没真正生效过。这里改成正确清空 SAM2/SAM3 各自的
                # 模型缓存，以及 SAM3 迁移新增的帧级 backbone 缓存。
                import ultralytics_sam_tasks as sam_tasks_module
                sam_tasks_module._sam_cache["sam2_image_predictor"] = None
                sam_tasks_module._sam_cache["sam2_video_predictor"] = None
                sam_tasks_module._sam_cache["image_model"] = None
                sam_tasks_module._sam_cache["image_processor"] = None
                sam_tasks_module._sam_cache["multiplex_predictor"] = None
                sam_tasks_module.clear_sam3_frame_state_cache()
            except Exception as e:
                logging.warning(f"Could not clear SAM model cache: {e}")

        if device_changed:
            settings_manager.update_device()
            ai_models.clear_retrieval_engine_cache()

        return jsonify({
            'success': True,
            'message': 'Settings saved successfully!',
            'restart_required': restart_required
        })
    else:
        return jsonify({'success': False, 'message': 'Failed to save settings to file.'}), 500


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    try:
        count = ai_models.sam_tasks.frame_cache_size() if ai_models.sam_tasks else 0
        ai_models.clear_retrieval_engine_cache()
        logging.info(f"Cleared {count} items from SAM3 frame-state cache.")
        return jsonify({'success': True, 'message': f'Successfully cleared {count} cached items.'})
    except Exception as e:
        logging.error(f"Failed to clear cache: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while clearing the cache.'}), 500


@app.route('/samPredict', methods=['POST'])
def sam_predict():
    try:
        from ultralytics_sam_tasks import predict_box_from_point_ultralytics, get_sam_model
        if not get_sam_model():
            return jsonify(
                {'success': False, 'message': 'SAM model is disabled in system settings to save resources.'}), 501
    except ImportError:
        return jsonify({'success': False, 'message': 'SAM features are not installed on server.'}), 501

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

        bbox = predict_box_from_point_ultralytics(image_path, coords_tuple)

        if bbox:
            return jsonify({'success': True, 'bbox': bbox, 'polygon': bbox.get('polygon', [])})
        else:
            return jsonify({'success': False, 'message': 'No object found at the specified point.'})

    except Exception as e:
        logging.error(f"SAM prediction failed: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/sam3_text_predict', methods=['POST'])
def sam3_text_predict():
    try:
        from ultralytics_sam_tasks import get_sam_model, sam3_query_frame
        if not get_sam_model():
            return jsonify(
                {'success': False, 'message': 'SAM model is disabled in system settings to save resources.'}), 501
    except ImportError:
        return jsonify({'success': False, 'message': 'SAM features are not installed on server.'}), 501

    data = request.json
    if not data:
        return jsonify({'success': False, 'message': 'No request payload provided.'}), 400

    video_uuid = data.get('video_uuid')
    frame_number = data.get('frame_number')
    text_prompt = data.get('text_prompt')

    confidence = float(data.get('confidence', 0.25))

    if not all([video_uuid, frame_number is not None, text_prompt]):
        return jsonify(
            {'success': False, 'message': 'Missing required data (video_uuid, frame_number, text_prompt).'}), 400

    try:
        # 改为直接调用统一原语 sam3_query_frame（同一帧多次查询会复用 backbone 缓存），
        # 取代原来独立实现、逻辑重复的 predict_boxes_from_text_sam3。
        results = sam3_query_frame(video_uuid, int(frame_number), text_prompt=text_prompt, confidence=confidence)

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        logging.error(f"SAM 3 Text prediction failed: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/interactive_segment/preprocess', methods=['POST'])
def interactive_segment_preprocess_route():
    settings = settings_manager.load_settings()
    if not settings.get('enable_feature_extractor', True):
        return jsonify({
            'success': False,
            'message': 'SAM3 retrieval features are disabled in system settings to save resources.'
        }), 501

    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = data.get('frame_number')

    if video_uuid is None or frame_number is None:
        return jsonify({'success': False, 'message': 'Missing video_uuid or frame_number.'}), 400

    try:
        # 预热该帧的 SAM3 backbone 缓存（跑一次最贵的图像前向），后续同一帧的文本/
        # 框样例查询都会直接复用，不再需要旧版"生成全部候选框+提特征"的预处理。
        ai_models.warm_frame_cache(video_uuid, int(frame_number))
        cache_key = f"{video_uuid}_{frame_number}"
        return jsonify({'success': True, 'message': 'Preprocessing successful', 'cache_key': cache_key})
    except Exception as e:
        logging.error(f"智能选择预处理失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Internal Server Error: {str(e)}'}), 500


@app.route('/interactive_segment/predict', methods=['POST'])
def interactive_segment_predict_route():
    settings = settings_manager.load_settings()
    if not settings.get('enable_feature_extractor', True):
        return jsonify({
            'success': False,
            'message': 'SAM3 retrieval features are disabled in system settings to save resources.'
        }), 501

    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = int(data.get('frame_number'))
    prompt_boxes = data.get('prompt_boxes', [])
    # 新增: 负例框 (框样例检索的 label=False)，同一帧内"这不是我想要的"的反例，
    # 前端可以让用户额外画几个负例框来提升精度。
    negative_prompt_boxes = data.get('negative_prompt_boxes', [])
    use_color = data.get('use_color', False)

    if not all([video_uuid, frame_number is not None, prompt_boxes]):
        return jsonify({'success': False, 'message': 'Missing required data.'}), 400
    if not prompt_boxes:
        return jsonify({'success': False, 'message': 'Positive prompt boxes are required.'}), 400

    try:
        positive_prompt_box = prompt_boxes[0]
        results = ai_models.predict_from_one_shot(
            video_uuid, frame_number, positive_prompt_box,
            negative_prompt_boxes=negative_prompt_boxes, use_color=use_color
        )
        return jsonify({'success': True, 'results': results})

    except Exception as e:
        logging.error(f"智能选择预测失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Internal Server Error: {str(e)}'}), 500


@app.route('/interactive_segment/predict_from_dataset', methods=['POST'])
def predict_from_dataset_route():
    settings = settings_manager.load_settings()
    if not settings.get('enable_feature_extractor', True):
        return jsonify({
            'success': False,
            'message': 'SAM3 retrieval features are disabled in system settings to save resources.'
        }), 501

    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = int(data.get('frame_number'))
    class_name = data.get('class_name')
    confidence_threshold = float(data.get('confidence', settings.get('default_preannotation_conf', 0.5)))

    if not all([video_uuid, frame_number is not None, class_name]):
        return jsonify({'success': False, 'message': 'Missing required data.'}), 400

    try:
        # 不再需要"先构建原型再预测"两步：直接用该类别的 SAM3 检索文本
        # (database.class_labels.sam3_prompt，没配置则回退用类别名) 查询这一帧。
        results = ai_models.predict_by_class_text(video_uuid, frame_number, class_name, confidence_threshold)
        response = {'success': True, 'results': results}
        if not ai_models.class_has_labeled_examples(class_name):
            response['warning'] = (
                f"类别 '{class_name}' 目前还没有任何标注样本，检索完全依赖 SAM3 描述文本，"
                f"建议先手动标注几个样本或在类别设置里完善检索描述以提升准确率。"
            )
        return jsonify(response)

    except Exception as e:
        logging.error(f"Dataset-driven prediction failed: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Internal Server Error: {str(e)}'}), 500


@app.route('/api/background_preprocess_frame', methods=['POST'])
def background_preprocess_frame():
    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = data.get('frame_number')

    if not video_uuid or frame_number is None:
        return jsonify({'success': False, 'message': 'Missing data.'}), 400

    settings = settings_manager.load_settings()
    if not settings.get('auto_preprocess', True):
        return jsonify({'success': True, 'message': 'Auto preprocess disabled in settings.'})

    if background_tasks.active_tasks.get(video_uuid):
        return jsonify({'success': False, 'message': 'Another task is active.'})
    if ai_models.is_frame_cached(video_uuid, int(frame_number)):
        return jsonify({'success': True, 'message': 'Already cached.'})

    threading.Thread(
        target=lambda: ai_models.warm_frame_cache(video_uuid, int(frame_number)),
        name=f"Preprocess-{video_uuid[:6]}-{frame_number}"
    ).start()

    return jsonify({'success': True, 'message': 'Preprocessing started in background.'})


# 注: 这里原来有一个 /api/get_random_frames_for_neg_sampling 路由，专给旧版
# negative-sampling-modal（随机抽帧、手绘负例框、给 MobileNet 建负原型）用。SAM3
# 迁移后批量应用改成纯文本检索驱动，前端已经不再调用这个接口，一并删除。


@app.route('/api/apply_class_to_videos', methods=['POST'])
def apply_class_to_videos_route():
    """
    把某个类别的 SAM3 检索文本批量应用到一批视频的所有未标注帧上。取代旧的
    /apply_prototypes_to_video（原来只能作用于单个视频、依赖 MobileNet 原型）。

    这是"SAM3 标注完一帧后应用到整个数据集"和"智能选择结果应用到整个数据集"两个
    入口共用的同一个后端接口——两者最终都归结为"给定一个类别名 + 一批视频，用这个
    类别的 SAM3 检索文本逐帧检测"，区别只在前端用什么方式收集到这个 class_name。

    请求体:
      video_uuids: [str, ...]  必填，要处理的视频列表（可以只有当前这一个视频，
                                也可以是某个数据集包含的全部视频）
      class_name: str          必填
      confidence_threshold: float  可选，默认读取全局 default_preannotation_conf
    """
    settings = settings_manager.load_settings()
    if not settings.get('enable_feature_extractor', True):
        return jsonify({
            'success': False,
            'message': 'SAM3 retrieval features are disabled in system settings to save resources.'
        }), 501

    data = request.json or {}
    video_uuids = data.get('video_uuids')
    class_name = data.get('class_name')
    confidence_threshold = float(data.get('confidence_threshold', settings.get('default_preannotation_conf', 0.5)))
    process_all_frames = bool(data.get('process_all_frames', True))

    if not video_uuids or not isinstance(video_uuids, list):
        return jsonify({'success': False, 'message': 'video_uuids (non-empty list) is required.'}), 400
    if not class_name:
        return jsonify({'success': False, 'message': 'class_name is required.'}), 400

    busy = [vu for vu in video_uuids if background_tasks.active_tasks.get(vu)]
    if busy:
        return jsonify({
            'success': False,
            'message': f"以下视频当前有其它任务在运行: {', '.join(v[:8] for v in busy)}"
        }), 409

    task_uuid = str(uuid.uuid4())
    threading.Thread(
        target=background_tasks.apply_class_to_videos_task,
        args=(task_uuid, video_uuids, class_name, confidence_threshold, app.app_context(), process_all_frames),
        name=f"ApplyClass-{task_uuid[:8]}"
    ).start()

    return jsonify({'success': True, 'task_uuid': task_uuid, 'message': 'Task to apply suggestions has started.'})


@app.route('/api/batchApplyStatus/<task_uuid>', methods=['GET'])
def batch_apply_status_route(task_uuid):
    session = background_tasks.batch_apply_sessions.get(task_uuid)
    if not session:
        return jsonify({'success': False, 'message': 'Task not found.'}), 404
    return jsonify({'success': True, **session})


@app.route('/api/datasetsContainingVideo/<video_uuid>', methods=['GET'])
def datasets_containing_video_route(video_uuid):
    """
    给"应用到数据集"的前端选择器用: 找出哪些已创建的数据集包含这个视频。一个视频可能
    属于 0 个、1 个或多个数据集（数据集是"选一批已标注视频打包导出"的快照，和视频不是
    一对一关系）。前端可以用这个列表给用户一个选择器；如果返回空列表，前端应该退化成
    "只应用到当前视频"或者提示用户先创建/选择数据集。
    """
    matching = []
    for ds in database.get_dataset_list():
        try:
            ds_video_uuids = json.loads(ds.get('video_uuids') or '[]')
        except (TypeError, ValueError):
            ds_video_uuids = []
        if video_uuid in ds_video_uuids:
            matching.append({
                'dataset_uuid': ds['dataset_uuid'],
                'description': ds.get('description'),
                'video_uuids': ds_video_uuids,
            })
    return jsonify({'success': True, 'datasets': matching})


@app.route('/startSam2Tracking', methods=['POST'])
def start_sam2_tracking():
    try:
        from ultralytics_sam_tasks import get_sam_model
        if not get_sam_model():
            return jsonify(
                {'success': False, 'message': 'SAM model is disabled in system settings to save resources.'}), 501
    except ImportError:
        return jsonify({'success': False, 'message': 'SAM features are not installed on server.'}), 501

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


@app.route('/startSam2BatchTracking', methods=['POST'])
def start_sam2_batch_tracking():
    try:
        from ultralytics_sam_tasks import get_sam_model
        if not get_sam_model():
            return jsonify(
                {'success': False, 'message': 'SAM model is disabled in system settings to save resources.'}), 501
    except ImportError:
        return jsonify({'success': False, 'message': 'SAM features are not installed on server.'}), 501

    data = request.json
    video_uuid = data.get('video_uuid')
    start_frame = int(data.get('start_frame'))
    end_frame = int(data.get('end_frame'))
    init_bboxes_text = data.get('init_bboxes_text')

    if not all([video_uuid, start_frame is not None, end_frame is not None, init_bboxes_text]):
        return jsonify({'success': False, 'message': 'Missing required data for batch tracking.'}), 400

    if background_tasks.active_tasks.get(video_uuid):
        return jsonify(
            {'success': False, 'message': 'Another task (extraction or tracking) is already running for this video.'})

    tracker_uuid = str(uuid.uuid4().hex)

    threading.Thread(target=background_tasks.start_sam2_batch_tracking_task, args=(
        video_uuid, tracker_uuid, start_frame, end_frame, init_bboxes_text
    ), name=f"SAM-Batch-Tracker-{video_uuid[:6]}").start()

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
    tracker_uuid = (request.json or {}).get('tracker_uuid')
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
    tracker_uuid = (request.json or {}).get('tracker_uuid')
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
    tracker_uuid = (request.json or {}).get('tracker_uuid')
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
    export_format = data.get('export_format', 'yolo_v8_detect')
    augmentation_options = data.get('augmentation_options', {})

    is_valid, message = validate_description(desc, [d['description'] for d in database.get_dataset_list()])
    if not is_valid:
        return jsonify({'success': False, 'message': message}), 400
    if not video_uuids:
        return jsonify({'success': False, 'message': 'Please select at least one video.'}), 400

    create_time = int(time.time() * 1000)
    dataset_uuid = database.create_dataset_entry(desc, video_uuids, create_time, eval_percent, test_percent, export_format)

    threading.Thread(target=background_tasks.create_dataset_task, args=(
        dataset_uuid, video_uuids, eval_percent, test_percent, export_format, augmentation_options
    ), name=f"Dataset-{dataset_uuid[:6]}").start()

    return jsonify({'success': True, 'dataset_uuid': dataset_uuid})


@app.route('/regenerateDataset', methods=['POST'])
def regenerate_dataset():
    dataset_uuid = (request.json or {}).get('dataset_uuid')
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
    export_format = dataset.get('export_format', 'yolo_v8_detect')
    augmentation_options = {'enabled': False}

    threading.Thread(target=background_tasks.create_dataset_task, args=(
        dataset_uuid, video_uuids, eval_percent, test_percent, export_format, augmentation_options
    ), name=f"Dataset-Regen-{dataset_uuid[:6]}").start()

    return jsonify({'success': True, 'message': 'Dataset regeneration started.'})


@app.route('/api/export_formats', methods=['GET'])
def get_export_formats():
    from exporters import ExporterRegistry
    return jsonify({'success': True, 'formats': ExporterRegistry.list_all()})


@app.route('/downloadDataset/<dataset_uuid>')
def download_dataset(dataset_uuid):
    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset or dataset['status'] != 'READY' or not dataset['zip_path'] or not os.path.exists(dataset['zip_path']):
        return "Dataset not found or not ready.", 404
    try:
        filename = f"{dataset['description']}.zip" if dataset.get('description') else "dataset.zip"
        return send_file(dataset['zip_path'], as_attachment=True, download_name=filename)
    except Exception as e:
        logging.error(f"Could not send file: {e}")
        return "Error downloading file.", 500


@app.route('/deleteDataset', methods=['POST'])
def delete_dataset():
    dataset_uuid = (request.json or {}).get('dataset_uuid')
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
    label_file = request.files.get('label_file')
    model_type = request.form.get('model_type')

    is_valid, message = validate_description(desc, [m['description'] for m in database.get_model_list()])
    if not is_valid:
        return jsonify({'success': False, 'message': message}), 400

    if not model_file:
        return jsonify({'success': False, 'message': 'No model file provided.'}), 400

    is_tflite = model_file.filename.endswith('.tflite')
    is_pt = model_file.filename.endswith('.pt')

    if not is_tflite and not is_pt:
        return jsonify({'success': False, 'message': 'Please provide a .tflite or .pt model file.'}), 400

    if is_tflite and (not label_file or not (label_file.filename.endswith('.txt') or label_file.filename.endswith('.labels'))):
        return jsonify({'success': False, 'message': 'TFLite models require a .txt or .labels file.'}), 400

    if model_type not in ['float32', 'float16', 'uint8', 'int8']:
        return jsonify({'success': False, 'message': 'Invalid model type selected.'}), 400

    create_time = int(time.time() * 1000)
    label_filename = label_file.filename if label_file else 'embedded_yolo_classes.txt'
    model_uuid = database.import_model_metadata(desc, label_filename, model_type, create_time)

    file_storage.save_imported_model(model_file, model_uuid)
    if label_file:
        file_storage.save_imported_label_file(label_file, model_uuid)
    else:
        label_path = file_storage.get_label_file_path(model_uuid)
        with open(label_path, 'w') as f:
            f.write("")

    return jsonify({'success': True, 'model_uuid': model_uuid})


@app.route('/deleteModel', methods=['POST'])
def delete_model():
    model_uuid = (request.json or {}).get('model_uuid')
    database.delete_model(model_uuid)
    file_storage.delete_model_file(model_uuid)
    file_storage.delete_label_file(model_uuid)
    return jsonify({'success': True})


@app.route('/startPreAnnotation', methods=['POST'])
def start_pre_annotation():
    data = request.json
    video_uuid = data.get('video_uuid')
    model_uuid = data.get('model_uuid')
    options = data.get('options', {})

    if not video_uuid or not model_uuid:
        return jsonify({'success': False, 'message': 'Video UUID and Model UUID are required.'}), 400

    video = database.get_video_entity(video_uuid)
    if not video:
        return jsonify({'success': False, 'message': 'Video not found.'}), 404
    if video['status'] != 'READY':
        return jsonify({'success': False, 'message': f"Video must be in READY state, but is {video['status']}."}), 400

    if background_tasks.active_tasks.get(video_uuid):
        return jsonify({'success': False, 'message': 'Another task is already running for this video.'}), 409

    try:
        options['start_frame'] = int(options.get('start_frame', 0))
        options['end_frame'] = int(options.get('end_frame', video['frame_count'] - 1))
        options['confidence'] = float(options.get('confidence', 0.5))
        options['merge_strategy'] = options.get('merge_strategy', 'overwrite')
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'message': f'Invalid options provided: {e}'}), 400

    threading.Thread(
        target=background_tasks.pre_annotate_video_task,
        args=(video_uuid, model_uuid, options),
        name=f"PreAnnotator-{video_uuid[:6]}"
    ).start()

    return jsonify({'success': True, 'message': 'Pre-annotation task started.'})


@app.route('/cancelTask', methods=['POST'])
def cancel_task():
    video_uuid = (request.json or {}).get('video_uuid')
    if not video_uuid:
        return jsonify({'success': False, 'message': 'Video UUID is required.'}), 400

    video = database.get_video_entity(video_uuid)
    if not video:
        return jsonify({'success': False, 'message': 'Video not found.'}), 404

    if video['status'] in ['PRE_ANNOTATING', 'APPLYING_PROTOTYPES', 'APPLYING_CLASS']:
        database.update_video_status(video_uuid, 'CANCELLING', 'Cancellation requested by user.')
        return jsonify({'success': True, 'message': 'Cancellation request sent.'})
    else:
        return jsonify({'success': False, 'message': f'Cannot cancel task, video status is {video["status"]}.'}), 400


@app.route('/datasetAnalysis/<dataset_uuid>')
def dataset_analysis(dataset_uuid):
    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset:
        return "Dataset not found", 404
    settings = settings_manager.load_settings()
    return render_template('dataset_analysis.html',
                           dataset=sanitize_dict(dataset),
                           limit_data=config.get_limit_data_for_render_template(),
                           is_feature_extractor_enabled=settings.get('enable_feature_extractor', True))


@app.route('/api/datasetAnalysis/<dataset_uuid>', methods=['GET'])
def get_dataset_analysis_data(dataset_uuid):
    dataset = database.get_dataset_entity(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': 'Dataset not found.'}), 404

    video_uuids = json.loads(dataset.get('video_uuids', '[]'))
    tasks_by_video = {vu: database.get_tasks_for_video(vu) for vu in video_uuids}
    video_info_cache = {vu: database.get_video_entity(vu) for vu in video_uuids}

    export_format = dataset.get('export_format', '')
    is_classification = (export_format in ['folder_classification', 'yolo_cls']) or any(
        video_info_cache.get(vu, {}).get('annotation_type') == 'classification' for vu in video_uuids
    )

    def get_task_for_frame(video_uuid, frame_number):
        for task in tasks_by_video.get(video_uuid, []):
            if task['start_frame'] <= frame_number <= task['end_frame']:
                return task['task_uuid']
        return None

    all_frames = []
    for vu in video_uuids:
        v_info = video_info_cache.get(vu)
        if not v_info:
            continue
        v_desc = v_info.get('description', 'Unknown')
        for frame in database.get_annotated_video_frames(vu):
            all_frames.append({
                **dict(frame),
                'video_uuid': vu,
                'video_description': v_desc
            })

    class_counts = Counter()
    labels_per_image_counts = Counter()
    co_occurrence_counts = Counter()
    aspect_ratios, objects_per_image, center_points, brightness_levels = [], [], [], []
    all_bboxes_for_outliers = []
    suspicious_pairs = []
    image_class_map = {}

    from annotation_model import AnnotationData

    for i, frame in enumerate(all_frames):
        video_uuid, frame_number = frame['video_uuid'], frame['frame_number']
        rects, labels = [], []

        # 1. 优先解析 annotations_json (适用于分割 Polygon、姿态 Keypoint、矢量 Bbox 和 图像分类 Tag)
        if frame.get('annotations_json') and frame['annotations_json'].strip():
            try:
                ann_data = AnnotationData.from_json(frame['annotations_json'])
                if ann_data.classifications:
                    for cls_name in ann_data.classifications:
                        labels.append(cls_name)

                for obj in ann_data.objects:
                    label = obj.label
                    rect = None
                    if obj.type == 'polygon' and obj.polygon:
                        xs = [pt[0] for pt in obj.polygon]
                        ys = [pt[1] for pt in obj.polygon]
                        if xs and ys:
                            rect = [min(xs), min(ys), max(xs), max(ys)]
                    elif obj.type == 'bbox' and obj.bbox:
                        rect = obj.bbox
                    if rect and label:
                        rects.append(rect)
                        labels.append(label)
            except Exception as e:
                logging.error(f"Error parsing annotations_json for frame {video_uuid}/{frame_number}: {e}")

        # 2. 备用解析传统 bboxes_text (适用于早期检测模式)
        if not rects and not labels and frame.get('bboxes_text') and frame['bboxes_text'].strip():
            rects, labels, _ = convert_text_to_rects_and_labels(frame['bboxes_text'])

        unique_frame_labels = list(set(labels))
        image_class_map[i] = unique_frame_labels

        if is_classification:
            # Classification-specific metrics
            for cls_name in unique_frame_labels:
                class_counts[cls_name] += 1
            
            labels_per_image_counts[len(unique_frame_labels)] += 1

            # Multi-label co-occurrence pairs
            if len(unique_frame_labels) > 1:
                for c1, c2 in itertools.combinations(sorted(unique_frame_labels), 2):
                    co_occurrence_counts[f"{c1} + {c2}"] += 1
            
            # Raw image aspect ratio calculation
            v_info = video_info_cache.get(video_uuid, {})
            if v_info.get('width', 0) > 0 and v_info.get('height', 0) > 0:
                aspect_ratios.append(round(float(v_info['width']) / float(v_info['height']), 2))
        else:
            # Detection/Segmentation metrics
            objects_per_image.append(len(labels))
            if len(rects) > 1:
                for (idx1, rect1), (idx2, rect2) in itertools.combinations(enumerate(rects), 2):
                    iou = calculate_iou(rect1, rect2)
                    if iou > 0.95:
                        suspicious_pairs.append({
                            'image_index': i, 'iou': iou,
                            'box1_label': labels[idx1], 'box2_label': labels[idx2]
                        })

            for j, rect in enumerate(rects):
                class_counts[labels[j]] += 1
                width, height = int(rect[2] - rect[0]), int(rect[3] - rect[1])

                if width > 0 and height > 0:
                    aspect_ratios.append(width / height)
                    video_info = video_info_cache.get(video_uuid)
                    if video_info and video_info.get('width', 0) > 0 and video_info.get('height', 0) > 0:
                        center_x = (float(rect[0]) + float(rect[2])) / 2.0 / float(video_info['width'])
                        center_y = (float(rect[1]) + float(rect[3])) / 2.0 / float(video_info['height'])
                        center_points.append({'x': center_x, 'y': center_y})
                    all_bboxes_for_outliers.append(
                        {'id': f'{video_uuid}_{frame_number}_{j}', 'image_index': i, 'area': width * height,
                         'aspect_ratio': width / height})

    # 随机抽样计算亮度分布 (最多200帧)
    if all_frames:
        sample_size = min(len(all_frames), 200)
        sampled_frames = random.sample(all_frames, sample_size)
        for frame in sampled_frames:
            try:
                frame_path = file_storage.get_frame_path(frame['video_uuid'], frame['frame_number'])
                img_gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is not None:
                    small = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_NEAREST)
                    brightness_levels.append(float(np.mean(small)))
            except Exception:
                pass

    annotator_stats = {}
    all_tasks = [task for vid_tasks in tasks_by_video.values() for task in vid_tasks]
    for task in all_tasks:
        user = task['assigned_to']
        if user not in annotator_stats:
            annotator_stats[user] = {'image_count': 0, 'class_counts': Counter()}

    user_frame_sets = {user: set() for user in annotator_stats.keys()}
    for frame in all_frames:
        processed_users = set()
        for task in all_tasks:
            if task['video_uuid'] == frame['video_uuid'] and task['start_frame'] <= frame['frame_number'] <= task[
                'end_frame']:
                user = task['assigned_to']
                user_frame_sets[user].add(f"{frame['video_uuid']}_{frame['frame_number']}")
                if user not in processed_users:
                    frame_labels = []
                    if frame.get('annotations_json') and frame['annotations_json'].strip():
                        try:
                            ann_data = AnnotationData.from_json(frame['annotations_json'])
                            frame_labels = ann_data.classifications or [o.label for o in ann_data.objects if o.label]
                        except Exception:
                            pass
                    if not frame_labels and frame.get('bboxes_text'):
                        frame_labels = extract_labels(frame['bboxes_text'])

                    annotator_stats[user]['class_counts'].update(frame_labels)
                    processed_users.add(user)

    for user, frame_set in user_frame_sets.items():
        annotator_stats[user]['image_count'] = len(frame_set)

    gallery_images = [{
        'original_url': f"/media/frames/{f['video_uuid']}/frame_{f['frame_number']:05d}.jpg",
        'video': f['video_description'], 'frame': f['frame_number'], 'video_uuid': f['video_uuid'],
        'task_uuid': get_task_for_frame(f['video_uuid'], f['frame_number']),
        'tags': image_class_map.get(i, [])
    } for i, f in enumerate(all_frames)]

    warnings = []
    total_instances = sum(class_counts.values())
    if class_counts:
        avg_instances = total_instances / len(class_counts)
        for class_name, count in class_counts.items():
            if count < 10 or count < avg_instances * 0.1:
                warnings.append(
                    f"<b>Class Imbalance:</b> Class '{class_name}' has very few images ({count}), which may affect classification performance.")

    if not is_classification:
        small_object_threshold = 100
        small_object_count = sum(1 for bbox in all_bboxes_for_outliers if bbox['area'] < small_object_threshold)
        if small_object_count > 0:
            warnings.append(
                f"<b>Small Object Warning:</b> Found {small_object_count} objects with an area smaller than {small_object_threshold} pixels. Please check if they are labeling errors.")

        if suspicious_pairs:
            warnings.append(
                f"<b>Potential Duplicate Warning:</b> Found {len(suspicious_pairs)} pairs of bounding boxes with high overlap (IoU > 0.95), which might be duplicate labels.")

    if is_classification:
        multi_label_images = sum(c for k, c in labels_per_image_counts.items() if k > 1)
        summary_text = f"This Classification dataset contains <strong>{len(class_counts)}</strong> unique classes across <strong>{len(all_frames)}</strong> labeled images. (Multi-label images: <strong>{multi_label_images}</strong>)"
    else:
        summary_text = f"This dataset contains <strong>{len(class_counts)}</strong> classes, with a total of <strong>{total_instances}</strong> instances across <strong>{len(all_frames)}</strong> labeled images."

    return jsonify({
        'success': True,
        'is_classification': is_classification,
        'summary_text': summary_text,
        'warnings': warnings,
        'class_counts': dict(class_counts),
        'labels_per_image': dict(labels_per_image_counts),
        'co_occurrence': dict(co_occurrence_counts),
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

@app.route('/api/datasetAnalysis/<dataset_uuid>/consistency_check', methods=['POST'])
def run_consistency_check(dataset_uuid):
    settings = settings_manager.load_settings()
    if not settings.get('enable_feature_extractor', True):
        return jsonify({
            'success': False,
            'message': 'SAM3 retrieval features are disabled in system settings to save resources.'
        }), 501

    try:
        request_data = request.get_json() or {}
        is_color_check_enabled = request_data.get('enable_color_check', True)

        if is_color_check_enabled:
            logging.info("Starting AI Quality Control with SEMANTIC (SAM3) and COLOR checks.")
        else:
            logging.info("Starting AI Quality Control with SEMANTIC (SAM3) check ONLY.")

        # 旧实现: 已标注框 -> MobileNet 语义向量 + HSV 颜色直方图 -> 类内/类间相似度找
        #        离群点。这一整段逻辑（含 embedding 提取、按类别建 prototype_library、
        #        余弦相似度比较）已经移进 ai_models.check_dataset_consistency，改用
        #        SAM3 自己的开放词汇置信度分数判断"标得对不对"，不再需要 embedding。
        outlier_image_indices, all_bboxes_info, message = ai_models.check_dataset_consistency(
            dataset_uuid, enable_color_check=is_color_check_enabled
        )

        if outlier_image_indices is None:
            return jsonify({'success': False, 'message': message}), 404

        return jsonify({
            'success': True,
            'message': message,
            'outlier_image_indices': list(outlier_image_indices)
        })

    except Exception as e:
        logging.error(f"On-demand consistency check failed: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'审查失败: {e}'}), 500


@app.route('/lam_predict', methods=['POST'])
def lam_predict_route():
    settings = settings_manager.load_settings()
    if not settings.get('enable_feature_extractor', True):
        return jsonify({
            'success': False,
            'message': 'SAM3 retrieval features are disabled in system settings to save resources.'
        }), 501

    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = data.get('frame_number')
    point = data.get('point')

    if not all([video_uuid, frame_number is not None, point]):
        return jsonify({'success': False, 'message': 'Missing required request parameters.'}), 400

    try:
        point_coords = (int(point['x']), int(point['y']))
        result, error_msg = ai_models.lam_predict(video_uuid, int(frame_number), point_coords)

        if error_msg:
            return jsonify({'success': False, 'message': error_msg})

        return jsonify({'success': True, **result})

    except Exception as e:
        logging.error(f"LAM 预测失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Internal Server Error: {str(e)}'}), 500


@app.route('/api/previewAugmentations', methods=['POST'])
def preview_augmentations():
    if not background_tasks.A:
        return jsonify({'success': False, 'message': 'Albumentations library not installed on server.'}), 501

    data = request.json
    video_uuid = data.get('video_uuid')
    frame_number = data.get('frame_number')
    augmentation_options = data.get('augmentation_options')
    sample_pool = data.get('sample_pool')

    if not all([video_uuid, frame_number is not None, augmentation_options]):
        return jsonify({'success': False, 'message': 'Missing required data.'}), 400

    try:
        if augmentation_options.get('mosaic', {}).get('enabled'):
            if not sample_pool:
                return jsonify({'success': False, 'message': 'Sample pool is required for Mosaic preview.'}), 400
            if random.random() < augmentation_options['mosaic'].get('p', 1.0):
                previews, err_msg = generate_mosaic_previews(sample_pool, video_uuid, frame_number)
                if err_msg:
                    return jsonify({'success': False, 'message': err_msg}), 400
                return jsonify({'success': True, 'previews': previews})

        frame_path = file_storage.get_frame_path(video_uuid, frame_number)
        if not os.path.exists(frame_path):
            return jsonify({'success': False, 'message': 'Frame image not found.'}), 404

        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_info = database.get_video_entity(video_uuid)

        frames = database.get_video_frames(video_uuid)
        frame_item = next((f for f in frames if f['frame_number'] == frame_number), None)

        if not frame_item:
            return jsonify({'success': False, 'message': 'Frame data not found in database.'}), 404

        all_labels = database.get_all_class_labels()
        class_map = {name: i for i, name in enumerate(all_labels)}

        # 1. 优先解析 annotations_json 分割多边形
        polygons_data = []
        if frame_item.get('annotations_json') and frame_item['annotations_json'].strip():
            from annotation_model import AnnotationData, AnnotationObject
            ann_data = AnnotationData.from_json(frame_item['annotations_json'])
            for obj in ann_data.objects:
                if obj.type == 'polygon' and obj.polygon:
                    polygons_data.append(obj)
                elif obj.type == 'bbox' and obj.bbox:
                    x1, y1, x2, y2 = obj.bbox
                    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    polygons_data.append(AnnotationObject(id=obj.id, type='polygon', label=obj.label, points=poly))

        if polygons_data:
            from exporters.detection.yolo_detect import build_augmentation_pipeline_for_keypoints
            augmentation_options['mosaic'] = {'enabled': False}
            pipeline = build_augmentation_pipeline_for_keypoints(augmentation_options)
            if not pipeline:
                return jsonify({'success': False, 'message': 'No valid augmentations selected.'}), 400

            flat_kpts = []
            kpt_labels = []
            poly_info = []
            for poly_idx, obj in enumerate(polygons_data):
                poly_pts = obj.polygon
                poly_info.append((obj.label, len(poly_pts)))
                for pt in poly_pts:
                    flat_kpts.append([float(pt[0]), float(pt[1])])
                    kpt_labels.append(poly_idx)

            previews = []
            for _ in range(6):
                transformed = pipeline(image=image_rgb, keypoints=flat_kpts, keypoint_labels=kpt_labels)
                aug_image_rgb = transformed['image']
                aug_kpts = transformed['keypoints']
                aug_kpt_labels = transformed['keypoint_labels']

                img_h, img_w, _ = aug_image_rgb.shape
                vis_image = cv2.cvtColor(aug_image_rgb, cv2.COLOR_RGB2BGR)

                for poly_idx, (label, count) in enumerate(poly_info):
                    pts = [aug_kpts[idx] for idx, l_idx in enumerate(aug_kpt_labels) if l_idx == poly_idx]
                    if len(pts) >= 3:
                        pts_np = np.array([[max(0, min(img_w - 1, int(pt[0]))), max(0, min(img_h - 1, int(pt[1])))] for pt in pts], np.int32).reshape((-1, 1, 2))
                        color = string_to_color_bgr(label)
                        cv2.polylines(vis_image, [pts_np], isClosed=True, color=color, thickness=2)

                        overlay = vis_image.copy()
                        cv2.fillPoly(overlay, [pts_np], color)
                        cv2.addWeighted(overlay, 0.35, vis_image, 0.65, 0, vis_image)

                        x_min = int(np.min(pts_np[:, 0, 0]))
                        y_min = int(np.min(pts_np[:, 0, 1]))
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(vis_image, (x_min, max(0, y_min - text_h - 5)), (x_min + text_w, y_min), color, -1)
                        cv2.putText(vis_image, label, (x_min, max(text_h, y_min - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                _, buffer = cv2.imencode('.jpg', vis_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                previews.append(f"data:image/jpeg;base64,{img_base64}")

            return jsonify({'success': True, 'previews': previews})

        # 2. 备用解析传统 bboxes_text
        yolo_bboxes = []
        class_indices = []
        if frame_item.get('bboxes_text') and frame_item['bboxes_text'].strip():
            yolo_bboxes, class_indices = file_storage.get_yolo_bboxes(frame_item['bboxes_text'], video_info['width'],
                                                                      video_info['height'], class_map)

        if not yolo_bboxes:
            # 3. 图像分类模式 / 无框图像 的通用图像数据增强预览
            classifications_data = []
            if frame_item.get('annotations_json') and frame_item['annotations_json'].strip():
                from annotation_model import AnnotationData
                try:
                    ann_data = AnnotationData.from_json(frame_item['annotations_json'])
                    classifications_data = ann_data.classifications or []
                except Exception:
                    pass

            from exporters.detection.yolo_detect import build_augmentation_pipeline_for_keypoints
            augmentation_options['mosaic'] = {'enabled': False}
            pipeline = build_augmentation_pipeline_for_keypoints(augmentation_options)

            previews = []
            for _ in range(6):
                if pipeline:
                    transformed = pipeline(image=image_rgb, keypoints=[[10.0, 10.0]], keypoint_labels=[0])
                    aug_image_rgb = transformed['image']
                else:
                    aug_image_rgb = image_rgb.copy()

                vis_image = cv2.cvtColor(aug_image_rgb, cv2.COLOR_RGB2BGR)

                # 若包含分类 Tag，在图像上方绘制亮色 Badge 标签
                if classifications_data:
                    x_offset = 15
                    for cls_tag in classifications_data:
                        (text_w, text_h), _ = cv2.getTextSize(cls_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        color = string_to_color_bgr(cls_tag)
                        cv2.rectangle(vis_image, (x_offset, 15), (x_offset + text_w + 14, 15 + text_h + 14), color, -1)
                        cv2.putText(vis_image, cls_tag, (x_offset + 7, 15 + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        x_offset += text_w + 24

                _, buffer = cv2.imencode('.jpg', vis_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                previews.append(f"data:image/jpeg;base64,{img_base64}")

            return jsonify({'success': True, 'previews': previews})

        augmentation_options['mosaic'] = {'enabled': False}
        pipeline = background_tasks.build_augmentation_pipeline(augmentation_options)
        if not pipeline:
            return jsonify({'success': False, 'message': 'No valid augmentations selected.'}), 400

        def _clamp_yolo(bbox):
            cx, cy, bw, bh = bbox
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
            EPS = 1e-6
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0 - EPS, x2), min(1.0 - EPS, y2)
            nw, nh = x2 - x1, y2 - y1
            return [x1 + nw / 2, y1 + nh / 2, nw, nh]
        yolo_bboxes = [_clamp_yolo(b) for b in yolo_bboxes]
        yolo_bboxes_filtered = []
        class_indices_filtered = []
        track_ids = []
        for i, b in enumerate(yolo_bboxes):
            if b[2] > 0 and b[3] > 0:
                yolo_bboxes_filtered.append(b)
                class_indices_filtered.append(class_indices[i])
                track_ids.append(i)
        yolo_bboxes = yolo_bboxes_filtered
        class_indices = class_indices_filtered
        if not yolo_bboxes:
            return jsonify({'success': False, 'message': 'All bboxes became invalid after clamping.'}), 500

        previews = []
        for _ in range(6):
            transformed = pipeline(image=image_rgb, bboxes=yolo_bboxes, class_labels=class_indices, track_ids=track_ids)
            aug_image_rgb = transformed['image']
            aug_bboxes_yolo = transformed['bboxes']
            aug_labels_indices = transformed['class_labels']
            aug_track_ids = transformed['track_ids']

            from exporters.detection.yolo_detect import tight_fit_bbox
            aug_bboxes_yolo = tight_fit_bbox(yolo_bboxes, aug_bboxes_yolo, orig_ids=track_ids, aug_ids=aug_track_ids)

            h, w, _ = aug_image_rgb.shape
            vis_image = aug_image_rgb.copy()
            for i, bbox in enumerate(aug_bboxes_yolo):
                class_index = int(aug_labels_indices[i])
                class_name = all_labels[class_index]
                color = string_to_color_bgr(class_name)
                x_center, y_center, width_norm, height_norm = bbox
                x1 = int((x_center - width_norm / 2) * w)
                y1 = int((y_center - height_norm / 2) * h)
                x2 = int((x_center + width_norm / 2) * w)
                y2 = int((y_center + height_norm / 2) * h)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', vis_image_bgr)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            previews.append(f"data:image/jpeg;base64,{img_base64}")

        return jsonify({'success': True, 'previews': previews})

    except Exception as e:
        logging.error(f"Augmentation preview failed: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500


def start_server():
    init(autoreset=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    settings = settings_manager.load_settings()
    if settings.get('initial_setup_done', False):
        logging.info("正在初始化AI模型，请稍候...")
        ai_models.startup_ai_models()
    else:
        logging.warning("=== 处于初始配置向导模式：已拦截 AI 模型加载，等待用户完成环境配置 ===")

    # 注: 原来这里有两个 atexit.register，退出时把 "原型库" 和 "预处理特征缓存" 落盘
    # (prototype_library.pt / preprocessed_cache.pt)，是 MobileNet 时代的产物。SAM3
    # 迁移后不再有需要跨进程持久化的 embedding/原型概念，帧级 backbone 缓存本来就只是
    # 一个短期的内存 LRU（退出即丢，下次用到时重新跑一次 backbone 即可），不需要落盘。

    time.sleep(0.01)

    logging.info("=" * 121)
    logging.info(
        "███████╗ ███████╗ ██████╗   ██████╗  ██████╗  ██╗   ██╗  ██████╗  ██╗       ██████╗  ██╗   ██╗  █████╗  ██████╗  ██████╗")
    logging.info(
        "╚══███╔╝ ██╔════╝ ██╔══██╗ ██╔═══██╗ ╚════██╗ ╚██╗ ██╔╝ ██╔═══██╗ ██║      ██╔═══██╗ ╚██╗ ██╔╝ ██╔══██╗ ██╔══██╗ ██╔══██╗")
    logging.info(
        "███╔╝    █████╗   ██████╔╝ ██║   ██║  █████╔╝  ╚████╔╝  ██║   ██║ ██║      ██║   ██║  ╚████╔╝  ███████║ ██████╔╝ ██║  ██║")
    logging.info(
        "███╔╝    ██╔══╝   ██╔══██╗ ██║   ██║ ██╔═══╝    ╚██╔╝   ██║   ██║ ██║      ██║   ██║   ╚██╔╝   ██╔══██║ ██╔══██╗ ██║  ██║")
    logging.info(
        "███████╗ ███████╗ ██║  ██║ ╚██████╔╝ ███████╗    ██║    ╚██████╔╝ ███████╗ ╚██████╔╝    ██║    ██║  ██║ ██║  ██║ ██████╔╝")
    logging.info(
        "╚══════╝ ╚══════╝ ╚═╝  ╚═╝  ╚═════╝  ╚══════╝    ╚═╝     ╚═════╝  ╚══════╝  ╚═════╝     ╚═╝    ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═════╝ ")
    logging.info(
        "Developed by BlueDarkUP from FIRST Tech Challenge team 27570           Be based on -- FIRST Machine Learning Toolchain --")
    logging.info("=" * 121)

    from waitress import serve
    serve(app, host='127.0.0.1', port=5000, threads=max_workers_setting)


if __name__ == '__main__':
    multiprocessing.freeze_support()

    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()

    time.sleep(2)

    window = webview.create_window(
        title='Zero2YoloYard | Developed by BlueDarkUP from FIRST Tech Challenge team 27570 | Be based on -- FIRST Machine Learning Toolchain --',
        url='http://127.0.0.1:5000',
        width=1920,
        height=1080,
        min_size=(1280, 720),
        background_color='#ffffff'
    )

    webview.start()