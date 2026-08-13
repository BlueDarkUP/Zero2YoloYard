# gkdt_tasks.py - Zero2YoloYard 与 GKDT 姿态 AI 引擎适配器
import os
import sys
import time
import json
import logging
import uuid
import inspect
import threading
import torch
import cv2
import numpy as np

# 1. 先导入根目录基础模块，确保 sys.modules['config'] 锁定为根目录配置
import config
import file_storage
import database
import settings_manager

# 2. 动态将 gkdt_engine 及其子目录追加到 Python 模块搜索路径末尾
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GKDT_ENGINE_DIR = os.path.join(PROJECT_ROOT, "gkdt_engine")
TEST_REAL_WORLD_DIR = os.path.join(GKDT_ENGINE_DIR, "test_real_world")

for p in [GKDT_ENGINE_DIR, TEST_REAL_WORLD_DIR]:
    if p not in sys.path:
        sys.path.append(p)

_gkdt_model_cache = {
    "model": None,
    "device": None,
    "model_type": None
}


# os.chdir() 是进程全局操作，在多线程 Flask 中必须串行化，否则会产生竞争条件
_WORKING_DIR_LOCK = threading.Lock()


class WorkingDirContext:
    """上下文管理器：临时切换工作目录到 gkdt_engine，确保 GKDT 内部相对路径正确。
    使用模块级锁保证多线程环境下 os.chdir() 的串行执行，避免竞争条件。"""

    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.prev_dir = None

    def __enter__(self):
        _WORKING_DIR_LOCK.acquire()
        self.prev_dir = os.getcwd()
        os.chdir(self.target_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.prev_dir:
                os.chdir(self.prev_dir)
        finally:
            _WORKING_DIR_LOCK.release()


def _apply_dinov3_bpe_path_patch():
    """动态热补丁：修复 DINOv3 源码在 Windows 环境下将 C:\ 本地盘符误识别为 URL Scheme ('c') 的 Bug"""
    local_bpe_path = os.path.join(GKDT_ENGINE_DIR, "network", "dinov3_kd", "bpe_simple_vocab_16e6.txt.gz")
    try:
        from io import BytesIO
        # noinspection PyUnresolvedReferences
        import dinov3.eval.text.tokenizer as dinov3_tokenizer
        _orig_get_tokenizer = dinov3_tokenizer.get_tokenizer

        def _patched_get_tokenizer(bpe_path_or_url=None, *args, **kwargs):
            target_path = bpe_path_or_url
            if not target_path or not os.path.exists(str(target_path)):
                target_path = local_bpe_path

            # 若为本地磁盘文件（包含 Windows 盘符 C:\），直接以二进制读取传入 Tokenizer，绕过 urllib.parse 的 scheme=='c' Bug
            if os.path.exists(str(target_path)):
                with open(target_path, "rb") as f:
                    file_buf = BytesIO(f.read())
                    return dinov3_tokenizer.Tokenizer(vocab_path=file_buf)

            return _orig_get_tokenizer(bpe_path_or_url=target_path, *args, **kwargs)

        dinov3_tokenizer.get_tokenizer = _patched_get_tokenizer
    except Exception as e:
        logging.warning(f"Failed to apply DINOv3 BPE patch: {e}")


def load_gkdt_model(model_type=None):
    """懒加载并缓存 GKDT 模型权重"""
    if model_type is None:
        model_type = settings_manager.load_settings().get("gkdt_model_type", "GKDT-L")
    device = settings_manager.get_device()

    if (_gkdt_model_cache["model"] is not None and
            _gkdt_model_cache["device"] == str(device) and
            _gkdt_model_cache["model_type"] == model_type):
        return _gkdt_model_cache["model"]

    logging.info(f"[GKDT] 正在加载姿态 AI 模型引擎 ({model_type}) 到设备 {device}...")

    cfg_file = os.path.join(GKDT_ENGINE_DIR, "test_real_world", "configs", "gkd.yaml")

    if "H" in model_type:
        ckpt_path = os.path.join(GKDT_ENGINE_DIR, "output", "GKDT-H_for_app", "model", "gkd_fullset.best")
        opts = ["MODEL.ENCODER.DINOv3.VISUAL_ENCODER", "dinov3_vith16plus"]
    else:
        ckpt_path = os.path.join(GKDT_ENGINE_DIR, "output", "GKDT-L_for_app", "model", "gkd_fullset.best")
        opts = ["MODEL.ENCODER.DINOv3.VISUAL_ENCODER", "dinov3_vitl16"]

    try:
        with WorkingDirContext(GKDT_ENGINE_DIR):
            _apply_dinov3_bpe_path_patch()

            try:
                # noinspection PyUnresolvedReferences
                from gkd_inference_lib.gkd_inference import GKDInference
            except ImportError:
                # noinspection PyUnresolvedReferences
                from test_real_world.gkd_inference_lib.gkd_inference import GKDInference

            gkd_engine = GKDInference(
                cfg_file=cfg_file,
                checkpoint_path=ckpt_path,
                opts=opts
            )

        _gkdt_model_cache["model"] = gkd_engine
        _gkdt_model_cache["device"] = str(device)
        _gkdt_model_cache["model_type"] = model_type
        logging.info("[GKDT] 姿态 AI 模型引擎加载成功！")
        return gkd_engine

    except Exception as e:
        logging.error(f"[GKDT] 加载失败: {e}", exc_info=True)
        return None


def predict_pose_from_text(video_uuid, frame_number, class_label, custom_kps_texts=None, bbox=None):
    """
    文本提示关键点自动识别 (Text Prompt Pose Auto-Detection)
    带 SAM 2.1 空间越界熔断与三阶置信度判定 (v=0/1/2)
    """
    gkd_engine = load_gkdt_model()
    if gkd_engine is None:
        raise RuntimeError("GKDT model failed to initialize. Please check the logs.")

    # 0. 规范化 bbox 为 Python 原生 float list，防止 numpy array 导致的布尔隐式转换歧义
    if bbox is not None:
        try:
            bbox = [float(b) for b in list(bbox)]
        except Exception:
            bbox = None

    # 1. 确定关键点名称列表
    kps_texts = custom_kps_texts
    if not kps_texts:
        schema_raw = database.get_class_keypoint_schema(class_label)
        if schema_raw:
            schema_data = json.loads(schema_raw) if isinstance(schema_raw, str) else schema_raw
            kps_texts = schema_data.get('points', [])

    if not kps_texts:
        raise ValueError(f"Class [{class_label}] has no keypoint schema configured. Please configure a keypoint template or enter keypoint names on the right panel!")

    # 2. 读取当前帧图像路径与尺寸
    image_path = file_storage.get_frame_path(video_uuid, frame_number)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_h, img_w = image_bgr.shape[:2]

    # 3. 计算 SAM 2.1 目标区域的合法动作延伸空间 (允许外扩 20% 作为手脚拉伸缓冲区)
    if bbox is not None and len(bbox) == 4:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        pad_x = bw * 0.20
        pad_y = bh * 0.20
        min_x_allowed = max(0, bbox[0] - pad_x)
        min_y_allowed = max(0, bbox[1] - pad_y)
        max_x_allowed = min(img_w, bbox[2] + pad_x)
        max_y_allowed = min(img_h, bbox[3] + pad_y)
    else:
        min_x_allowed, min_y_allowed = 0, 0
        max_x_allowed, max_y_allowed = img_w, img_h

    # 4. 使用 GKDT 官方 demo() 位置参数调用
    with WorkingDirContext(GKDT_ENGINE_DIR):
        try:
            # noinspection PyUnresolvedReferences
            from gkd_inference_lib.gkd_inference import demo
        except ImportError:
            # noinspection PyUnresolvedReferences
            from test_real_world.gkd_inference_lib.gkd_inference import demo

        bbox_input = list(bbox) if (bbox is not None and len(bbox) > 0) else []
        with torch.inference_mode():
            predictions_o, predict_score, w_h_origin = demo(
                gkd_engine,
                image_path,
                bbox_input,
                "",
                [],
                kps_texts
            )

    # 5. 解析预测出的关键点坐标并施加【空间越界熔断】+【三阶置信度判定】
    keypoints_out = []
    xs, ys = [], []

    # 转换 Tensor 到 numpy
    if hasattr(predictions_o, 'cpu'):
        predictions_o = predictions_o.cpu().numpy()
    if hasattr(predict_score, 'cpu'):
        predict_score = predict_score.cpu().numpy()

    # 展平单目标 Batch 维度 (shape [1, N, 2] -> [N, 2])
    if len(np.shape(predictions_o)) == 3:
        predictions_o = predictions_o[0]
    if len(np.shape(predict_score)) == 2:
        predict_score = predict_score[0]

    for idx, name in enumerate(kps_texts):
        if idx < len(predictions_o):
            x = float(predictions_o[idx][0])
            y = float(predictions_o[idx][1])
            score = float(predict_score[idx]) if idx < len(predict_score) else 1.0
        else:
            x, y, score = 0.0, 0.0, 0.0

        # === 核心判定逻辑升级 ===
        # A. 坐标越界检测（超出图像边界，或超出了 SAM 2.1 目标的动作安全区）
        is_out_of_bounds = (x < min_x_allowed or x > max_x_allowed or y < min_y_allowed or y > max_y_allowed)

        # B. 三阶概率与空间双重熔断机制
        if is_out_of_bounds or score < 0.12:
            v = 0  # ⚪ 不存在 / 画面外 (Absent)
        elif score >= 0.38:
            v = 2  # 🟢 清晰可见 (Visible)
        else:
            v = 1  # 🟠 躯体遮挡 (Occluded)

        keypoints_out.append({
            'name': name,
            'x': round(x, 2),
            'y': round(y, 2),
            'v': v
        })

        # 只有存在或遮挡的点才参与包围框重新计算
        if v > 0:
            xs.append(x)
            ys.append(y)

    # 计算最终包围框
    calc_bbox = None
    if xs and ys:
        pad = 15
        calc_bbox = [
            max(0, int(min(xs) - pad)),
            max(0, int(min(ys) - pad)),
            min(img_w, int(max(xs) + pad)),
            min(img_h, int(max(ys) + pad))
        ]

    pose_object = {
        "id": f"pose_{int(time.time() * 1000)}_{uuid.uuid4().hex[:4]}",
        "type": "keypoint",
        "label": class_label,
        "bbox": calc_bbox or bbox,
        "keypoints": keypoints_out
    }

    return pose_object


def predict_pose_from_sam_point(video_uuid, frame_number, class_label, point_coords, custom_kps_texts=None):
    """
    SAM 2.1 + GKDT 级联姿态生成 (带二段精准隔离)
    """
    import ultralytics_sam_tasks

    image_path = file_storage.get_frame_path(video_uuid, frame_number)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    # 1. SAM 2.1 点选分割，获取精准 Instance BBox
    sam_res = ultralytics_sam_tasks.predict_box_from_point_ultralytics(image_path, point_coords)
    if not sam_res:
        raise ValueError("SAM 2.1 failed to detect a valid object at the click position. Please click near the object center!")

    sam_bbox = [sam_res['x1'], sam_res['y1'], sam_res['x2'], sam_res['y2']]

    # 2. 传入带有越界熔断器的 GKDT 函数
    pose_object = predict_pose_from_text(
        video_uuid=video_uuid,
        frame_number=frame_number,
        class_label=class_label,
        custom_kps_texts=custom_kps_texts,
        bbox=sam_bbox
    )

    if 'polygon' in sam_res and sam_res['polygon']:
        pose_object['polygon'] = sam_res['polygon']

    return pose_object


def predict_sam3_gkdt_batch_pose(video_uuid, frame_number, class_label, text_prompt=None, confidence=0.25, custom_kps_texts=None):
    """
    SAM3 开放词汇盲扫 + GKDT 级联全图多目标姿态识别 (TrueLAM Pose)
    1. 使用 SAM3 输入开放词汇文本 query 盲扫全图，提取画面中所有对应目标的 BBox 列表；
    2. 针对每一个 SAM3 识别出的目标 BBox，传入 GKDT 提取高精度骨架关键点 (带 20% 空间缓冲区与越界熔断)；
    3. 返回生成的所有姿态对象列表。
    """
    import ultralytics_sam_tasks

    prompt = text_prompt
    if not prompt:
        try:
            prompt = database.get_class_sam3_prompt(class_label)
        except Exception:
            pass
    if not prompt:
        prompt = class_label

    logging.info(f"[TrueLAM Pose] 正在使用 SAM3 盲扫全图 (Prompt: '{prompt}', Confidence: {confidence})...")

    # 1. SAM3 全图开放词汇目标检索
    sam3_res = ultralytics_sam_tasks.sam3_query_frame(
        video_uuid=video_uuid,
        frame_number=int(frame_number),
        text_prompt=prompt,
        confidence=confidence
    )

    if not sam3_res:
        logging.info(f"[TrueLAM Pose] SAM3 未能识别到符合 '{prompt}' 的目标。")
        return []

    logging.info(f"[TrueLAM Pose] SAM3 找到 {len(sam3_res)} 个目标，开始批量级联 GKDT 识别...")

    pose_objects = []
    for idx, item in enumerate(sam3_res):
        box = item.get("box")  # [x1, y1, x2, y2]
        score = item.get("score", 1.0)
        if box and len(box) == 4:
            try:
                pose_obj = predict_pose_from_text(
                    video_uuid=video_uuid,
                    frame_number=frame_number,
                    class_label=class_label,
                    custom_kps_texts=custom_kps_texts,
                    bbox=box
                )
                pose_obj['confidence'] = round(float(score), 4)
                pose_objects.append(pose_obj)
            except Exception as e:
                logging.warning(f"[TrueLAM Pose] 目标 #{idx+1} BBox {box} GKDT 姿态识别跳过: {e}")

    return pose_objects