import json
import os
import logging
import config
import torch

SETTINGS_FILE = os.path.join(config.BASE_DIR, 'settings.json')
DEFAULT_SETTINGS = {
    "sam_model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam_model_checkpoint": "sam2.1_t.pt",
    "feature_extractor_model_name": "mobilenet_v3_large",
    "gpu_device": "auto",

    "sam_mask_confidence": 0.70,
    "nms_iou_threshold": 0.7,
    "prototype_temperature": 0.07,
    "prototype_sample_limit": 50,
    "default_preannotation_conf": 0.5,
    "default_opencv_tracker": "CSRT",
    "frame_extraction_jpeg_quality": 75,
    "default_annotation_mode": "manual",
    "autosave_enabled": False,
    "cache_save_interval_seconds": 30,
    "class_colors": {},

    # ================= 新增配置 =================
    # 1. AI 与推理进阶参数
    "inference_size": 512,  # 图像推理分辨率 (512, 640, 1024)
    "default_confidence": 0.5,  # 全局默认置信度
    "sam_box_padding": 0.0,  # SAM 提示框扩展系数 (0.0 - 0.2)

    # 2. 系统资源与并发调度
    "max_workers": 8,  # 最大工作线程数
    "max_cache_size": 30,  # 内存/显存特征缓存最大帧数
    "use_autocast": True,  # 混合精度推理开关 (FP16/BF16)

    # 3. 自动化工作流默认值
    "default_eval_percent": 20.0,  # 默认验证集比例
    "default_test_percent": 10.0,  # 默认测试集比例
    "default_color_check": True,  # 默认是否开启色彩检查
    "color_confusion_factor": 2.0,  # 色彩偏离的警告系数
    "auto_preprocess": True,  # 自动后台预处理特征提取

    # 4. 存储与自动清理
    "auto_cleanup_frames": False,
    "zip_compression": "standard",

    # 5. 模型与功能开关 (Model & Feature Toggles)
    "enable_sam_model": True,  # 总开关：启用SAM系列功能 (点选、追踪、智能选择)
    "enable_feature_extractor": True  # 总开关：启用特征提取功能 (智能选择、LAM、一致性检查)
}
_device = None


def get_device():
    global _device
    if _device is not None:
        return _device

    settings = load_settings()
    device_setting = settings.get("gpu_device", "auto")

    if device_setting == "auto":
        if torch.cuda.is_available():
            _device = torch.device("cuda:0")
            logging.info("Auto-detected and using CUDA device: cuda:0")
        else:
            _device = torch.device("cpu")
            logging.info("Auto-detected and using CPU.")
    elif "cuda" in device_setting and torch.cuda.is_available():
        try:
            device_id = int(device_setting.split(':')[1])
            if device_id < torch.cuda.device_count():
                _device = torch.device(device_setting)
                logging.info(f"Using specified CUDA device: {device_setting}")
            else:
                _device = torch.device("cuda:0")
                logging.warning(f"Device {device_setting} not found, falling back to cuda:0.")
        except (IndexError, ValueError):
            _device = torch.device("cuda:0")
            logging.warning(f"Invalid CUDA device format '{device_setting}', falling back to cuda:0.")
    else:
        if "cuda" in device_setting:
            logging.warning("CUDA device specified but not available. Falling back to CPU.")
        _device = torch.device("cpu")
        logging.info("Using CPU.")

    return _device


def update_device():
    global _device
    _device = None
    logging.info("Device setting updated. Will re-evaluate on next use.")


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        logging.info(f"Settings file not found. Creating a new one at {SETTINGS_FILE}")
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)

            # 补全缺失的默认设置
            for key, value in DEFAULT_SETTINGS.items():
                if key not in settings:
                    settings[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if key in settings and isinstance(settings[key], dict) and sub_key not in settings[key]:
                            settings[key][sub_key] = sub_value

            ckpt = settings.get("sam_model_checkpoint", "sam2.1_t.pt")

            if ckpt == "sam2.1_t.pt":
                settings["sam_model_config"] = "configs/sam2.1/sam2.1_hiera_t.yaml"
            elif ckpt == "sam2.1_s.pt":
                settings["sam_model_config"] = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif ckpt == "sam2.1_b.pt":
                settings["sam_model_config"] = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif ckpt == "sam2.1_l.pt":
                settings["sam_model_config"] = "configs/sam2.1/sam2.1_hiera_l.yaml"

            return settings

    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load settings file: {e}. Returning default settings.")
        return DEFAULT_SETTINGS


def save_settings(settings_data):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings_data, f, indent=4)
        return True
    except IOError as e:
        logging.error(f"Failed to save settings file: {e}")
        return False