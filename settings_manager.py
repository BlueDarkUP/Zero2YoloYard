import json
import os
import logging
import config
import torch

SETTINGS_FILE = os.path.join(config.BASE_DIR, 'settings.json')
DEFAULT_SETTINGS = {
    # 指向 sam2 仓库里的 config 文件路径 (相对路径或绝对路径)
    # 注意：确保你的项目中能找到 configs/sam2.1/sam2.1_hiera_l.yaml
    "sam_model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam_model_checkpoint": "sam2.1_hiera_large.pt",

    "feature_extractor_model_name": "mobilenet_v3_large",
    "gpu_device": "auto",

    # SAM2 自动掩码生成的参数
    "sam_mask_confidence": 0.70,  # pred_iou_thresh
    "nms_iou_threshold": 0.7,  # stability_score_thresh

    # 原型与追踪参数
    "prototype_temperature": 0.07,
    "prototype_sample_limit": 50,
    "default_preannotation_conf": 0.5,
    "default_opencv_tracker": "CSRT",
    "frame_extraction_jpeg_quality": 75,
    "default_annotation_mode": "manual",
    "autosave_enabled": False,
    "cache_save_interval_seconds": 30,
    "class_colors": {}
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
            for key, value in DEFAULT_SETTINGS.items():
                if key not in settings:
                    settings[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if key in settings and isinstance(settings[key], dict) and sub_key not in settings[key]:
                            settings[key][sub_key] = sub_value
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