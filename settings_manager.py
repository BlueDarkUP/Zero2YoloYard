import json
import os
import logging
import config
import torch

SETTINGS_FILE = os.path.join(config.BASE_DIR, 'settings.json')
DEFAULT_SETTINGS = {
    "initial_setup_done": False,
    "sam_model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam_model_checkpoint": "sam2.1_t.pt",
    "gpu_device": "auto",

    "sam_mask_confidence": 0.70,
    "nms_iou_threshold": 0.7,
    "default_preannotation_conf": 0.5,
    "default_opencv_tracker": "CSRT",
    "frame_extraction_jpeg_quality": 75,
    "default_annotation_mode": "manual",
    "autosave_enabled": False,
    "cache_save_interval_seconds": 30,
    "class_colors": {},

    "inference_size": 512,
    "default_confidence": 0.5,
    "sam_box_padding": 0.0,

    "max_workers": 8,
    "max_cache_size": 30,
    "use_autocast": True,

    "default_eval_percent": 20.0,
    "default_test_percent": 10.0,
    "color_confusion_factor": 2.0,
    "consistency_semantic_threshold": 0.05,
    "consistency_confusion_margin": 0.15,
    "auto_preprocess": True,

    "auto_cleanup_frames": False,
    "zip_compression": "standard",

    "enable_sam_model": True,
    "enable_feature_extractor": True
}
_device = None
_cached_settings = None
_cached_mtime = 0


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
    global _cached_settings, _cached_mtime
    if not os.path.exists(SETTINGS_FILE):
        logging.info(f"Settings file not found. Creating a new one at {SETTINGS_FILE}")
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    try:
        current_mtime = os.path.getmtime(SETTINGS_FILE)
        if _cached_settings is not None and _cached_mtime == current_mtime:
            return dict(_cached_settings)

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

            _cached_settings = settings
            _cached_mtime = current_mtime
            return dict(settings)

    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load settings file: {e}. Returning default settings.")
        return DEFAULT_SETTINGS.copy()


def save_settings(settings_data):
    global _cached_settings, _cached_mtime
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings_data, f, indent=4)
        _cached_mtime = os.path.getmtime(SETTINGS_FILE) if os.path.exists(SETTINGS_FILE) else 0
        _cached_settings = dict(settings_data)
        return True
    except IOError as e:
        logging.error(f"Failed to save settings file: {e}")
        return False