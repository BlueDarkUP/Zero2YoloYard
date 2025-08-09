import json
import os
import logging
import config

SETTINGS_FILE = os.path.join(config.BASE_DIR, 'settings.json')
DEFAULT_SETTINGS = {
    "sam_model_name": "SAM 2.1 Tiny",
    "sam_model_checkpoint": "sam2.1_t.pt",
    "frame_extraction_jpeg_quality": 75,
    "default_opencv_tracker": "CSRT"
}

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