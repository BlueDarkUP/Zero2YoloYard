# config.py

import os
import sys

if getattr(sys, 'frozen', False):
    # 如果是打包后的 exe，则基目录是 exe 文件所在的目录
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # 如果是正常的 python 环境，则基目录是.py文件所在的目录
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATABASE_FILE = os.path.join(BASE_DIR, 'ftc_ml.db')
STORAGE_DIR = os.path.join(BASE_DIR, 'local_storage')



MAX_DESCRIPTION_LENGTH = 30
MAX_VIDEO_SIZE_MB = 10000
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1000 * 1000
MAX_VIDEO_LENGTH_SECONDS = 120
MAX_FRAMES_PER_VIDEO = 10000
MAX_VIDEO_RESOLUTION_WIDTH = 3840
MAX_VIDEO_RESOLUTION_HEIGHT = 2160
MAX_DATASETS_PER_TEAM = 50
MAX_VIDEOS_PER_TEAM = 50
TRACKER_FNS = [
    'CSRT', 'MedianFlow', 'MIL', 'MOSSE', 'TLD', 'KCF', 'Boosting',
]

def get_limit_data_for_render_template():
    return {
        'MAX_VIDEO_SIZE_BYTES': MAX_VIDEO_SIZE_BYTES,
        'MAX_VIDEO_SIZE_MB': MAX_VIDEO_SIZE_MB,
        'MAX_VIDEO_LENGTH_SECONDS': MAX_VIDEO_LENGTH_SECONDS,
        'MAX_FRAMES_PER_VIDEO': MAX_FRAMES_PER_VIDEO,
        'MAX_DESCRIPTION_LENGTH': MAX_DESCRIPTION_LENGTH,
    }