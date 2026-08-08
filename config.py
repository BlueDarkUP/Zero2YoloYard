import os
import sys

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATABASE_FILE = os.path.join(BASE_DIR, 'ftc_ml.db')
STORAGE_DIR = os.path.join(BASE_DIR, 'local_storage')
# 注：原来这里还有 PROTOTYPE_FILE / PREPROCESSED_CACHE_FILE 两个常量，是给 MobileNet
# 时代的"类别原型库"和"预处理特征缓存"落盘用的。SAM3 迁移后这两个概念都不存在了
# （见 ai_models.py 顶部说明），随对应的磁盘缓存函数一起删除。


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

# 注：原来这里还有 ONNX_MODELS_DIR / MOBILENET_LARGE_ONNX / MOBILENET_SMALL_ONNX 三个
# 常量。核查过仓库全部源码，这三个常量在删除前就已经没有任何代码引用（是更早期 ONNX
# 版本特征提取器的遗留死代码），随这次重构一并清理。
