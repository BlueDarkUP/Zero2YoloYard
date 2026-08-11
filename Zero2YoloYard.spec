# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 修复 PyTorch 与 TensorFlow 同时加载引发的 OpenMP / C++ 0xC0000005 (-1073741819) 内存越界崩溃
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. 突破 PyInstaller 深度递归解算限制
sys.setrecursionlimit(10000)

# ==============================================================
# Pywebview 原生桌面窗口底层平台依赖
# ==============================================================
hidden_webview = [
    'webview',
    'webview.platforms.winforms',
    'webview.platforms.edgechromium',
    'webview.platforms.cef',
    'clr',
]

# 动态将 gkdt_engine 及其子目录追加到 sys.path 末尾，避免子模块递归收集时找不到内部相对导入，且不影响根目录 config.py 优先导入
gkdt_path = os.path.abspath('gkdt_engine')
test_rw_path = os.path.join(gkdt_path, 'test_real_world')
for p in [gkdt_path, test_rw_path]:
    if p not in sys.path:
        sys.path.append(p)

# ==============================================================
# 自动递归收集本地核心包与关键扩展模块子模块
# ==============================================================
submodules_to_collect = (
    collect_submodules('sam2') +
    collect_submodules('sam3') +
    collect_submodules('gkdt_engine') +
    collect_submodules('exporters') +
    collect_submodules('ultralytics') +
    collect_submodules('pycocotools') +
    collect_submodules('skimage')
)

# ==============================================================
# 精准隐式依赖导入 (包含 PyTorch/TensorFlow/SAM2/SAM3/GKDT/CLIP/DINOv3 等)
# ==============================================================
my_hidden_imports = [
    # Core PyTorch & TensorFlow Frameworks (包含 FTC Machine Learning TFLite 解释器)
    'torch',
    'torchvision',
    'tensorflow',
    'tensorflow.lite.python.interpreter',
    'cv2',
    'numpy',
    'PIL',
    'PIL.Image',

    # HuggingFace & Vision Foundation Backbones
    'transformers',
    'huggingface_hub',
    'timm',
    'einops',
    'safetensors',
    'accelerate',
    'sentencepiece',
    'tokenizers',
    'peft',

    # SAM 2 & SAM 3 Foundation Models
    'ultralytics',
    'sam2',
    'sam3',

    # GKDT (Grounded Keypoint Transformer) & DINOv3 Engine Dependencies
    'gkdt_engine',
    'albumentations',
    'hydra',
    'omegaconf',
    'fairscale',
    'shapely',
    'tensorboardX',
    'termcolor',
    'torchmetrics',
    'lmdb',
    'h5py',

    # Image Processing, Visualization & COCO Tools (解决 SAM3/GroundingDINO 可视化依赖)
    'pycocotools',
    'pycocotools.mask',
    'pycocotools.coco',
    'pycocotools.cocoeval',
    'pandas',
    'skimage',
    'skimage.io',
    'skimage.color',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.patches',
    'matplotlib.colors',

    # Text Tokenizers & Path Managers for SAM3/DINOv3
    'ftfy',
    'regex',
    'iopath',
    'pkg_resources',
    'setuptools',

    # Video Decoding & Object Detection Utilities
    'decord',
    'supervision',
    'addict',
    'yacs',
    'tqdm',

    # Scipy & Scikit-Learn Clustering Utilities
    'scipy',
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
    'scipy.ndimage',
    'scipy.stats',
    'sklearn',
    'sklearn.cluster',
    'sklearn.metrics',
    'sklearn.utils._typedefs',
    'sklearn.neighbors._partition_nodes',

    # Flask Backend & SQLite Database Engine
    'flask',
    'waitress',
    'engineio.async_drivers.threading',
    'colorama',
    'yaml',
    'sqlite3',
    'sqlalchemy',
    'sqlalchemy.dialects.sqlite',

    # Exporters Engine Extensions
    'exporters',
] + submodules_to_collect

final_hidden_imports = my_hidden_imports + hidden_webview

# 收集 PyWebView 及 AI 模型库数据文件
extra_datas = (
    collect_data_files('webview') +
    collect_data_files('ultralytics') +
    collect_data_files('transformers') +
    collect_data_files('timm') +
    collect_data_files('albumentations') +
    collect_data_files('torchvision')
)

# 智能收集轻量配置，剔除超大权重 (.pt / .pth / .safetensors / .bin / .best)
def collect_light_datas(src_dir, dst_dir):
    result = []
    if not os.path.exists(src_dir):
        return result
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if not f.endswith(('.pt', '.pth', '.safetensors', '.bin', '.best', '.ckpt', '.onnx')):
                rel_path = os.path.relpath(root, src_dir)
                target_dir = os.path.join(dst_dir, rel_path) if rel_path != '.' else dst_dir
                result.append((os.path.join(root, f), target_dir))
    return result

# 基础数据打包路径表
project_datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('configs', 'configs'),
    ('sam2', 'sam2'),
    ('sam3', 'sam3'),
    ('gkdt_engine', 'gkdt_engine'),
    ('exporters', 'exporters'),
] + collect_light_datas('checkpoints', 'checkpoints') + extra_datas

# 动态图标检测 (防止绝对路径垮塌)
icon_file = 'icon.ico' if os.path.exists('icon.ico') else (
    'static/favicon.ico' if os.path.exists('static/favicon.ico') else None
)

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=project_datas,
    hiddenimports=final_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'tkinter',
        'IPython',
        'jupyter',
        'dask',
        'bokeh',
        'numba',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Zero2YoloYard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Zero2YoloYard',
)