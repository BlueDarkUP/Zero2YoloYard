# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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

# ==============================================================
# 精准隐式依赖导入 (包含 PyTorch/TensorFlow/SAM2/SAM3/GKDT/CLIP 等)
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

    # SAM 2 & SAM 3 Foundation Models
    'ultralytics',
    'sam2',
    'sam2.build_sam',
    'sam2.automatic_mask_generator',
    'sam2.sam2_image_predictor',
    'sam2.sam2_video_predictor',
    'sam3',
    'sam3.model_builder',
    'sam3.visualization_utils',

    # GKDT (Grounded Keypoint Transformer) & DINOv3 Engine
    'gkdt_engine',
    'albumentations',
    'hydra',
    'omegaconf',

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
    'exporters.base',
    'exporters.detection.yolo_detect',
    'exporters.detection.coco_detect',
    'exporters.detection.pascal_voc',
    'exporters.segmentation.yolo_seg',
    'exporters.segmentation.semantic_mask',
    'exporters.pose.yolo_pose',
    'exporters.pose.coco_pose',
    'exporters.classification.yolo_cls',
    'exporters.classification.folder_class',
]

final_hidden_imports = my_hidden_imports + hidden_webview

# 收集 PyWebView 数据文件
extra_datas = collect_data_files('webview')

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
        'matplotlib',
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