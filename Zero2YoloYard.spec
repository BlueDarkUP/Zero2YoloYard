# -*- mode: python ; coding: utf-8 -*-
# Zero2YoloYard PyInstaller Spec File
# 特性：
# 1. 严格排除所有模型大文件（SAM 2.1、SAM 3、CLIP、GKDT、GroundingDINO、LocateAnything 等权重），
#    保证打包出的单文件 EXE 仅包含程序主体代码与必需结构配置，体积轻巧；
# 2. 模型支持用户在运行 EXE 后通过设置页面（Settings -> Local Model Management）一键断点续传/镜像下载，
#    或手动将下载好的模型放置于 EXE 同级 checkpoints / gkdt_engine 目录下；
# 3. 完整收集 Flask 模板静态资源、SAM2/Hydra 配置文件、SAM3 分词器词表、DINOv3/CLIP 词表、
#    pywebview/pythonnet/EdgeChromium 运行时、decord 视音频解码库、torchvision/albumentations 等动态链接库与元数据；
# 4. 彻底规避旧打包中因盲目 exclude unittest/pydoc 导致 scipy/numpy.testing 崩溃的 Bug；
# 5. 适配 TensorFlow Lite hook，彻底避免与 PyTorch 的 OpenMP 0xC0000005 内存越界崩溃。

import os
import sys

# 彻底避免 PyTorch + TensorFlow / scikit-learn OpenMP 运行时冲突导致的 0xC0000005 崩溃
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_all,
)

# 动态获取当前项目根目录
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__) if '__file__' in locals() else '.')

# ------------------------------------------------------------------------------
# 1. 收集项目内部非权重资产（模板、静态文件、配置文件、词表、代码架构）
# ------------------------------------------------------------------------------
def get_project_datas():
    datas = []

    # 1.1 前端资源与根目录配置
    for sub in ['templates', 'static', 'configs']:
        base = os.path.join(PROJECT_DIR, sub)
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for f in files:
                    src = os.path.join(root, f)
                    rel = os.path.relpath(src, PROJECT_DIR)
                    dst = os.path.dirname(rel)
                    datas.append((src, dst))

    # 1.2 sam2 架构配置文件（Hydra 推理必需）
    sam2_cfg = os.path.join(PROJECT_DIR, 'sam2', 'configs')
    if os.path.isdir(sam2_cfg):
        for root, _, files in os.walk(sam2_cfg):
            for f in files:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, PROJECT_DIR)
                dst = os.path.dirname(rel)
                datas.append((src, dst))

    # 1.3 sam3 词表、资产与源码（保证 TorchScript / inspect.getsource 在单文件运行环境下可访问 .py 源码）
    sam3_dir = os.path.join(PROJECT_DIR, 'sam3')
    if os.path.isdir(sam3_dir):
        for root, dirs, files in os.walk(sam3_dir):
            dirs[:] = [d for d in dirs if d not in ('__pycache__',)]
            for f in files:
                if f.endswith(('.py', '.txt', '.gz', '.yaml', '.yml', '.json')):
                    src = os.path.join(root, f)
                    rel = os.path.relpath(src, PROJECT_DIR)
                    dst = os.path.dirname(rel)
                    datas.append((src, dst))

    # 1.4 gkdt_engine 源码与非权重配置（严格过滤大模型权重与冗余测试图片，保留结构配置文件与词表）
    # 绝对排除：.pt, .pth, .safetensors, .best, .bin, .tmp, output 目录, .cache 目录, assets 目录, ims1 目录, __pycache__
    model_exts = {'.pt', '.pth', '.safetensors', '.best', '.bin', '.tmp', '.pyc', '.jpg', '.jpeg', '.png'}
    gkdt_dir = os.path.join(PROJECT_DIR, 'gkdt_engine')
    if os.path.isdir(gkdt_dir):
        for root, dirs, files in os.walk(gkdt_dir):
            dirs[:] = [d for d in dirs if d not in ('output', '.cache', '__pycache__', 'assets', 'ims1')]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in model_exts:
                    continue
                src = os.path.join(root, f)
                rel = os.path.relpath(src, PROJECT_DIR)
                dst = os.path.dirname(rel)
                datas.append((src, dst))

    return datas

project_datas = get_project_datas()

# ------------------------------------------------------------------------------
# 2. 收集第三方包的关键资源、动态库与隐藏模块
# ------------------------------------------------------------------------------
extra_datas = []
extra_binaries = []
extra_hiddenimports = []

# 桌面 WebView 与 .NET CLR 桥接、视频解码库
for pkg in ['pythonnet', 'clr_loader', 'webview', 'decord']:
    try:
        d, b, h = collect_all(pkg)
        extra_datas.extend(d)
        extra_binaries.extend(b)
        extra_hiddenimports.extend(h)
    except Exception as e:
        print(f"Warning collecting {pkg}: {e}")

# 图像处理与视觉模型包配置
for pkg in ['albumentations', 'timm', 'ultralytics']:
    try:
        extra_datas.extend(collect_data_files(pkg))
    except Exception as e:
        print(f"Warning collecting data files for {pkg}: {e}")

# ------------------------------------------------------------------------------
# 3. 完整隐藏导入清单（严禁排除任何 Python 标准库）
# ------------------------------------------------------------------------------
hidden_imports = [
    # 标准库核心
    'unittest',
    'pydoc',
    'multiprocessing',
    'multiprocessing.freeze_support',
    'sqlite3',
    'ctypes',
    
    # Web 服务与网络
    'waitress',
    'flask',
    'jinja2',
    'werkzeug',
    'colorama',
    'requests',
    'urllib3',
    'tqdm',
    
    # 数据库
    'sqlalchemy',
    'sqlalchemy.dialects.sqlite',
    'sqlalchemy.dialects.sqlite.pysqlite',
    
    # 桌面 GUI 引擎
    'webview',
    'webview.platforms.winforms',
    'webview.platforms.edgechromium',
    'clr',
    'clr_loader',
    'pythonnet',
    
    # 图像与视觉计算
    'cv2',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'skimage',
    'skimage.io',
    'skimage.io._plugins',
    'skimage.io._plugins.pil_plugin',
    'skimage.color',
    'skimage.color.colorconv',
    'scipy',
    'scipy.special',
    'scipy.special._special_ufuncs',
    'scipy.linalg',
    'scipy.ndimage',
    'albumentations',
    'supervision',
    'lapx',
    'decord',
    'matplotlib',
    'matplotlib.pyplot',
    
    # 深度学习框架与大模型生态
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchvision.ops',
    'torchvision.transforms',
    'yaml',
    'timm',
    'einops',
    'transformers',
    'huggingface_hub',
    'iopath',
    'fairscale',
    'tensorboardX',
    'torchmetrics',
    'lmdb',
    'h5py',
    'addict',
    'yacs',
    'ultralytics',
    'pycocotools',
    'shapely',
    'hydra',
    'hydra.core.global_hydra',
    'omegaconf',
    'tensorflow.lite.python.interpreter',

    # 项目自身模块
    'config',
    'database',
    'file_storage',
    'settings_manager',
    'ai_models',
    'local_model_manager',
    'background_tasks',
    'bbox_writer',
    'annotation_model',
    'clip_model',
    'gkdt_tasks',
    'ultralytics_sam_tasks',
    'clean_downloaded_models',
    
    # 数据集导出器模块
    'exporters',
    'exporters.base',
    'exporters.classification',
    'exporters.detection',
    'exporters.pose',
    'exporters.segmentation',
    
    # sam2 与 sam3 核心构建入口
    'sam2',
    'sam2.build_sam',
    'sam2.automatic_mask_generator',
    'sam2.sam2_image_predictor',
    'sam2.sam2_video_predictor',
    'sam3',
    'sam3.model_builder',
    'sam3.model.sam3_image_processor',

    # DINOv3 与 GKDT
    'dinov3',
    'dinov3.hub',
    'dinov3.hub.backbones',
    'dinov3.hub.classifiers',
    'dinov3.hub.detectors',
    'dinov3.hub.dinotxt',
    'dinov3.hub.segmentors',
    'dinov3.hub.depthers',
    'dinov3.hub.utils',
    'dinov3.eval.text.tokenizer',
] + extra_hiddenimports

# 去重
hidden_imports = sorted(list(set(hidden_imports)))

all_datas = project_datas + extra_datas
all_binaries = extra_binaries

# ------------------------------------------------------------------------------
# 4. 分析与打包任务构建
# ------------------------------------------------------------------------------
a = Analysis(
    [os.path.join(PROJECT_DIR, 'app.py')],
    pathex=[
        PROJECT_DIR,
        os.path.join(PROJECT_DIR, 'gkdt_engine'),
        os.path.join(PROJECT_DIR, 'gkdt_engine', 'test_real_world'),
        os.path.join(PROJECT_DIR, 'gkdt_engine', 'network'),
        os.path.join(PROJECT_DIR, 'gkdt_engine', 'network', 'dinov3_kd'),
        os.path.join(PROJECT_DIR, 'gkdt_engine', 'network', 'clip_kd'),
        os.path.join(PROJECT_DIR, 'gkdt_engine', 'test_real_world', 'object_detector_lib'),
    ],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[os.path.join(PROJECT_DIR, 'hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'pytest',
        'polars',
        '_polars_runtime_32',
        'tensorboard',
        'tensorboard.compat',
        'tensorflow.python.profiler',
    ],
    noarchive=False,
    optimize=0,
)

# ------------------------------------------------------------------------------
# 4.5 手术级体积精简过滤：移除无用运行时与冗余素材，保障底层计算库 100% 完整
# ------------------------------------------------------------------------------
excluded_bin_patterns = [
    '_polars_runtime',
    'profiler_plugin',
]

filtered_binaries = []
for item in a.binaries:
    dest_lower = item[0].lower()
    if any(pat in dest_lower for pat in excluded_bin_patterns):
        print(f"[SPEC FILTER] Dropping unnecessary binary: {item[0]}")
        continue
    filtered_binaries.append(item)
a.binaries = filtered_binaries

filtered_datas = []
for item in a.datas:
    dest_lower = item[0].lower()
    if 'gkdt_engine' in dest_lower and ('assets' in dest_lower or 'ims1' in dest_lower):
        continue
    filtered_datas.append(item)
a.datas = filtered_datas

pyz = PYZ(a.pure)

exe_icon = os.path.join(PROJECT_DIR, 'static', 'favicon.ico')
if not os.path.exists(exe_icon):
    exe_icon = None

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
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=exe_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Zero2YoloYard',
)
