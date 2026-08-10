# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_data_files

# 1. 突破递归深度限制
sys.setrecursionlimit(5000)

# ==============================================================
# 绝对禁止使用 collect_submodules('ultralytics'/'sam2_repo')，防止内存爆炸
# ==============================================================

# Pywebview 底层依赖
hidden_webview = [
    'webview',
    'webview.platforms.winforms',
    'webview.platforms.edgechromium',
    'webview.platforms.cef',
    'clr',
]

# 手动定义隐式依赖 (精准狙击，绝不牵连无辜文件)
my_hidden_imports = [
    'torch',
    'torchvision',
    'tensorflow',
    'tensorflow.lite.python.interpreter',
    'cv2',
    'numpy',
    'PIL',

    # 核心 AI 库（只声明主模块，让它自己顺藤摸瓜）
    'ultralytics',
    'sam2',
    'sam2.build_sam',
    'sam2.automatic_mask_generator',
    'sam2.sam2_image_predictor',
    'albumentations',
    'hydra',
    'omegaconf',

    # 精准指定 sklearn
    'sklearn.cluster',
    'sklearn.metrics',
    'sklearn.utils._typedefs',
    'sklearn.neighbors._partition_nodes',

    # Utils / Web
    'flask',
    'waitress',
    'engineio.async_drivers.threading',
    'colorama',
    'yaml',
    'sqlite3',
    'sqlalchemy.dialects.sqlite',

    # Scipy 底层 (防止 C 扩展丢失)
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
]

final_hidden_imports = my_hidden_imports + hidden_webview

# 只收集核心数据文件
extra_datas = collect_data_files('webview')

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('configs', 'configs'),
        ('sam2', 'sam2'),
        ('checkpoints', 'checkpoints'),
    ] + extra_datas,
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
        'pytest', # 排除测试模块
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
    icon='C:\\Users\\BlueDarkUP\\OneDrive\\Desktop\\icon.ico',
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