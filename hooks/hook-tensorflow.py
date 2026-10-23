# -*- mode: python ; coding: utf-8 -*-
# 自定义轻量化 TensorFlow Hook
# 仅收集 FTC ML 所需的 TensorFlow Lite Interpreter，彻底避免全量 TensorFlow + PyTorch 引发的 OpenMP 0xC0000005 内存越界崩溃

from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

hiddenimports = [
    'tensorflow.lite.python.interpreter',
]

binaries = [
    (src, dst) for src, dst in collect_dynamic_libs('tensorflow')
    if 'lite' in src.lower()
]
datas = []
