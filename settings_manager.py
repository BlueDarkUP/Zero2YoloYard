import json
import os
import logging
import shutil
import config
import torch

SETTINGS_FILE = os.path.join(config.BASE_DIR, 'settings.json')

DEFAULT_SETTINGS = {
    # --- 1. 向导与环境初始化 ---
    "initial_setup_done": False,
    "system_preset": "balanced",  # "eco", "balanced", "pro", "research"

    # --- 2. 计算硬件与算力 ---
    "gpu_device": "auto",  # "auto", "cpu", "cuda:0", "cuda:1", ...
    "use_autocast": True,  # 混合精度开关 (FP16/BF16)
    "max_workers": 8,  # 后台多线程工作池最大线程数
    "torch_num_threads": 0,  # 0 为自动系统默认
    "cuda_empty_cache_on_batch": True,  # 批处理推断后自动清理 PyTorch 显存碎片

    # --- 3. AI 基础大模型矩阵总开关 ---
    "enable_sam_model": True,  # SAM 2.1 交互式分割与追踪
    "enable_feature_extractor": True,  # SAM 3 开放词汇检索 / Smart Select / TrueLAM
    "enable_cls_model": True,  # CLIP 零样本图像分类与聚类
    "enable_pose_model": True,  # GKDT 姿态与关键点估计

    # --- 4. SAM 2.1 交互分割与长时跟踪 ---
    "sam_model_checkpoint": "sam2.1_t.pt",  # "sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt"
    "sam_model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam_mask_confidence": 0.70,  # 分割掩膜置信度阈值
    "sam_box_padding": 0.0,  # 提示框扩展系数 (0.0 ~ 0.2)
    "batch_tracking_conf": 0.30,  # 视频连续跟踪目标保持置信度
    "batch_tracking_imgsz": 1024,  # 跟踪图像输入尺寸 (512, 640, 1024)
    "batch_tracking_chunk_size": 10,  # 跟踪分块帧数 (10 ~ 100)
    "max_cache_size": 30,  # SAM 帧状态显存 LRU 缓存上限 (10 ~ 100 帧)
    "auto_preprocess": True,  # 后台自动预热与帧扫描开关

    # --- 5. SAM 3 开放词汇检索与目标发现 ---
    "default_preannotation_conf": 0.50,  # SAM 3 开放词汇检索置信度
    "default_confidence": 0.50,  # 全局通用默认置信度
    "nms_iou_threshold": 0.70,  # 重叠候选框 NMS IoU 抑制阈值
    "color_veto_threshold": 1.0,  # 色彩直方图否决差异阈值

    # --- 6. GKDT 姿态估计与关键点检测 (ECCV 2026) ---
    "gkdt_model_type": "GKDT-L",  # "GKDT-L", "GKDT-H", "GKDT-L-Research", "GKDT-H-Research"
    "gkdt_detector_backend": "LocateAnything-3B",  # "LocateAnything-3B", "GroundingDINO", "SAM3-Text-Head"
    "gkdt_kpt_conf_threshold": 0.25,  # 关键点截断置信度 (0.1 ~ 0.9)
    "gkdt_bbox_padding": 0.20,  # 姿态候选框外扩安全边距 (0.0 ~ 0.35)
    "gkdt_soft_argmax_tau": 1.0,  # SoftArgmax 温度系数
    "gkdt_default_schema": "COCO_17",  # 默认关键点骨架拓扑

    # --- 7. CLIP 语义分类与图像检索 ---
    "clip_model_checkpoint": "clip-vit-base-patch32",  # "clip-vit-base-patch32", "clip-vit-base-patch16", "clip-vit-large-patch14"
    "clip_use_prompt_ensemble": True,  # 开启 8 模板 Prompt Ensembling 均值融合
    "clip_negative_filter": True,  # 开启单一类别时的背景负类保护
    "clip_kmeans_feature_source": "semantic",  # "semantic", "color", "hybrid"
    "clip_kmeans_default_k": 5,  # K-Means 聚类默认簇数 (2 ~ 30)

    # --- 8. 传统 OpenCV 跟踪器 ---
    "default_opencv_tracker": "CSRT",  # "CSRT", "KCF", "MOSSE", "MedianFlow", "MIL", "TLD", "Boosting"

    # --- 9. 视频摄入与智能抽帧 ---
    "video_extraction_mode": "all",  # "all", "fps_1", "fps_2", "fps_5", "fps_10", "step_2", "step_5", "step_10"
    "video_max_resolution": 1080,  # 抽帧图片最大边限制 (0 为不缩放, 720, 1080, 1440, 2160)
    "frame_extraction_jpeg_quality": 85,  # 抽帧保存 JPEG 压缩画质 (10 ~ 100)
    "auto_cleanup_frames": False,  # 导出后是否自动清理未标注孤儿帧

    # --- 10. 标注画板交互与人机工程 (Canvas Ergonomics) ---
    "default_annotation_mode": "manual",  # "manual", "sam", "lam", "smart_select"
    "speedrun_auto_next": True,  # 极速标注模式 (打标后自动滚入下一帧)
    "autosave_enabled": False,  # 翻帧时自动保存
    "cache_save_interval_seconds": 30,  # 内存未保存数据自动存盘周期
    "canvas_line_width": 2,  # 标注外框线宽 (1 ~ 8 px)
    "canvas_point_radius": 4,  # 控制点与关键点抓手半径 (2 ~ 12 px)
    "canvas_mask_opacity": 0.35,  # 分割掩膜填充不透明度 (0.1 ~ 0.9)
    "canvas_show_crosshair": False,  # 是否开启全屏辅助十字参考线
    "canvas_label_display_mode": "hover",  # 类别标签显示模式: "always", "hover", "never"
    "canvas_zoom_step": 1.15,  # 滚轮缩放步长灵敏度 (1.05 ~ 1.30)
    "canvas_invert_zoom": False,  # 是否反转滚轮缩放方向
    "canvas_history_limit": 50,  # 撤销/重做最大历史栈深度 (20 ~ 100)
    "class_colors": {},  # 自定义类别色彩映射字典 {"class_name": "#hex"}

    # --- 11. 数据质量审计与异常过滤 (QC) ---
    "color_confusion_factor": 2.0,  # 色彩偏离报警敏感度 (Sigma 倍数 1.0 ~ 4.0)
    "consistency_semantic_threshold": 0.30,  # 语义一致性绝对下限阈值
    "consistency_confusion_margin": 0.15,  # 类别交叉混淆容忍度
    "qc_semantic_weight": 0.70,  # 双流特征审计中语义特征权重
    "qc_color_weight": 0.30,  # 双流特征审计中颜色特征权重
    "qc_cross_class_conf_min": 0.65,  # 跨类混淆预警置信度底线
    "qc_kpt_out_of_bounds_px": 50,  # 关节点越界容差阈值 (像素)

    # --- 12. 数据集切分、编译与导出 ---
    "default_train_percent": 70.0,  # 默认训练集比例
    "default_eval_percent": 20.0,  # 默认验证集比例
    "default_test_percent": 10.0,  # 默认测试集比例
    "dataset_split_strategy": "group_by_video",  # "group_by_video" (防相邻帧泄露), "random"
    "export_default_format": "yolo_detect",  # 默认导出目标格式
    "export_tight_fit_bbox": True,  # 分割转框时是否紧贴掩膜边缘
    "export_coord_precision": 6,  # 归一化坐标保留小数位 (4 ~ 8)
    "export_include_background": True,  # 导出是否包含未标注纯背景帧
    "export_image_format": "jpg",  # "jpg", "png"
    "export_safe_workers": 4,  # 多进程导出并发进程数
    "zip_compression": "standard",  # "fast" (STORE_ONLY), "standard" (DEFLATED), "max" (HIGH_CPU)

    # --- 13. 网络镜像与模型资产仓库 ---
    "use_mirror_acceleration": True,  # 国内镜像加速总开关
    "hf_mirror_domain": "https://hf-mirror.com",  # HuggingFace 镜像源
    "github_mirror_prefix": "https://ghfast.top/",  # GitHub 代理源前缀
    "http_proxy": "",  # 自定义 HTTP 代理
    "https_proxy": "",  # 自定义 HTTPS 代理
    "download_timeout_seconds": 300  # 模型资产下载超时时间
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
    elif "cuda" in str(device_setting).lower() and torch.cuda.is_available():
        try:
            parts = device_setting.split(':')
            device_id = int(parts[1]) if len(parts) > 1 else 0
            if device_id < torch.cuda.device_count():
                _device = torch.device(f"cuda:{device_id}")
                logging.info(f"Using specified CUDA device: cuda:{device_id}")
            else:
                _device = torch.device("cuda:0")
                logging.warning(f"Device {device_setting} not found (count={torch.cuda.device_count()}), falling back to cuda:0.")
        except (IndexError, ValueError):
            _device = torch.device("cuda:0")
            logging.warning(f"Invalid CUDA device format '{device_setting}', falling back to cuda:0.")
    else:
        if "cuda" in str(device_setting).lower():
            logging.warning("CUDA device specified but not available. Falling back to CPU.")
        _device = torch.device("cpu")
        logging.info("Using CPU.")

    return _device


def update_device():
    global _device
    _device = None
    logging.info("Device setting updated. Will re-evaluate on next use.")


def sanitize_settings(settings):
    """
    对设置字典进行类型转换、范围约束与逻辑合法性校验。
    """
    if not isinstance(settings, dict):
        return dict(DEFAULT_SETTINGS)

    # 1. 递归补全缺失键
    for k, v in DEFAULT_SETTINGS.items():
        if k not in settings:
            settings[k] = v
        elif isinstance(v, dict) and isinstance(settings[k], dict):
            for sub_k, sub_v in v.items():
                if sub_k not in settings[k]:
                    settings[k][sub_k] = sub_v

    # 2. SAM checkpoint 与 config 动态映射
    ckpt = settings.get("sam_model_checkpoint", "sam2.1_t.pt")
    if ckpt == "sam2.1_t.pt":
        settings["sam_model_config"] = "configs/sam2.1/sam2.1_hiera_t.yaml"
    elif ckpt == "sam2.1_s.pt":
        settings["sam_model_config"] = "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif ckpt == "sam2.1_b.pt":
        settings["sam_model_config"] = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif ckpt == "sam2.1_l.pt":
        settings["sam_model_config"] = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # 3. 数值与范围约束安全校验
    try:
        settings["sam_mask_confidence"] = max(0.05, min(0.99, float(settings.get("sam_mask_confidence", 0.70))))
        settings["nms_iou_threshold"] = max(0.05, min(1.00, float(settings.get("nms_iou_threshold", 0.70))))
        settings["default_preannotation_conf"] = max(0.01, min(0.99, float(settings.get("default_preannotation_conf", 0.50))))
        settings["default_confidence"] = max(0.01, min(0.99, float(settings.get("default_confidence", 0.50))))
        settings["batch_tracking_conf"] = max(0.01, min(0.99, float(settings.get("batch_tracking_conf", 0.30))))

        settings["max_cache_size"] = max(5, min(200, int(settings.get("max_cache_size", 30))))
        settings["frame_extraction_jpeg_quality"] = max(10, min(100, int(settings.get("frame_extraction_jpeg_quality", 85))))
        settings["cache_save_interval_seconds"] = max(5, min(600, int(settings.get("cache_save_interval_seconds", 30))))

        # 画板参数约束
        settings["canvas_line_width"] = max(1, min(12, int(settings.get("canvas_line_width", 2))))
        settings["canvas_point_radius"] = max(1, min(20, int(settings.get("canvas_point_radius", 4))))
        settings["canvas_mask_opacity"] = max(0.05, min(0.95, float(settings.get("canvas_mask_opacity", 0.35))))
        settings["canvas_zoom_step"] = max(1.02, min(1.50, float(settings.get("canvas_zoom_step", 1.15))))
        settings["canvas_history_limit"] = max(10, min(200, int(settings.get("canvas_history_limit", 50))))

        # 审计权重约束
        settings["color_confusion_factor"] = max(0.5, min(5.0, float(settings.get("color_confusion_factor", 2.0))))
        settings["consistency_semantic_threshold"] = max(0.05, min(0.95, float(settings.get("consistency_semantic_threshold", 0.30))))
        settings["consistency_confusion_margin"] = max(0.01, min(0.50, float(settings.get("consistency_confusion_margin", 0.15))))
        settings["qc_semantic_weight"] = max(0.0, min(1.0, float(settings.get("qc_semantic_weight", 0.70))))
        settings["qc_color_weight"] = max(0.0, min(1.0, float(settings.get("qc_color_weight", 0.30))))
        settings["qc_cross_class_conf_min"] = max(0.1, min(0.95, float(settings.get("qc_cross_class_conf_min", 0.65))))
        settings["qc_kpt_out_of_bounds_px"] = max(0, min(500, int(settings.get("qc_kpt_out_of_bounds_px", 50))))

        # GKDT 约束
        settings["gkdt_kpt_conf_threshold"] = max(0.05, min(0.95, float(settings.get("gkdt_kpt_conf_threshold", 0.25))))
        settings["gkdt_bbox_padding"] = max(0.0, min(0.50, float(settings.get("gkdt_bbox_padding", 0.20))))
        settings["gkdt_soft_argmax_tau"] = max(0.1, min(5.0, float(settings.get("gkdt_soft_argmax_tau", 1.0))))

        # 数据切分比例安全归一化
        train_p = float(settings.get("default_train_percent", 70.0))
        eval_p = float(settings.get("default_eval_percent", 20.0))
        test_p = float(settings.get("default_test_percent", 10.0))
        total_p = train_p + eval_p + test_p
        if abs(total_p - 100.0) > 0.5 and total_p > 0:
            settings["default_train_percent"] = round((train_p / total_p) * 100.0, 1)
            settings["default_eval_percent"] = round((eval_p / total_p) * 100.0, 1)
            settings["default_test_percent"] = round(100.0 - settings["default_train_percent"] - settings["default_eval_percent"], 1)

    except Exception as e:
        logging.warning(f"Error sanitizing settings values: {e}")

    return settings


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        logging.info(f"Settings file not found. Creating a new one at {SETTINGS_FILE}")
        default_copy = dict(DEFAULT_SETTINGS)
        save_settings(default_copy)
        return default_copy
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        return sanitize_settings(settings)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load settings file: {e}. Returning default settings.")
        return dict(DEFAULT_SETTINGS)


def save_settings(settings_data, overwrite=False):
    try:
        if not overwrite and os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f_in:
                    current = json.load(f_in)
                if isinstance(current, dict):
                    merged = dict(current)
                    merged.update(settings_data)
                    settings_data = merged
            except Exception:
                pass
        sanitized = sanitize_settings(settings_data)
        tmp_file = SETTINGS_FILE + ".tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(sanitized, f, indent=4, ensure_ascii=False)
        os.replace(tmp_file, SETTINGS_FILE)
        return True
    except IOError as e:
        logging.error(f"Failed to save settings file: {e}")
        return False


def reset_to_defaults():
    """重置系统配置为出厂默认值（保留已初始化的标记）"""
    current = load_settings()
    is_setup_done = current.get("initial_setup_done", False)
    new_settings = dict(DEFAULT_SETTINGS)
    new_settings["initial_setup_done"] = is_setup_done
    if save_settings(new_settings, overwrite=True):
        update_device()
        return True, new_settings
    return False, current
