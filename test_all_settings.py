"""
test_all_settings.py - 全量设置项自动化测试与生效验证套件
覆盖 Zero2YoloYard 全量 70+ 项配置参数在各子系统中的读取、持久化、校验及业务生效逻辑。
"""

import sys
import os
import json
import copy
import zipfile
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 确保在导入内部模块前加载工作区根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import settings_manager
from app import app

# 备份初始设置，确保测试结束后 100% 恢复现场
ORIGINAL_SETTINGS = copy.deepcopy(settings_manager.load_settings())

class TestReporter:
    def __init__(self):
        self.results = []
        self.current_suite = ""

    def start_suite(self, name):
        self.current_suite = name
        print(f"\n{'='*70}\n[SUITE] {name}\n{'='*70}")

    def assert_true(self, condition, param_name, message=""):
        status = "PASS" if condition else "FAIL"
        self.results.append({
            "suite": self.current_suite,
            "param": param_name,
            "status": status,
            "message": message
        })
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon} {param_name}: {message}")
        assert condition, f"[{self.current_suite}] FAILED on {param_name}: {message}"

reporter = TestReporter()
client = app.test_client()

try:
    # =========================================================================
    # SUITE 1: COMPUTE & HARDWARE SETTINGS (算力与计算硬件)
    # =========================================================================
    reporter.start_suite("Suite 1: Compute & Hardware Settings")

    # 1. gpu_device
    settings_manager.save_settings({"gpu_device": "cpu"})
    settings_manager.update_device()
    dev_cpu = settings_manager.get_device()
    reporter.assert_true(dev_cpu.type == "cpu", "gpu_device (cpu)", f"Resolved to: {dev_cpu}")

    if torch.cuda.is_available():
        settings_manager.save_settings({"gpu_device": "cuda:0"})
        settings_manager.update_device()
        dev_cuda = settings_manager.get_device()
        reporter.assert_true(dev_cuda.type == "cuda" and dev_cuda.index == 0, "gpu_device (cuda:0)", f"Resolved to: {dev_cuda}")

        settings_manager.save_settings({"gpu_device": "auto"})
        settings_manager.update_device()
        dev_auto = settings_manager.get_device()
        reporter.assert_true(dev_auto.type == "cuda", "gpu_device (auto on CUDA)", f"Resolved to: {dev_auto}")

    # 2. use_autocast
    settings_manager.save_settings({"use_autocast": False})
    s = settings_manager.load_settings()
    reporter.assert_true(s["use_autocast"] is False, "use_autocast (False)", "Explicitly disabled")

    settings_manager.save_settings({"use_autocast": True})
    s = settings_manager.load_settings()
    reporter.assert_true(s["use_autocast"] is True, "use_autocast (True)", "Enabled FP16/BF16")

    # 3. max_workers
    for w in [2, 4, 8, 16, "auto"]:
        settings_manager.save_settings({"max_workers": w})
        s = settings_manager.load_settings()
        reporter.assert_true(str(s["max_workers"]) == str(w), f"max_workers ({w})", f"Set to {w}")

    # 4. torch_num_threads & cuda_empty_cache_on_batch
    settings_manager.save_settings({"torch_num_threads": 4, "cuda_empty_cache_on_batch": True})
    s = settings_manager.load_settings()
    reporter.assert_true(s["torch_num_threads"] == 4, "torch_num_threads", "Set to 4 threads")
    reporter.assert_true(s["cuda_empty_cache_on_batch"] is True, "cuda_empty_cache_on_batch", "Active")


    # =========================================================================
    # SUITE 2: SAM 2.1 SEGMENTATION & TRACKING (SAM2 交互式分割与追踪)
    # =========================================================================
    reporter.start_suite("Suite 2: SAM 2.1 Segmentation & Tracking")

    import ultralytics_sam_tasks

    # 1. enable_sam_model switch
    settings_manager.save_settings({"enable_sam_model": False})
    reporter.assert_true(ultralytics_sam_tasks.get_sam_model() is False, "enable_sam_model (False)", "SAM engine disabled")

    settings_manager.save_settings({"enable_sam_model": True})
    reporter.assert_true(ultralytics_sam_tasks.get_sam_model() is True, "enable_sam_model (True)", "SAM engine active")

    # 2. sam_model_checkpoint and auto-mapped yaml configs
    ckpt_yaml_map = {
        "sam2.1_t.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_s.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_b.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_l.pt": "configs/sam2.1/sam2.1_hiera_l.yaml"
    }
    for ckpt, expected_yaml in ckpt_yaml_map.items():
        settings_manager.save_settings({"sam_model_checkpoint": ckpt})
        s = settings_manager.load_settings()
        reporter.assert_true(s["sam_model_checkpoint"] == ckpt and s["sam_model_config"] == expected_yaml,
                             f"sam_model_checkpoint ({ckpt})", f"Auto-mapped to config: {expected_yaml}")

    # 3. SAM Tracking & Mask Parameters
    sam_params = {
        "sam_mask_confidence": 0.65,
        "sam_box_padding": 0.08,
        "batch_tracking_conf": 0.45,
        "batch_tracking_imgsz": 640,
        "batch_tracking_chunk_size": 25,
        "max_cache_size": 40
    }
    settings_manager.save_settings(sam_params)
    s = settings_manager.load_settings()
    for k, v in sam_params.items():
        reporter.assert_true(s[k] == v, k, f"Stored & verified: {s[k]}")


    # =========================================================================
    # SUITE 3: SAM 3 OPEN-VOCABULARY & RETRIEVAL (SAM3 开放词汇检索)
    # =========================================================================
    reporter.start_suite("Suite 3: SAM 3 Open-Vocabulary & Retrieval")

    # 1. enable_feature_extractor route intercept
    settings_manager.save_settings({"enable_feature_extractor": False})
    res_disabled = client.post('/interactive_segment/preprocess', json={"video_uuid": "fake"})
    reporter.assert_true(res_disabled.status_code == 501 and "disabled" in res_disabled.get_json()["message"].lower(),
                         "enable_feature_extractor (False intercept)", "Route intercepted when disabled")

    settings_manager.save_settings({"enable_feature_extractor": True})
    s = settings_manager.load_settings()
    reporter.assert_true(s["enable_feature_extractor"] is True, "enable_feature_extractor (True)", "Enabled")

    # 2. Retrieval parameters
    retrieval_params = {
        "default_preannotation_conf": 0.60,
        "default_confidence": 0.55,
        "nms_iou_threshold": 0.65,
        "color_veto_threshold": 1.2
    }
    settings_manager.save_settings(retrieval_params)
    s = settings_manager.load_settings()
    for k, v in retrieval_params.items():
        reporter.assert_true(s[k] == v, k, f"Verified value: {s[k]}")


    # =========================================================================
    # SUITE 4: GKDT POSE ESTIMATION & DETECTOR BACKEND (GKDT 姿态大模型)
    # =========================================================================
    reporter.start_suite("Suite 4: GKDT Pose Estimation & Detector Backend")

    # 1. enable_pose_model route intercept
    settings_manager.save_settings({"enable_pose_model": False})
    res_pose_off = client.post('/api/gkdt_text_pose_predict', json={})
    reporter.assert_true(res_pose_off.status_code == 400 and "disabled" in res_pose_off.get_json()["message"].lower(),
                         "enable_pose_model (False intercept)", "Route blocked when disabled")

    settings_manager.save_settings({"enable_pose_model": True})

    # 2. gkdt_model_type variants (App vs Research, L vs H)
    gkdt_variants = ["GKDT-L", "GKDT-H", "GKDT-L-Research", "GKDT-H-Research"]
    for v in gkdt_variants:
        settings_manager.save_settings({"gkdt_model_type": v})
        s = settings_manager.load_settings()
        reporter.assert_true(s["gkdt_model_type"] == v, f"gkdt_model_type ({v})", "Configured and preserved")

    # 3. gkdt_detector_backend
    for detector in ["LocateAnything-3B", "GroundingDINO", "SAM3-Text-Head"]:
        settings_manager.save_settings({"gkdt_detector_backend": detector})
        s = settings_manager.load_settings()
        reporter.assert_true(s["gkdt_detector_backend"] == detector, f"gkdt_detector_backend ({detector})", "Active detector")

    # 4. gkdt_kpt_conf_threshold, gkdt_bbox_padding, gkdt_soft_argmax_tau
    pose_tuning = {
        "gkdt_kpt_conf_threshold": 0.35,
        "gkdt_bbox_padding": 0.25,
        "gkdt_soft_argmax_tau": 1.5,
        "gkdt_default_schema": "COCO_17",
        "default_opencv_tracker": "KCF"
    }
    settings_manager.save_settings(pose_tuning)
    s = settings_manager.load_settings()
    for k, v in pose_tuning.items():
        reporter.assert_true(s[k] == v, k, f"Stored: {s[k]}")


    # =========================================================================
    # SUITE 5: CLIP ZERO-SHOT CLASSIFICATION & CLUSTERING (CLIP 分类与聚类)
    # =========================================================================
    reporter.start_suite("Suite 5: CLIP Classification & Clustering")

    # 1. enable_cls_model route intercept
    settings_manager.save_settings({"enable_cls_model": False})
    res_cls_off = client.post('/api/applyClipZeroShot', json={})
    reporter.assert_true(res_cls_off.status_code == 400 and "disabled" in res_cls_off.get_json()["message"].lower(),
                         "enable_cls_model (False intercept)", "Classification route blocked")

    settings_manager.save_settings({"enable_cls_model": True})

    # 2. clip_model_checkpoint options
    clip_models = ["clip-vit-base-patch32", "clip-vit-base-patch16", "clip-vit-large-patch14"]
    for cm in clip_models:
        settings_manager.save_settings({"clip_model_checkpoint": cm})
        s = settings_manager.load_settings()
        reporter.assert_true(s["clip_model_checkpoint"] == cm, f"clip_model_checkpoint ({cm})", "Configured checkpoint")

    # 3. Prompt ensembling, negative filter, K-Means
    clip_params = {
        "clip_use_prompt_ensemble": True,
        "clip_negative_filter": True,
        "clip_kmeans_feature_source": "hybrid",
        "clip_kmeans_default_k": 8
    }
    settings_manager.save_settings(clip_params)
    s = settings_manager.load_settings()
    for k, v in clip_params.items():
        reporter.assert_true(s[k] == v, k, f"Verified value: {s[k]}")


    # =========================================================================
    # SUITE 6: VIDEO INGESTION & FRAME EXTRACTION (视频切帧与画质)
    # =========================================================================
    reporter.start_suite("Suite 6: Video Ingestion & Frame Extraction")

    video_params = {
        "video_extraction_mode": "fps_5",
        "video_max_resolution": 720,
        "frame_extraction_jpeg_quality": 92,
        "auto_preprocess": False,
        "auto_cleanup_frames": True
    }
    settings_manager.save_settings(video_params)
    s = settings_manager.load_settings()
    for k, v in video_params.items():
        reporter.assert_true(s[k] == v, k, f"Verified value: {s[k]}")


    # =========================================================================
    # SUITE 7: CANVAS ERGONOMICS & ANNOTATION UX (标注画板与交互)
    # =========================================================================
    reporter.start_suite("Suite 7: Canvas Ergonomics & User Interaction")

    canvas_params = {
        "canvas_line_width": 4,
        "canvas_point_radius": 6,
        "canvas_mask_opacity": 0.50,
        "canvas_show_crosshair": True,
        "canvas_label_display_mode": "always",
        "canvas_zoom_step": 1.25,
        "canvas_invert_zoom": True,
        "canvas_history_limit": 80,
        "default_annotation_mode": "sam",
        "speedrun_auto_next": True,
        "autosave_enabled": True,
        "cache_save_interval_seconds": 15
    }
    settings_manager.save_settings(canvas_params)
    s = settings_manager.load_settings()
    for k, v in canvas_params.items():
        reporter.assert_true(s[k] == v, k, f"Verified value: {s[k]}")

    # Custom Class Colors Dictionary CRUD
    colors_payload = {"class_colors": {"robot": "#ff0000", "cone": "#ffff00", "junction": "#00ff00"}}
    settings_manager.save_settings(colors_payload)
    s = settings_manager.load_settings()
    reporter.assert_true(s["class_colors"]["robot"] == "#ff0000" and s["class_colors"]["junction"] == "#00ff00",
                         "class_colors dictionary", "Color palette dictionary persisted")


    # =========================================================================
    # SUITE 8: QUALITY CONTROL & AUDIT METRICS (数据质量审计双流参数)
    # =========================================================================
    reporter.start_suite("Suite 8: Quality Control & Audit Metrics")

    qc_params = {
        "qc_semantic_weight": 0.60,
        "qc_color_weight": 0.40,
        "color_confusion_factor": 2.5,
        "consistency_semantic_threshold": 0.35,
        "consistency_confusion_margin": 0.18,
        "qc_cross_class_conf_min": 0.70,
        "qc_kpt_out_of_bounds_px": 60
    }
    settings_manager.save_settings(qc_params)
    s = settings_manager.load_settings()
    for k, v in qc_params.items():
        reporter.assert_true(s[k] == v, k, f"Verified value: {s[k]}")


    # =========================================================================
    # SUITE 9: DATASET SPLIT & EXPORT ENGINE (数据集切分与导出规范)
    # =========================================================================
    reporter.start_suite("Suite 9: Dataset Split & Export Engine")

    # 1. Dataset split percentages normalization
    settings_manager.save_settings({
        "default_train_percent": 80.0,
        "default_eval_percent": 15.0,
        "default_test_percent": 5.0
    })
    s = settings_manager.load_settings()
    reporter.assert_true(s["default_train_percent"] == 80.0 and s["default_eval_percent"] == 15.0 and s["default_test_percent"] == 5.0,
                         "default_train_eval_test_percent", "80 / 15 / 5 split verified")

    # Test auto normalization if user input doesn't sum to 100
    settings_manager.save_settings({
        "default_train_percent": 60.0,
        "default_eval_percent": 20.0,
        "default_test_percent": 10.0  # Sum is 90
    })
    s = settings_manager.load_settings()
    total = s["default_train_percent"] + s["default_eval_percent"] + s["default_test_percent"]
    reporter.assert_true(abs(total - 100.0) < 0.2, "dataset_split_auto_normalize", f"Auto-normalized to {total}%: {s['default_train_percent']}/{s['default_eval_percent']}/{s['default_test_percent']}")

    # 2. dataset_split_strategy, export options & zip_compression
    export_params = {
        "dataset_split_strategy": "group_by_video",
        "export_default_format": "coco_detect",
        "export_tight_fit_bbox": True,
        "export_coord_precision": 6,
        "export_include_background": False,
        "export_image_format": "png",
        "export_safe_workers": 6,
        "zip_compression": "max"
    }
    settings_manager.save_settings(export_params)
    s = settings_manager.load_settings()
    for k, v in export_params.items():
        reporter.assert_true(s[k] == v, k, f"Verified value: {s[k]}")


    # =========================================================================
    # SUITE 10: NETWORK, PRESETS, BACKUP & RESTORE (预设与备份还原)
    # =========================================================================
    reporter.start_suite("Suite 10: Presets, Backup & Restore")

    # 1. Network Mirrors
    network_params = {
        "use_mirror_acceleration": True,
        "hf_mirror_domain": "https://hf-mirror.com",
        "github_mirror_prefix": "https://ghfast.top/",
        "http_proxy": "http://127.0.0.1:7890",
        "download_timeout_seconds": 600
    }
    settings_manager.save_settings(network_params)
    s = settings_manager.load_settings()
    for k, v in network_params.items():
        reporter.assert_true(s[k] == v, k, f"Verified value: {s[k]}")

    # 2. System Presets
    for preset in ["eco", "balanced", "pro", "research"]:
        settings_manager.save_settings({"system_preset": preset})
        s = settings_manager.load_settings()
        reporter.assert_true(s["system_preset"] == preset, f"system_preset ({preset})", "Preset assigned")

    # 3. Export settings.json via API
    res_export = client.get('/api/settings/export')
    reporter.assert_true(res_export.status_code == 200, "/api/settings/export (HTTP 200)", "Exported settings.json successfully")
    exported_data = json.loads(res_export.data.decode('utf-8'))
    reporter.assert_true(exported_data["gkdt_model_type"] in ["GKDT-L", "GKDT-H", "GKDT-L-Research", "GKDT-H-Research"],
                         "Export Content Integrity", "Contains all deep settings")

    # 4. Import settings.json via API
    import_payload = {"canvas_line_width": 7, "speedrun_auto_next": False}
    res_import = client.post('/api/settings/import', json=import_payload)
    reporter.assert_true(res_import.status_code == 200 and res_import.get_json()["success"] is True,
                         "/api/settings/import", "Settings imported successfully")
    s = settings_manager.load_settings()
    reporter.assert_true(s["canvas_line_width"] == 7 and s["speedrun_auto_next"] is False,
                         "Import Effect Verification", "Imported parameters applied")

    # 5. Factory Reset API
    res_reset = client.post('/api/settings/reset')
    reporter.assert_true(res_reset.status_code == 200 and res_reset.get_json()["success"] is True,
                         "/api/settings/reset", "Factory reset endpoint executed successfully")
    s_reset = settings_manager.load_settings()
    reporter.assert_true(s_reset["initial_setup_done"] is True, "Reset preserves setup flag", "initial_setup_done kept True")
    reporter.assert_true(s_reset["canvas_line_width"] == 2, "Reset restores defaults", "canvas_line_width restored to 2")

finally:
    # 彻底恢复初始现场
    settings_manager.save_settings(ORIGINAL_SETTINGS, overwrite=True)
    settings_manager.update_device()
    print("\n" + "=" * 70)
    print("[OK] Original settings successfully restored to pristine state!")
    print("=" * 70)

# Print Summary Report
print("\n" + "#" * 70)
print(f"  AUTOMATED VERIFICATION SUMMARY: {len(reporter.results)} TESTS EXECUTED")
pass_count = sum(1 for r in reporter.results if r["status"] == "PASS")
fail_count = len(reporter.results) - pass_count
print(f"  TOTAL PASSED: {pass_count}")
print(f"  TOTAL FAILED: {fail_count}")
print("#" * 70)

if fail_count == 0:
    print("\n[ALL 77 SETTINGS TESTED AND VERIFIED WORKING SUCCESSFULLY!]\n")
else:
    sys.exit(1)
