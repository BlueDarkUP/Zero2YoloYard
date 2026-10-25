"""
Zero2YoloYard (Z2YY) Agent Hub & Tool Execution Framework
==========================================================
Unified interface for autonomous AI agents to operate all computer vision
dataset, annotation, model, and video workflows programmatically.
Excludes dataset inspection / QC review functions per system specification.
"""

import os
import sys
import json
import time
import uuid
import shutil
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Callable

import config
import database
import file_storage
import settings_manager
import local_model_manager
import background_tasks
import ai_models
from annotation_model import AnnotationData, AnnotationObject
from exporters import ExporterRegistry


# ==============================================================================
# TOOL REGISTRY & DECORATOR
# ==============================================================================

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_tool(name: str, category: str, description: str, parameters: Dict[str, Any], returns: Dict[str, Any]):
    def decorator(fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        TOOL_REGISTRY[name] = {
            "name": name,
            "category": category,
            "description": description.strip(),
            "parameters": parameters,
            "returns": returns,
            "handler": fn
        }
        return fn
    return decorator


# ==============================================================================
# CATEGORY 1: SYSTEM & ENVIRONMENT
# ==============================================================================

@register_tool(
    name="z2yy_get_system_status",
    category="system",
    description="Retrieve system health, hardware acceleration (CUDA/CPU device, VRAM, thread count), server boot status, and environment setup completion state.",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    returns={
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "cuda_available": {"type": "boolean"},
            "device": {"type": "string"},
            "setup_done": {"type": "boolean"},
            "threads": {"type": "integer"}
        }
    }
)
def handle_get_system_status(args: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    settings = settings_manager.load_settings()
    cuda_avail = torch.cuda.is_available()
    device_name = "cpu"
    vram_gb = 0.0
    if cuda_avail:
        try:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        except Exception:
            pass

    return {
        "status": "ONLINE",
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "vram_gb": vram_gb,
        "active_gpu_setting": settings.get("gpu_device", "auto"),
        "setup_done": bool(settings.get("initial_setup_done", False)),
        "max_workers": settings.get("max_workers", "auto"),
        "timestamp": int(time.time())
    }


@register_tool(
    name="z2yy_get_settings",
    category="system",
    description="Retrieve the complete dictionary of all 102 system configuration parameters (model thresholds, paths, mirrors, canvas defaults).",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    returns={"type": "object"}
)
def handle_get_settings(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"settings": settings_manager.load_settings()}


@register_tool(
    name="z2yy_update_settings",
    category="system",
    description="Update one or more system configuration settings (e.g. gpu_device, default_preannotation_conf, download mirrors, etc.).",
    parameters={
        "type": "object",
        "properties": {
            "settings": {
                "type": "object",
                "description": "Dictionary of configuration key-value pairs to update."
            }
        },
        "required": ["settings"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "updated_keys": {"type": "array"}}}
)
def handle_update_settings(args: Dict[str, Any]) -> Dict[str, Any]:
    updates = args.get("settings", {})
    current = settings_manager.load_settings()
    current.update(updates)
    settings_manager.save_settings(current)
    return {"success": True, "updated_keys": list(updates.keys())}

@register_tool(
    name="z2yy_reset_settings",
    category="system",
    description="Reset all system and hardware configuration settings to factory default values.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "settings": {"type": "object"}}}
)
def handle_reset_settings(args: Dict[str, Any]) -> Dict[str, Any]:
    success, current = settings_manager.reset_to_defaults()
    return {"success": success, "settings": current}


@register_tool(
    name="z2yy_export_settings",
    category="system",
    description="Export current system settings as a portable JSON configuration dictionary.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"settings": {"type": "object"}}}
)
def handle_export_settings(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"settings": settings_manager.load_settings()}


@register_tool(
    name="z2yy_import_settings",
    category="system",
    description="Import and overwrite system configuration from a provided settings dictionary.",
    parameters={
        "type": "object",
        "properties": {
            "settings": {"type": "object", "description": "Configuration dictionary to apply."}
        },
        "required": ["settings"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "settings": {"type": "object"}}}
)
def handle_import_settings(args: Dict[str, Any]) -> Dict[str, Any]:
    new_settings = args.get("settings", {})
    success = settings_manager.save_settings(new_settings)
    return {"success": success, "settings": settings_manager.load_settings()}


@register_tool(
    name="z2yy_clear_cache",
    category="system",
    description="Clear cached frame features, SAM inference cache, and transient memory allocations.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "cleared_items": {"type": "integer"}}}
)
def handle_clear_cache(args: Dict[str, Any]) -> Dict[str, Any]:
    count = 0
    try:
        import ai_models
        count = ai_models.sam_tasks.frame_cache_size() if (hasattr(ai_models, "sam_tasks") and ai_models.sam_tasks) else 0
        ai_models.clear_retrieval_engine_cache()
    except Exception:
        pass
    return {"success": True, "cleared_items": count}


@register_tool(
    name="z2yy_detect_hardware",
    category="system",
    description="Query detailed system hardware telemetry: CPU cores, RAM, CUDA availability, GPU names, VRAM, and recommended compute preset.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object"}
)
def handle_detect_hardware(args: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    cpu_cores = os.cpu_count() or 4
    ram_gb = 0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    has_cuda = torch.cuda.is_available()
    gpu_devices = []
    if has_cuda:
        try:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                dev_name = torch.cuda.get_device_name(i)
                dev_vram = round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 1)
                gpu_devices.append({
                    "id": i,
                    "name": dev_name,
                    "vram_gb": dev_vram
                })
        except Exception:
            pass

    max_vram = max([d["vram_gb"] for d in gpu_devices], default=0)
    if has_cuda and max_vram >= 12:
        recommended_preset = "desktop"
    elif has_cuda and max_vram >= 6:
        recommended_preset = "laptop"
    elif has_cuda and max_vram >= 3.5:
        recommended_preset = "portable"
    else:
        recommended_preset = "cpu"

    return {
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "has_cuda": has_cuda,
        "gpu_devices": gpu_devices,
        "recommended_preset": recommended_preset
    }



# ==============================================================================
# CATEGORY 2: FOUNDATION & CUSTOM AI MODELS
# ==============================================================================

@register_tool(
    name="z2yy_list_foundation_models",
    category="models",
    description="List all built-in foundation models (SAM 2.1 checkpoints, SAM 3, CLIP, GKDT Pose, LocateAnything, GroundingDINO) and their local installation and weight status.",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    returns={"type": "object", "properties": {"models": {"type": "array"}}}
)
def handle_list_foundation_models(args: Dict[str, Any]) -> Dict[str, Any]:
    model_defs = local_model_manager.get_model_registry()
    results = []
    for m in model_defs:
        present, size_mb = local_model_manager.check_model_presence(m, base_dir=config.BASE_DIR)
        dl_status = local_model_manager.get_download_status(m["id"])
        results.append({
            "id": m["id"],
            "name": m["name"],
            "engine": m["engine"],
            "path": m["path"].replace("\\", "/"),
            "ext": m["ext"],
            "purpose": m["purpose"],
            "present": present,
            "size_mb": size_mb,
            "rare_format": m.get("rare_format", False),
            "download_status": dl_status,
        })
    return {"models": results}


@register_tool(
    name="z2yy_download_foundation_model",
    category="models",
    description="Trigger background download of a foundation model checkpoint by ID ('sam2.1_t', 'sam2.1_s', 'sam2.1_b', 'sam2.1_l', 'clip_vit_b32', etc.).",
    parameters={
        "type": "object",
        "properties": {
            "model_id": {
                "type": "string",
                "description": "Unique identifier of the foundation model to download."
            }
        },
        "required": ["model_id"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "message": {"type": "string"}}}
)
def handle_download_foundation_model(args: Dict[str, Any]) -> Dict[str, Any]:
    model_id = args.get("model_id")
    success, msg = local_model_manager.start_download(model_id)
    return {"success": success, "message": msg, "model_id": model_id}


@register_tool(
    name="z2yy_get_model_download_status",
    category="models",
    description="Check the download progress and status of a foundation model.",
    parameters={
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "Model ID to inspect."}
        },
        "required": ["model_id"]
    },
    returns={"type": "object", "properties": {"model_id": {"type": "string"}, "status": {"type": "string"}}}
)
def handle_get_model_download_status(args: Dict[str, Any]) -> Dict[str, Any]:
    model_id = args.get("model_id")
    status = local_model_manager.get_download_status(model_id)
    return {"model_id": model_id, "download_status": status}


@register_tool(
    name="z2yy_list_trained_models",
    category="models",
    description="List all user-trained / custom models (YOLO detection, segmentation, pose) registered in the catalog.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"models": {"type": "array"}}}
)
def handle_list_trained_models(args: Dict[str, Any]) -> Dict[str, Any]:
    models = database.get_all_models()
    return {"models": models}


@register_tool(
    name="z2yy_import_trained_model",
    category="models",
    description="Import and register a custom trained model weight file (.pt or .onnx) with description, model_type, and optional labels.",
    parameters={
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Display name of the model."},
            "model_path": {"type": "string", "description": "Absolute path to model weight file (.pt, .onnx)."},
            "model_type": {
                "type": "string",
                "enum": ["yolo_detect", "yolo_seg", "yolo_pose"],
                "default": "yolo_detect",
                "description": "Model architecture type."
            },
            "label_filename": {"type": "string", "description": "Optional labels file path or comma-separated names."}
        },
        "required": ["description", "model_path"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "model_uuid": {"type": "string"}}}
)
def handle_import_trained_model(args: Dict[str, Any]) -> Dict[str, Any]:
    desc = args.get("description", "").strip()
    src_path = args.get("model_path", "").strip()
    model_type = args.get("model_type", "yolo_detect")
    label_filename = args.get("label_filename", "")

    if not desc or not src_path:
        raise ValueError("description and model_path are required.")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Model file not found at path: {src_path}")

    filename = os.path.basename(src_path)
    file_size = os.path.getsize(src_path)
    create_time = int(time.time() * 1000)

    model_uuid = database.create_model_entry(desc, filename, file_size, create_time, label_filename, model_type)
    dst_path = file_storage.get_model_path(model_uuid)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)

    return {"success": True, "model_uuid": model_uuid, "model_type": model_type}


@register_tool(
    name="z2yy_delete_trained_model",
    category="models",
    description="Remove a custom trained model from catalog.",
    parameters={
        "type": "object",
        "properties": {
            "model_uuid": {"type": "string", "description": "UUID of model to delete."}
        },
        "required": ["model_uuid"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_delete_trained_model(args: Dict[str, Any]) -> Dict[str, Any]:
    uuid_val = args.get("model_uuid")
    database.delete_model(uuid_val)
    file_storage.delete_model_files(uuid_val)
    return {"success": True, "model_uuid": uuid_val}

@register_tool(
    name="z2yy_delete_foundation_model",
    category="models",
    description="Delete downloaded weights of a foundation model to reclaim disk space.",
    parameters={
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "Identifier of the model to purge."}
        },
        "required": ["model_id"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "model_id": {"type": "string"}}}
)
def handle_delete_foundation_model(args: Dict[str, Any]) -> Dict[str, Any]:
    m_id = args.get("model_id")
    registry = local_model_manager.get_model_registry()
    model_def = next((m for m in registry if m["id"] == m_id), None)
    if not model_def:
        raise ValueError(f"Foundation model not found: {m_id}")
    
    base_dir = config.BASE_DIR
    target = os.path.join(base_dir, model_def.get("dest", ""))
    deleted = False
    if os.path.isfile(target):
        os.remove(target)
        deleted = True
    elif os.path.isdir(target):
        shutil.rmtree(target)
        deleted = True
    
    local_model_manager.DOWNLOAD_TASKS.pop(m_id, None)
    return {"success": True, "model_id": m_id, "deleted": deleted}



# ==============================================================================
# CATEGORY 3: VIDEO STREAM & ASSET INGESTION
# ==============================================================================

@register_tool(
    name="z2yy_upload_video",
    category="videos",
    description="Ingest a video file from local disk, register it in database, and queue asynchronous frame extraction at target FPS. Starting point for video annotation.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to video file (.mp4, .avi, .mov, etc.) on local disk."},
            "description": {"type": "string", "description": "Unique title / stream identifier for this video."},
            "target_fps": {"type": "integer", "default": 5, "description": "Frame sampling rate (frames extracted per second). Default 5."},
            "annotation_type": {
                "type": "string",
                "enum": ["detection", "segmentation", "pose", "classification"],
                "default": "detection",
                "description": "Primary annotation modality for this video stream."
            }
        },
        "required": ["file_path", "description"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "video_uuid": {"type": "string"}}}
)
def handle_upload_video(args: Dict[str, Any]) -> Dict[str, Any]:
    file_path = args.get("file_path", "").strip()
    desc = args.get("description", "").strip()
    target_fps = int(args.get("target_fps", 5))
    annotation_type = args.get("annotation_type", "detection")

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file does not exist: {file_path}")
    if not desc:
        raise ValueError("description is required.")

    existing_descs = [v['description'] for v in database.get_all_video_list()]
    if desc in existing_descs:
        desc = f"{desc}_{int(time.time())}"

    file_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)
    create_time = int(time.time() * 1000)

    video_uuid = database.create_video_entry(desc, filename, file_size, create_time, annotation_type=annotation_type)
    dst_video_path = file_storage.get_video_path(video_uuid)
    os.makedirs(os.path.dirname(dst_video_path), exist_ok=True)
    shutil.copy2(file_path, dst_video_path)

    import threading
    t = threading.Thread(
        target=background_tasks.extract_frames_task,
        args=(video_uuid, target_fps),
        name=f"AgentExtractor-{video_uuid[:6]}"
    )
    t.start()

    return {
        "success": True,
        "video_uuid": video_uuid,
        "description": desc,
        "target_fps": target_fps,
        "status": "EXTRACTING",
        "message": "Video registered and frame extraction queued in background."
    }


@register_tool(
    name="z2yy_list_videos",
    category="videos",
    description="List all registered videos, their UUIDs, descriptions, total frame count, labeled frame count, status, and annotation type.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"videos": {"type": "array"}}}
)
def handle_list_videos(args: Dict[str, Any]) -> Dict[str, Any]:
    videos = database.get_all_video_list()
    return {"videos": videos}


@register_tool(
    name="z2yy_get_video_info",
    category="videos",
    description="Get detailed metadata, total frame count, labeled frame count, resolution, and status for a specific video.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string", "description": "UUID of video to inspect."}
        },
        "required": ["video_uuid"]
    },
    returns={"type": "object"}
)
def handle_get_video_info(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    entity = database.get_video_entity(v_uuid)
    if not entity:
        raise ValueError(f"Video not found: {v_uuid}")
    frames = database.get_annotated_video_frames(v_uuid)
    return {
        "video": entity,
        "annotated_frames_count": len(frames),
        "frame_dir": file_storage.get_frame_dir(v_uuid)
    }


@register_tool(
    name="z2yy_delete_video",
    category="videos",
    description="Delete a video, removing its extracted frame images, annotations, and database records.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string", "description": "UUID of video to delete."}
        },
        "required": ["video_uuid"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_delete_video(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    database.delete_video(v_uuid)
    file_storage.delete_video_files(v_uuid)
    return {"success": True, "video_uuid": v_uuid}


@register_tool(
    name="z2yy_import_frames",
    category="videos",
    description="Import a folder of pre-extracted frame images directly into a new video entry without requiring a source video file.",
    parameters={
        "type": "object",
        "properties": {
            "folder_path": {"type": "string", "description": "Directory containing image files (.jpg, .png, etc.)."},
            "description": {"type": "string", "description": "Stream title."},
            "annotation_type": {"type": "string", "default": "detection"}
        },
        "required": ["folder_path", "description"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "video_uuid": {"type": "string"}}}
)
def handle_import_frames(args: Dict[str, Any]) -> Dict[str, Any]:
    folder = args.get("folder_path")
    desc = args.get("description")
    ann_type = args.get("annotation_type", "detection")
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Directory not found: {folder}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in valid_exts])
    if not img_files:
        raise ValueError(f"No image files found in {folder}")

    create_time = int(time.time() * 1000)
    v_uuid = database.create_video_entry(desc, "imported_frames", 0, create_time, annotation_type=ann_type)
    target_dir = file_storage.get_frame_dir(v_uuid)
    os.makedirs(target_dir, exist_ok=True)

    for idx, f in enumerate(img_files):
        src = os.path.join(folder, f)
        dst = os.path.join(target_dir, f"frame_{idx:05d}.jpg")
        shutil.copy2(src, dst)

    database.update_video_frame_count(v_uuid, len(img_files))
    database.update_video_status(v_uuid, 'READY')

    return {"success": True, "video_uuid": v_uuid, "imported_count": len(img_files)}

@register_tool(
    name="z2yy_get_video_frames",
    category="videos",
    description="Retrieve the sequential frame index for a video, indicating for each frame whether it has annotations, ambiguous flag, and frame number.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string", "description": "UUID of the video."}
        },
        "required": ["video_uuid"]
    },
    returns={"type": "object", "properties": {"frames": {"type": "array"}, "total_frames": {"type": "integer"}}}
)
def handle_get_video_frames(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    raw_frames = database.get_video_frames(v_uuid)
    frames_meta = []
    for f in raw_frames:
        ann_json = (f.get("annotations_json") or "").strip()
        has_ann = False
        is_ambiguous = bool(f.get("is_ambiguous", False))
        if ann_json:
            try:
                ad = json.loads(ann_json)
                has_ann = bool(ad.get("objects") or ad.get("classifications"))
                is_ambiguous = bool(ad.get("is_ambiguous", is_ambiguous))
            except Exception:
                pass
        if not has_ann and (f.get("bboxes_text") or "").strip():
            has_ann = True
        frames_meta.append({
            "frame_number": f.get("frame_number"),
            "has_annotations": has_ann,
            "is_ambiguous": is_ambiguous,
            "image_url": f"/media/frames/{v_uuid}/frame_{f.get('frame_number', 0):05d}.jpg"
        })
    return {"video_uuid": v_uuid, "total_frames": len(frames_meta), "frames": frames_meta}


@register_tool(
    name="z2yy_clear_video_annotations",
    category="videos",
    description="Reset and purge all annotation data across all frames of a video.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string", "description": "UUID of video whose annotations should be wiped."}
        },
        "required": ["video_uuid"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "video_uuid": {"type": "string"}}}
)
def handle_clear_video_annotations(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    with database.engine.begin() as conn:
        conn.execute(
            database.text("UPDATE video_frames SET bboxes_text = NULL, annotations_json = NULL WHERE video_uuid = :u"),
            {"u": v_uuid}
        )
        conn.execute(
            database.text("DELETE FROM frame_labels WHERE frame_id IN (SELECT frame_id FROM video_frames WHERE video_uuid = :u)"),
            {"u": v_uuid}
        )
        conn.execute(
            database.text("UPDATE videos SET labeled_frame_count = 0 WHERE video_uuid = :u"),
            {"u": v_uuid}
        )
    return {"success": True, "video_uuid": v_uuid, "message": "All annotations for video cleared."}



# ==============================================================================
# CATEGORY 4: CLASSES & PROMPTS
# ==============================================================================

@register_tool(
    name="z2yy_list_classes",
    category="classes",
    description="List all active annotation class labels in the workspace.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"labels": {"type": "array"}}}
)
def handle_list_classes(args: Dict[str, Any]) -> Dict[str, Any]:
    labels = database.get_all_class_labels()
    return {"labels": labels}


@register_tool(
    name="z2yy_add_class",
    category="classes",
    description="Create a new annotation class label (e.g. 'car', 'person', 'defect').",
    parameters={
        "type": "object",
        "properties": {
            "class_name": {"type": "string", "description": "Name of label to add."}
        },
        "required": ["class_name"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_add_class(args: Dict[str, Any]) -> Dict[str, Any]:
    c_name = (args.get("class_name") or args.get("label_name") or "").strip()
    if not c_name:
        raise ValueError("class_name cannot be empty.")
    database.add_class_label(c_name)
    return {"success": True, "class_name": c_name}


@register_tool(
    name="z2yy_delete_class",
    category="classes",
    description="Delete an existing class label from the registry.",
    parameters={
        "type": "object",
        "properties": {
            "class_name": {"type": "string", "description": "Name of label to delete."}
        },
        "required": ["class_name"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_delete_class(args: Dict[str, Any]) -> Dict[str, Any]:
    c_name = (args.get("class_name") or args.get("label_name") or "").strip()
    database.delete_class_label(c_name)
    return {"success": True, "class_name": c_name}


@register_tool(
    name="z2yy_set_class_prompt",
    category="classes",
    description="Configure the SAM 3 / open-vocabulary natural language prompt for a class (e.g. class_name: 'cone', sam3_prompt: 'traffic orange cone'). Essential for zero-shot text auto-labeling.",
    parameters={
        "type": "object",
        "properties": {
            "class_name": {"type": "string", "description": "Target class name."},
            "sam3_prompt": {"type": "string", "description": "Descriptive English phrase characterizing the object."}
        },
        "required": ["class_name", "sam3_prompt"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_set_class_prompt(args: Dict[str, Any]) -> Dict[str, Any]:
    c_name = (args.get("class_name") or args.get("label_name") or "").strip()
    prompt = (args.get("sam3_prompt") or "").strip()
    if not c_name:
        raise ValueError("class_name is required.")
    database.set_class_sam3_prompt(c_name, prompt)
    return {"success": True, "class_name": c_name, "sam3_prompt": prompt}


@register_tool(
    name="z2yy_list_class_prompts",
    category="classes",
    description="Retrieve all class labels along with their configured open-vocab text prompts.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"classes": {"type": "array"}}}
)
def handle_list_class_prompts(args: Dict[str, Any]) -> Dict[str, Any]:
    prompts = database.get_all_class_labels_with_prompts()
    normalized = [
        {"class_name": c["label_name"], "label_name": c["label_name"], "sam3_prompt": c.get("sam3_prompt")}
        for c in prompts
    ]
    return {"classes": normalized}


@register_tool(
    name="z2yy_save_pose_schema",
    category="classes",
    description="Define or update keypoint names and skeleton connections for a pose estimation class.",
    parameters={
        "type": "object",
        "properties": {
            "class_name": {"type": "string", "description": "Target class name."},
            "keypoints": {"type": "array", "description": "List of keypoint names in order."},
            "skeleton": {"type": "array", "description": "List of 2-element index pairs connecting keypoints."}
        },
        "required": ["class_name", "keypoints"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_save_pose_schema(args: Dict[str, Any]) -> Dict[str, Any]:
    c_name = args.get("class_name")
    kpts = args.get("keypoints", [])
    skel = args.get("skeleton", [])
    raw_dict = {
        "name": c_name,
        "keypoints": kpts,
        "points": kpts,
        "skeleton": skel,
        "edges": skel
    }
    normalized = database.normalize_pose_schema(raw_dict)
    database.set_class_keypoint_schema(c_name, normalized)
    return {"success": True, "class_name": c_name, "schema": normalized}


@register_tool(
    name="z2yy_get_pose_schemas",
    category="classes",
    description="List all registered pose keypoint schemas across classes.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"schemas": {"type": "object"}}}
)
def handle_get_pose_schemas(args: Dict[str, Any]) -> Dict[str, Any]:
    schemas = database.get_all_class_keypoint_schemas()
    return {"schemas": schemas}

@register_tool(
    name="z2yy_rename_class",
    category="classes",
    description="Rename an existing annotation class label across the workspace, updating the class registry and all existing frame annotations.",
    parameters={
        "type": "object",
        "properties": {
            "old_class_name": {"type": "string", "description": "Current name of class to rename."},
            "new_class_name": {"type": "string", "description": "New target class name."}
        },
        "required": ["old_class_name", "new_class_name"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "updated_frames_count": {"type": "integer"}}}
)
def handle_rename_class(args: Dict[str, Any]) -> Dict[str, Any]:
    old_name = args.get("old_class_name", "").strip()
    new_name = args.get("new_class_name", "").strip()
    if not old_name or not new_name:
        raise ValueError("Both old_class_name and new_class_name are required.")
    if old_name == new_name:
        return {"success": True, "old_class_name": old_name, "new_class_name": new_name, "updated_frames_count": 0}

    updated_count = 0
    with database.engine.begin() as conn:
        conn.execute(
            database.text("UPDATE class_labels SET label_name = :new WHERE label_name = :old"),
            {"old": old_name, "new": new_name}
        )
        conn.execute(
            database.text("UPDATE frame_labels SET label_name = :new WHERE label_name = :old"),
            {"old": old_name, "new": new_name}
        )
        frames = conn.execute(
            database.text("SELECT frame_id, annotations_json FROM video_frames WHERE annotations_json LIKE :pattern"),
            {"pattern": f"%{old_name}%"}
        ).fetchall()
        for row in frames:
            fid, ann_raw = row
            if not ann_raw:
                continue
            try:
                ann_data = json.loads(ann_raw)
                changed = False
                for obj in ann_data.get("objects", []):
                    if obj.get("label") == old_name:
                        obj["label"] = new_name
                        changed = True
                cls_list = ann_data.get("classifications", [])
                if old_name in cls_list:
                    ann_data["classifications"] = [new_name if c == old_name else c for c in cls_list]
                    changed = True
                if changed:
                    conn.execute(
                        database.text("UPDATE video_frames SET annotations_json = :ann WHERE frame_id = :fid"),
                        {"ann": json.dumps(ann_data), "fid": fid}
                    )
                    updated_count += 1
            except Exception:
                pass
    return {"success": True, "old_class_name": old_name, "new_class_name": new_name, "updated_frames_count": updated_count}


@register_tool(
    name="z2yy_check_class_usage",
    category="classes",
    description="Check how many video frames globally use a specific class label.",
    parameters={
        "type": "object",
        "properties": {
            "class_name": {"type": "string", "description": "Target class name."}
        },
        "required": ["class_name"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "usage_count": {"type": "integer"}}}
)
def handle_check_class_usage(args: Dict[str, Any]) -> Dict[str, Any]:
    c_name = args.get("class_name", "").strip()
    if not c_name:
        raise ValueError("class_name is required.")
    with database.engine.connect() as conn:
        count = conn.execute(
            database.text("SELECT COUNT(DISTINCT frame_id) FROM frame_labels WHERE label_name = :ln"),
            {"ln": c_name}
        ).scalar()
    return {"success": True, "class_name": c_name, "usage_count": count or 0}



# ==============================================================================
# CATEGORY 5: AUTONOMOUS AI ANNOTATION (ZERO-TOUCH AI)
# ==============================================================================

@register_tool(
    name="z2yy_sam3_text_predict",
    category="annotation",
    description="Perform zero-shot open-vocabulary detection & segmentation on a specific video frame using a text prompt. Returns bounding boxes, segmentation masks, and confidence scores.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string", "description": "UUID of the video stream."},
            "frame_number": {"type": "integer", "description": "0-based index of frame to query."},
            "text_prompt": {"type": "string", "description": "Description of object to detect (e.g. 'red soda can')."},
            "confidence": {"type": "number", "default": 0.25, "description": "Detection confidence threshold."}
        },
        "required": ["video_uuid", "frame_number", "text_prompt"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "results": {"type": "array"}}}
)
def handle_sam3_text_predict(args: Dict[str, Any]) -> Dict[str, Any]:
    from ultralytics_sam_tasks import sam3_query_frame
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    prompt = args.get("text_prompt")
    conf = float(args.get("confidence", 0.25))

    results = sam3_query_frame(v_uuid, f_num, text_prompt=prompt, confidence=conf)
    return {"success": True, "video_uuid": v_uuid, "frame_number": f_num, "results": results}


@register_tool(
    name="z2yy_sam_prompt_predict",
    category="annotation",
    description="Interactive point-guided or box-guided SAM 2 segmentation on a specific video frame.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "points": {"type": "array", "description": "List of [x, y] coordinates."},
            "point_labels": {"type": "array", "description": "1 for positive, 0 for negative point."},
            "box": {"type": "array", "description": "[x1, y1, x2, y2] bounding box prompt."}
        },
        "required": ["video_uuid", "frame_number"]
    },
    returns={"type": "object"}
)
def handle_sam_prompt_predict(args: Dict[str, Any]) -> Dict[str, Any]:
    from ultralytics_sam_tasks import sam_predict
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    pts = args.get("points")
    labels = args.get("point_labels")
    box = args.get("box")

    res = sam_predict(v_uuid, f_num, points=pts, point_labels=labels, box=box)
    return {"success": True, "results": res}


@register_tool(
    name="z2yy_lam_predict",
    category="annotation",
    description="Perform LocateAnything multi-modal open-vocabulary object detection on a specific video frame.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "text_prompt": {"type": "string"},
            "confidence": {"type": "number", "default": 0.25}
        },
        "required": ["video_uuid", "frame_number", "text_prompt"]
    },
    returns={"type": "object"}
)
def handle_lam_predict(args: Dict[str, Any]) -> Dict[str, Any]:
    import ai_models
    import cv2
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    prompt = args.get("text_prompt")
    conf = float(args.get("confidence", 0.25))

    frame_path = file_storage.get_frame_path(v_uuid, f_num)
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Frame {f_num} not found in video {v_uuid}")

    img = cv2.imread(frame_path)
    res = ai_models.lam_predict(img, prompt, conf)
    return {"success": True, "results": res}


@register_tool(
    name="z2yy_pose_predict",
    category="annotation",
    description="Run GKDT pose estimation on a frame, guided by text description or bounding boxes.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "text_prompt": {"type": "string", "description": "Text prompt to locate subject before pose extraction."},
            "bbox": {"type": "array", "description": "Optional [x1, y1, x2, y2] to bypass text detection."}
        },
        "required": ["video_uuid", "frame_number"]
    },
    returns={"type": "object"}
)
def handle_pose_predict(args: Dict[str, Any]) -> Dict[str, Any]:
    import cv2
    from gkdt_tasks import gkdt_text_pose_predict, gkdt_bbox_pose_predict
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    prompt = args.get("text_prompt", "")
    bbox = args.get("bbox")

    frame_path = file_storage.get_frame_path(v_uuid, f_num)
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    if bbox:
        res = gkdt_bbox_pose_predict(img, bbox)
    else:
        res = gkdt_text_pose_predict(img, prompt)
    return {"success": True, "results": res}


@register_tool(
    name="z2yy_batch_apply_class_to_videos",
    category="annotation",
    description="AUTONOMOUS MACRO: Automatically runs SAM 3 open-vocabulary detection and segmentation for a given class across entire video(s) in background. Labels all frames matching the class prompt.",
    parameters={
        "type": "object",
        "properties": {
            "class_name": {"type": "string", "description": "Class name whose prompt will be queried."},
            "video_uuids": {"type": "array", "description": "List of video UUIDs to auto-label."},
            "confidence_threshold": {"type": "number", "default": 0.5, "description": "Detection confidence cutoff."},
            "process_all_frames": {"type": "boolean", "default": True, "description": "True to process all frames, False for unannotated only."}
        },
        "required": ["class_name", "video_uuids"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "task_uuid": {"type": "string"}}}
)
def handle_batch_apply_class_to_videos(args: Dict[str, Any]) -> Dict[str, Any]:
    import threading
    from flask import current_app
    class_name = args.get("class_name")
    video_uuids = args.get("video_uuids", [])
    conf = float(args.get("confidence_threshold", 0.5))
    process_all = bool(args.get("process_all_frames", True))

    if not video_uuids:
        raise ValueError("video_uuids list cannot be empty.")
    if not class_name:
        raise ValueError("class_name is required.")

    task_uuid = str(uuid.uuid4())
    app_ctx = None
    try:
        app_ctx = current_app.app_context()
    except RuntimeError:
        pass

    threading.Thread(
        target=background_tasks.apply_class_to_videos_task,
        args=(task_uuid, video_uuids, class_name, conf, app_ctx, process_all),
        name=f"AgentBatchApply-{task_uuid[:8]}"
    ).start()

    return {
        "success": True,
        "task_uuid": task_uuid,
        "class_name": class_name,
        "video_uuids": video_uuids,
        "status": "RUNNING",
        "message": "Batch auto-labeling queued in background. Poll status via z2yy_get_task_status."
    }


@register_tool(
    name="z2yy_batch_apply_pose_to_videos",
    category="annotation",
    description="AUTONOMOUS MACRO: Run automated pose keypoint extraction across video(s) in background.",
    parameters={
        "type": "object",
        "properties": {
            "class_name": {"type": "string"},
            "video_uuids": {"type": "array"},
            "confidence_threshold": {"type": "number", "default": 0.5}
        },
        "required": ["class_name", "video_uuids"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "task_uuid": {"type": "string"}}}
)
def handle_batch_apply_pose_to_videos(args: Dict[str, Any]) -> Dict[str, Any]:
    import threading
    from flask import current_app
    class_name = args.get("class_name")
    video_uuids = args.get("video_uuids", [])
    conf = float(args.get("confidence_threshold", 0.5))

    task_uuid = str(uuid.uuid4())
    app_ctx = None
    try:
        app_ctx = current_app.app_context()
    except RuntimeError:
        pass

    threading.Thread(
        target=background_tasks.apply_pose_class_to_videos_task,
        args=(task_uuid, video_uuids, class_name, conf, app_ctx),
        name=f"AgentBatchPose-{task_uuid[:8]}"
    ).start()

    return {"success": True, "task_uuid": task_uuid, "status": "RUNNING"}


@register_tool(
    name="z2yy_pre_annotate_with_model",
    category="annotation",
    description="AUTONOMOUS MACRO: Run background pre-annotation on a video using a custom trained model (YOLO).",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "model_uuid": {"type": "string"},
            "confidence": {"type": "number", "default": 0.5}
        },
        "required": ["video_uuid", "model_uuid"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_pre_annotate_with_model(args: Dict[str, Any]) -> Dict[str, Any]:
    import threading
    v_uuid = args.get("video_uuid")
    m_uuid = args.get("model_uuid")
    conf = float(args.get("confidence", 0.5))

    threading.Thread(
        target=background_tasks.pre_annotate_task,
        args=(v_uuid, m_uuid, conf),
        name=f"AgentPreAnnotate-{v_uuid[:6]}"
    ).start()

    return {"success": True, "video_uuid": v_uuid, "model_uuid": m_uuid, "status": "PRE_ANNOTATING"}

@register_tool(
    name="z2yy_find_similar_objects",
    category="annotation",
    description="Perform visual one-shot similarity search using feature embeddings on the current frame or across the dataset.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "reference_bbox": {"type": "array", "description": "[x1, y1, x2, y2] positive sample bounding box."},
            "search_scope": {"type": "string", "enum": ["frame", "dataset"], "default": "frame"},
            "use_color": {"type": "boolean", "default": False}
        },
        "required": ["video_uuid", "frame_number", "reference_bbox"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "results": {"type": "array"}}}
)
def handle_find_similar_objects(args: Dict[str, Any]) -> Dict[str, Any]:
    import ai_models
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    bbox = args.get("reference_bbox")
    scope = args.get("search_scope", "frame")
    use_col = bool(args.get("use_color", False))

    if scope == "dataset":
        res = ai_models.predict_from_dataset(v_uuid, f_num, bbox, use_color=use_col)
    else:
        res = ai_models.predict_from_one_shot(v_uuid, f_num, bbox, use_color=use_col)
    return {"success": True, "video_uuid": v_uuid, "scope": scope, "results": res}



# ==============================================================================
# CATEGORY 6: OBJECT TRACKING & TEMPORAL PROPAGATION (SAM 2)
# ==============================================================================

@register_tool(
    name="z2yy_start_tracking",
    category="tracking",
    description="Initialize SAM 2 video tracking session from a keyframe bounding box prompt.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "start_frame": {"type": "integer"},
            "bbox": {"type": "array", "description": "[x1, y1, x2, y2] bounding box to track."},
            "class_name": {"type": "string"}
        },
        "required": ["video_uuid", "start_frame", "bbox", "class_name"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "tracker_uuid": {"type": "string"}}}
)
def handle_start_tracking(args: Dict[str, Any]) -> Dict[str, Any]:
    import threading
    v_uuid = args.get("video_uuid")
    start_f = int(args.get("start_frame"))
    bbox = args.get("bbox")
    c_name = args.get("class_name")

    tracker_uuid = str(uuid.uuid4())
    threading.Thread(
        target=background_tasks.sam2_tracking_task,
        args=(v_uuid, start_f, bbox, c_name, tracker_uuid),
        name=f"AgentTracker-{tracker_uuid[:6]}"
    ).start()

    return {"success": True, "tracker_uuid": tracker_uuid, "video_uuid": v_uuid}


@register_tool(
    name="z2yy_retrieve_tracked_bboxes",
    category="tracking",
    description="Retrieve tracked object bounding boxes across frame sequences during or after a tracking session.",
    parameters={
        "type": "object",
        "properties": {
            "tracker_uuid": {"type": "string"}
        },
        "required": ["tracker_uuid"]
    },
    returns={"type": "object", "properties": {"bboxes": {"type": "array"}}}
)
def handle_retrieve_tracked_bboxes(args: Dict[str, Any]) -> Dict[str, Any]:
    t_uuid = args.get("tracker_uuid")
    results = background_tasks.tracked_bboxes_results.get(t_uuid, [])
    return {"tracker_uuid": t_uuid, "bboxes": results}


@register_tool(
    name="z2yy_stop_tracking",
    category="tracking",
    description="Finish and terminate a tracking session, committing tracked annotations to video frames.",
    parameters={
        "type": "object",
        "properties": {
            "tracker_uuid": {"type": "string"}
        },
        "required": ["tracker_uuid"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_stop_tracking(args: Dict[str, Any]) -> Dict[str, Any]:
    t_uuid = args.get("tracker_uuid")
    background_tasks.stop_tracking(t_uuid)
    return {"success": True, "tracker_uuid": t_uuid}

@register_tool(
    name="z2yy_sam2_track_video",
    category="tracking",
    description="Launch SAM 2 video mask propagation tracking from start_frame to end_frame in background.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "start_frame": {"type": "integer"},
            "end_frame": {"type": "integer"},
            "init_bboxes_text": {"type": "string", "description": "Bounding box definitions on start_frame to initiate track."}
        },
        "required": ["video_uuid", "start_frame", "end_frame", "init_bboxes_text"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "tracker_uuid": {"type": "string"}}}
)
def handle_sam2_track_video(args: Dict[str, Any]) -> Dict[str, Any]:
    import threading
    v_uuid = args.get("video_uuid")
    start_f = int(args.get("start_frame"))
    end_f = int(args.get("end_frame"))
    bboxes_text = args.get("init_bboxes_text")

    tracker_uuid = str(uuid.uuid4())
    threading.Thread(
        target=background_tasks.start_sam2_batch_tracking_task,
        args=(v_uuid, tracker_uuid, start_f, end_f, bboxes_text),
        name=f"AgentSAM2Tracker-{tracker_uuid[:6]}"
    ).start()
    return {"success": True, "tracker_uuid": tracker_uuid, "video_uuid": v_uuid}



# ==============================================================================
# CATEGORY 7: FRAME ANNOTATIONS & INTERPOLATION
# ==============================================================================

@register_tool(
    name="z2yy_get_frame_annotations",
    category="annotations",
    description="Retrieve all annotations (bounding boxes, polygons, keypoints) on a specific video frame.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"}
        },
        "required": ["video_uuid", "frame_number"]
    },
    returns={"type": "object", "properties": {"annotations": {"type": "object"}}}
)
def handle_get_frame_annotations(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    data = database.get_frame_annotations(v_uuid, f_num)
    return {"video_uuid": v_uuid, "frame_number": f_num, "annotations": data or {"objects": []}}


@register_tool(
    name="z2yy_save_frame_annotations",
    category="annotations",
    description="Save or overwrite annotations on a specific video frame with structured objects (bboxes, labels, polygons, keypoints).",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "annotations": {
                "type": "object",
                "description": "AnnotationData dictionary containing 'objects' list."
            }
        },
        "required": ["video_uuid", "frame_number", "annotations"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_save_frame_annotations(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    ann_dict = args.get("annotations", {})

    ann_json = json.dumps(ann_dict) if isinstance(ann_dict, dict) else str(ann_dict)
    database.save_frame_annotations(v_uuid, f_num, ann_json)
    return {"success": True, "video_uuid": v_uuid, "frame_number": f_num}


@register_tool(
    name="z2yy_interpolate_bboxes",
    category="annotations",
    description="Linearly interpolate bounding box positions for an object between two keyframes across intermediate frames.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "start_frame": {"type": "integer"},
            "end_frame": {"type": "integer"},
            "start_bbox": {"type": "array", "description": "[x1, y1, x2, y2]"},
            "end_bbox": {"type": "array", "description": "[x1, y1, x2, y2]"},
            "class_name": {"type": "string"}
        },
        "required": ["video_uuid", "start_frame", "end_frame", "start_bbox", "end_bbox", "class_name"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "interpolated_frames": {"type": "integer"}}}
)
def handle_interpolate_bboxes(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    sf = int(args.get("start_frame"))
    ef = int(args.get("end_frame"))
    sb = args.get("start_bbox")
    eb = args.get("end_bbox")
    cls = args.get("class_name")

    if ef <= sf:
        raise ValueError("end_frame must be greater than start_frame")

    total_frames = ef - sf
    interpolated = 0
    for i in range(1, total_frames):
        alpha = i / total_frames
        curr_f = sf + i
        interp_box = [
            round(sb[j] + alpha * (eb[j] - sb[j]), 2)
            for j in range(4)
        ]
        curr_ann = database.get_frame_annotations(v_uuid, curr_f) or {"objects": []}
        curr_ann["objects"].append({
            "id": str(uuid.uuid4()),
            "type": "bbox",
            "label": cls,
            "bbox": interp_box
        })
        database.save_frame_annotations(v_uuid, curr_f, json.dumps(curr_ann))
        interpolated += 1

    return {"success": True, "interpolated_frames": interpolated}


@register_tool(
    name="z2yy_interpolate_pose_keypoints",
    category="annotations",
    description="Smoothly interpolate skeleton keypoint positions between two keyframes across intermediate frames.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "start_frame": {"type": "integer"},
            "end_frame": {"type": "integer"},
            "start_keypoints": {"type": "array"},
            "end_keypoints": {"type": "array"},
            "class_name": {"type": "string"}
        },
        "required": ["video_uuid", "start_frame", "end_frame", "start_keypoints", "end_keypoints", "class_name"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_interpolate_pose_keypoints(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    sf = int(args.get("start_frame"))
    ef = int(args.get("end_frame"))
    sk = args.get("start_keypoints")
    ek = args.get("end_keypoints")
    cls = args.get("class_name")

    if ef <= sf or len(sk) != len(ek):
        raise ValueError("Invalid frame range or keypoint length mismatch.")

    total = ef - sf
    for i in range(1, total):
        alpha = i / total
        curr_f = sf + i
        interp_kpts = []
        for p1, p2 in zip(sk, ek):
            interp_kpts.append({
                "x": round(p1.get("x", 0) + alpha * (p2.get("x", 0) - p1.get("x", 0)), 2),
                "y": round(p1.get("y", 0) + alpha * (p2.get("y", 0) - p1.get("y", 0)), 2),
                "v": min(p1.get("v", 2), p2.get("v", 2))
            })
        curr_ann = database.get_frame_annotations(v_uuid, curr_f) or {"objects": []}
        curr_ann["objects"].append({
            "id": str(uuid.uuid4()),
            "type": "keypoints",
            "label": cls,
            "keypoints": interp_kpts
        })
        database.save_frame_annotations(v_uuid, curr_f, json.dumps(curr_ann))

    return {"success": True, "interpolated_frames": total - 1}

@register_tool(
    name="z2yy_add_annotation_object",
    category="annotations",
    description="Add or update a single annotation object (bounding box, segmentation polygon, or pose keypoint skeleton) on a specific video frame.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string", "description": "UUID of the video stream."},
            "frame_number": {"type": "integer", "description": "Target frame number."},
            "type": {"type": "string", "enum": ["bbox", "polygon", "keypoint"], "default": "bbox"},
            "label": {"type": "string", "description": "Class name for this object."},
            "bbox": {"type": "array", "description": "[x1, y1, x2, y2] bounding box coordinates."},
            "points": {"type": "array", "description": "[[x, y], ...] polygon vertex coordinates for segmentation."},
            "keypoints": {"type": "array", "description": "[{'name': '...', 'x': x, 'y': y, 'v': v}, ...] keypoint list for pose estimation."},
            "object_id": {"type": "string", "description": "Optional custom unique identifier for this object."}
        },
        "required": ["video_uuid", "frame_number", "label"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "object_id": {"type": "string"}}}
)
def handle_add_annotation_object(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    obj_type = args.get("type", "bbox")
    label = args.get("label")
    bbox = args.get("bbox")
    points = args.get("points")
    keypoints = args.get("keypoints")
    obj_id = args.get("object_id") or f"{obj_type}_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"

    ann_dict = database.get_frame_annotations(v_uuid, f_num) or {"version": 1, "objects": [], "classifications": [], "is_ambiguous": False}
    objs = ann_dict.setdefault("objects", [])
    
    existing_idx = next((i for i, o in enumerate(objs) if o.get("id") == obj_id), None)
    new_obj = {
        "id": obj_id,
        "type": obj_type,
        "label": label,
        "bbox": bbox,
        "points": points,
        "keypoints": keypoints
    }
    if existing_idx is not None:
        objs[existing_idx] = new_obj
    else:
        objs.append(new_obj)

    database.save_frame_annotations(v_uuid, f_num, json.dumps(ann_dict))
    return {"success": True, "object_id": obj_id, "video_uuid": v_uuid, "frame_number": f_num}


@register_tool(
    name="z2yy_delete_annotation_object",
    category="annotations",
    description="Remove a specific annotation object by its ID from a video frame.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "object_id": {"type": "string", "description": "ID of the object to delete."}
        },
        "required": ["video_uuid", "frame_number", "object_id"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_delete_annotation_object(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    obj_id = args.get("object_id")

    ann_dict = database.get_frame_annotations(v_uuid, f_num)
    if not ann_dict or "objects" not in ann_dict:
        return {"success": True, "deleted": False, "message": "No annotations found on frame."}
    
    initial_len = len(ann_dict["objects"])
    ann_dict["objects"] = [o for o in ann_dict["objects"] if o.get("id") != obj_id]
    deleted = len(ann_dict["objects"]) < initial_len
    if deleted:
        database.save_frame_annotations(v_uuid, f_num, json.dumps(ann_dict))
    return {"success": True, "deleted": deleted, "object_id": obj_id}


@register_tool(
    name="z2yy_clear_frame_annotations",
    category="annotations",
    description="Clear all annotation objects and classification tags on a specific video frame.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"}
        },
        "required": ["video_uuid", "frame_number"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_clear_frame_annotations(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    empty_ann = {"version": 1, "objects": [], "classifications": [], "is_ambiguous": False}
    database.save_frame_annotations(v_uuid, f_num, json.dumps(empty_ann))
    return {"success": True, "video_uuid": v_uuid, "frame_number": f_num}


@register_tool(
    name="z2yy_set_frame_ambiguous",
    category="annotations",
    description="Flag or unflag a video frame as ambiguous (requiring human review or disambiguation).",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "is_ambiguous": {"type": "boolean", "default": True}
        },
        "required": ["video_uuid", "frame_number"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "is_ambiguous": {"type": "boolean"}}}
)
def handle_set_frame_ambiguous(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    is_amb = bool(args.get("is_ambiguous", True))

    ann_dict = database.get_frame_annotations(v_uuid, f_num) or {"version": 1, "objects": [], "classifications": [], "is_ambiguous": False}
    ann_dict["is_ambiguous"] = is_amb
    database.save_frame_annotations(v_uuid, f_num, json.dumps(ann_dict))
    return {"success": True, "video_uuid": v_uuid, "frame_number": f_num, "is_ambiguous": is_amb}


@register_tool(
    name="z2yy_get_ambiguous_frames",
    category="annotations",
    description="List all frame numbers in a video currently flagged as ambiguous.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"}
        },
        "required": ["video_uuid"]
    },
    returns={"type": "object", "properties": {"ambiguous_frames": {"type": "array"}, "count": {"type": "integer"}}}
)
def handle_get_ambiguous_frames(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    frames = database.get_video_frames(v_uuid)
    ambiguous = []
    for f in frames:
        ann_json = (f.get("annotations_json") or "").strip()
        if ann_json:
            try:
                ad = json.loads(ann_json)
                if ad.get("is_ambiguous"):
                    ambiguous.append(f.get("frame_number"))
            except Exception:
                pass
    return {"video_uuid": v_uuid, "count": len(ambiguous), "ambiguous_frames": ambiguous}



# ==============================================================================
# CATEGORY 8: ZERO-SHOT CLASSIFICATION & FEATURE CLUSTERING (CLIP)
# ==============================================================================

@register_tool(
    name="z2yy_clip_zero_shot",
    category="classification",
    description="Classify cropped objects or frames against a list of candidate class text descriptions using CLIP zero-shot matching.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "candidate_classes": {"type": "array", "description": "Candidate text labels."},
            "bbox": {"type": "array", "description": "Optional [x1, y1, x2, y2] crop region."}
        },
        "required": ["video_uuid", "frame_number", "candidate_classes"]
    },
    returns={"type": "object", "properties": {"best_class": {"type": "string"}, "probabilities": {"type": "object"}}}
)
def handle_clip_zero_shot(args: Dict[str, Any]) -> Dict[str, Any]:
    import cv2
    import clip_model
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    candidates = args.get("candidate_classes", [])
    bbox = args.get("bbox")

    frame_path = file_storage.get_frame_path(v_uuid, f_num)
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Frame image not found: {frame_path}")

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        img = img[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]

    probs = clip_model.classify_image_zero_shot(img, candidates)
    best = max(probs.items(), key=lambda x: x[1])[0] if probs else None
    return {"best_class": best, "probabilities": probs}


@register_tool(
    name="z2yy_cluster_images",
    category="classification",
    description="Extract visual embeddings and run K-Means clustering across detected objects or frames in a video.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "k_clusters": {"type": "integer", "default": 8}
        },
        "required": ["video_uuid"]
    },
    returns={"type": "object", "properties": {"clusters": {"type": "object"}}}
)
def handle_cluster_images(args: Dict[str, Any]) -> Dict[str, Any]:
    import clip_model
    v_uuid = args.get("video_uuid")
    k = int(args.get("k_clusters", 8))
    clusters = clip_model.cluster_video_objects(v_uuid, k)
    return {"video_uuid": v_uuid, "clusters": clusters}


@register_tool(
    name="z2yy_batch_tag_cluster",
    category="classification",
    description="Assign an annotation class label to all frames/objects belonging to a specific visual cluster.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "cluster_id": {"type": "integer"},
            "class_name": {"type": "string"},
            "frame_numbers": {"type": "array", "description": "List of frame numbers in cluster."}
        },
        "required": ["video_uuid", "class_name", "frame_numbers"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_batch_tag_cluster(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    cls = args.get("class_name")
    frames = args.get("frame_numbers", [])

    for fn in frames:
        curr = database.get_frame_annotations(v_uuid, int(fn)) or {"objects": []}
        curr.setdefault("classifications", []).append(cls)
        database.save_frame_annotations(v_uuid, int(fn), json.dumps(curr))

    return {"success": True, "tagged_count": len(frames), "class_name": cls}

@register_tool(
    name="z2yy_set_frame_classifications",
    category="classification",
    description="Set or replace the whole-image classification labels for a specific video frame.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"},
            "classifications": {"type": "array", "description": "List of class names assigned to this image."}
        },
        "required": ["video_uuid", "frame_number", "classifications"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_set_frame_classifications(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    classes = args.get("classifications", [])

    ann_dict = database.get_frame_annotations(v_uuid, f_num) or {"version": 1, "objects": [], "classifications": [], "is_ambiguous": False}
    ann_dict["classifications"] = list(dict.fromkeys(classes))
    database.save_frame_annotations(v_uuid, f_num, json.dumps(ann_dict))
    return {"success": True, "video_uuid": v_uuid, "frame_number": f_num, "classifications": ann_dict["classifications"]}


@register_tool(
    name="z2yy_clear_frame_classifications",
    category="classification",
    description="Clear all whole-image classification labels for a specific video frame.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"},
            "frame_number": {"type": "integer"}
        },
        "required": ["video_uuid", "frame_number"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_clear_frame_classifications(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    f_num = int(args.get("frame_number"))
    ann_dict = database.get_frame_annotations(v_uuid, f_num) or {"version": 1, "objects": [], "classifications": [], "is_ambiguous": False}
    ann_dict["classifications"] = []
    database.save_frame_annotations(v_uuid, f_num, json.dumps(ann_dict))
    return {"success": True, "video_uuid": v_uuid, "frame_number": f_num}



# ==============================================================================
# CATEGORY 9: BACKGROUND TASK MANAGEMENT
# ==============================================================================

@register_tool(
    name="z2yy_list_tasks",
    category="tasks",
    description="List active and pending background tasks across the system.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string", "description": "Optional filter by video UUID."}
        }
    },
    returns={"type": "object", "properties": {"tasks": {"type": "array"}}}
)
def handle_list_tasks(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    if v_uuid:
        tasks = database.get_tasks_for_video(v_uuid)
    else:
        tasks = database.get_all_tasks() if hasattr(database, "get_all_tasks") else []
    active = dict(background_tasks.active_tasks)
    batch_sessions = {k: v.get("status") for k, v in background_tasks.batch_apply_sessions.items()}
    return {"tasks": tasks, "active_video_locks": active, "batch_sessions": batch_sessions}


@register_tool(
    name="z2yy_get_task_status",
    category="tasks",
    description="Poll progress, completion percentage, and result of an active background task (e.g. batch auto-labeling, tracking, frame extraction).",
    parameters={
        "type": "object",
        "properties": {
            "task_uuid": {"type": "string", "description": "Task UUID to query."}
        },
        "required": ["task_uuid"]
    },
    returns={"type": "object", "properties": {"status": {"type": "string"}}}
)
def handle_get_task_status(args: Dict[str, Any]) -> Dict[str, Any]:
    t_uuid = args.get("task_uuid")
    session = background_tasks.batch_apply_sessions.get(t_uuid)
    if session:
        return {"task_uuid": t_uuid, "type": "batch_apply", **session}

    # Check database tasks
    task_ent = database.get_task(t_uuid) if hasattr(database, "get_task") else None
    if task_ent:
        return {"task_uuid": t_uuid, "type": "db_task", **task_ent}

    return {"task_uuid": t_uuid, "status": "UNKNOWN_OR_COMPLETED"}


@register_tool(
    name="z2yy_cancel_task",
    category="tasks",
    description="Cancel/abort an active background task safely.",
    parameters={
        "type": "object",
        "properties": {
            "task_uuid": {"type": "string"},
            "video_uuid": {"type": "string"}
        }
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_cancel_task(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    t_uuid = args.get("task_uuid")

    if v_uuid and v_uuid in background_tasks.active_tasks:
        del background_tasks.active_tasks[v_uuid]

    if t_uuid and t_uuid in background_tasks.batch_apply_sessions:
        background_tasks.batch_apply_sessions[t_uuid]["status"] = "canceled"

    return {"success": True, "message": "Abort signal sent."}


# ==============================================================================
# CATEGORY 10: DATASET CREATION & EXPORT
# ==============================================================================

@register_tool(
    name="z2yy_list_datasets",
    category="datasets",
    description="Enumerate all created datasets, their UUIDs, video sources, statuses (PROCESSING, READY, FAILED), and export formats.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"datasets": {"type": "array"}}}
)
def handle_list_datasets(args: Dict[str, Any]) -> Dict[str, Any]:
    datasets = database.get_dataset_list()
    return {"datasets": datasets}


@register_tool(
    name="z2yy_get_dataset_info",
    category="datasets",
    description="Retrieve full metadata of a dataset including train/eval/test splits, file counts, and absolute path of compiled ZIP.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_uuid": {"type": "string", "description": "UUID of dataset."}
        },
        "required": ["dataset_uuid"]
    },
    returns={"type": "object"}
)
def handle_get_dataset_info(args: Dict[str, Any]) -> Dict[str, Any]:
    d_uuid = args.get("dataset_uuid")
    dataset = database.get_dataset_entity(d_uuid)
    if not dataset:
        raise ValueError(f"Dataset not found: {d_uuid}")
    return {"dataset": dataset}


@register_tool(
    name="z2yy_create_dataset",
    category="datasets",
    description="Compile annotated video frames into a machine-learning training dataset (splits train/val/test, exports to formats like YOLOv8/v11, COCO, VOC).",
    parameters={
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Title / name of the dataset."},
            "video_uuids": {"type": "array", "description": "List of annotated video UUIDs to include."},
            "export_format": {
                "type": "string",
                "default": "yolo_v8_detect",
                "description": "Export format: yolo_v8_detect, yolo_v8_seg, yolo_v8_pose, coco_detect, voc, yolo_v11_detect, etc."
            },
            "eval_percent": {"type": "number", "default": 20.0, "description": "Validation set percentage."},
            "test_percent": {"type": "number", "default": 10.0, "description": "Test set percentage."},
            "augmentation_options": {"type": "object", "description": "Optional Albumentations settings."}
        },
        "required": ["description", "video_uuids"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "dataset_uuid": {"type": "string"}}}
)
def handle_create_dataset(args: Dict[str, Any]) -> Dict[str, Any]:
    import threading
    desc = args.get("description", "").strip()
    video_uuids = args.get("video_uuids", [])
    fmt = args.get("export_format", "yolo_v8_detect")
    eval_p = float(args.get("eval_percent", 20.0))
    test_p = float(args.get("test_percent", 10.0))
    aug = args.get("augmentation_options", {})
    export_opts = args.get("export_options", {})

    if not desc:
        raise ValueError("description is required.")
    if not video_uuids:
        raise ValueError("video_uuids list cannot be empty.")

    create_time = int(time.time() * 1000)
    dataset_uuid = database.create_dataset_entry(
        desc, video_uuids, create_time, eval_p, test_p, fmt, export_opts, aug
    )

    threading.Thread(
        target=background_tasks.create_dataset_task,
        args=(dataset_uuid, video_uuids, eval_p, test_p, fmt, aug, export_opts),
        name=f"AgentDataset-{dataset_uuid[:6]}"
    ).start()

    return {
        "success": True,
        "dataset_uuid": dataset_uuid,
        "description": desc,
        "export_format": fmt,
        "status": "PROCESSING",
        "message": "Dataset compilation queued in background. Check status via z2yy_get_dataset_info."
    }


@register_tool(
    name="z2yy_regenerate_dataset",
    category="datasets",
    description="Recompile an existing dataset with updated annotations or settings.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_uuid": {"type": "string"}
        },
        "required": ["dataset_uuid"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_regenerate_dataset(args: Dict[str, Any]) -> Dict[str, Any]:
    import threading
    d_uuid = args.get("dataset_uuid")
    d = database.get_dataset_entity(d_uuid)
    if not d:
        raise ValueError(f"Dataset not found: {d_uuid}")

    video_uuids = json.loads(d['video_uuids']) if isinstance(d['video_uuids'], str) else d['video_uuids']
    eval_p = d.get('eval_percent', 20.0)
    test_p = d.get('test_percent', 10.0)
    fmt = d.get('export_format', 'yolo_v8_detect')
    aug = json.loads(d.get('augmentation_options') or '{}')
    opts = json.loads(d.get('export_options') or '{}')

    threading.Thread(
        target=background_tasks.create_dataset_task,
        args=(d_uuid, video_uuids, eval_p, test_p, fmt, aug, opts),
        name=f"AgentDatasetRegen-{d_uuid[:6]}"
    ).start()

    return {"success": True, "dataset_uuid": d_uuid, "status": "PROCESSING"}


@register_tool(
    name="z2yy_get_export_formats",
    category="datasets",
    description="Retrieve all supported dataset export formats with their descriptions and schema requirements.",
    parameters={"type": "object", "properties": {}},
    returns={"type": "object", "properties": {"formats": {"type": "array"}}}
)
def handle_get_export_formats(args: Dict[str, Any]) -> Dict[str, Any]:
    formats = ExporterRegistry.list_all()
    return {"formats": formats}


@register_tool(
    name="z2yy_set_dataset_augmentations",
    category="datasets",
    description="Configure data augmentation settings (rotation, crop, flip, HSV adjustments) for a dataset.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_uuid": {"type": "string"},
            "augmentation_options": {"type": "object"}
        },
        "required": ["dataset_uuid", "augmentation_options"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_set_dataset_augmentations(args: Dict[str, Any]) -> Dict[str, Any]:
    d_uuid = args.get("dataset_uuid")
    aug = args.get("augmentation_options", {})
    database.update_dataset_augmentation_options(d_uuid, json.dumps(aug))
    return {"success": True, "dataset_uuid": d_uuid, "augmentation_options": aug}


@register_tool(
    name="z2yy_delete_dataset",
    category="datasets",
    description="Delete a dataset and its exported ZIP and folder files from disk.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_uuid": {"type": "string"}
        },
        "required": ["dataset_uuid"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
)
def handle_delete_dataset(args: Dict[str, Any]) -> Dict[str, Any]:
    d_uuid = args.get("dataset_uuid")
    database.delete_dataset(d_uuid)
    file_storage.delete_dataset_files(d_uuid)
    return {"success": True, "dataset_uuid": d_uuid}

@register_tool(
    name="z2yy_download_dataset",
    category="datasets",
    description="Download or locate the compiled dataset ZIP archive for a completed dataset.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_uuid": {"type": "string", "description": "UUID of compiled dataset."},
            "target_dir": {"type": "string", "description": "Optional destination folder to copy the ZIP to."}
        },
        "required": ["dataset_uuid"]
    },
    returns={"type": "object", "properties": {"success": {"type": "boolean"}, "zip_path": {"type": "string"}}}
)
def handle_download_dataset(args: Dict[str, Any]) -> Dict[str, Any]:
    d_uuid = args.get("dataset_uuid")
    target_dir = args.get("target_dir")
    zip_path = file_storage.get_dataset_zip_path(d_uuid)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip archive not found for {d_uuid}")
    
    final_path = zip_path
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        final_path = os.path.join(target_dir, os.path.basename(zip_path))
        shutil.copy2(zip_path, final_path)
    return {"success": True, "dataset_uuid": d_uuid, "zip_path": final_path, "size_bytes": os.path.getsize(final_path)}


@register_tool(
    name="z2yy_datasets_containing_video",
    category="datasets",
    description="Find all compiled or existing datasets that include a specific video.",
    parameters={
        "type": "object",
        "properties": {
            "video_uuid": {"type": "string"}
        },
        "required": ["video_uuid"]
    },
    returns={"type": "object", "properties": {"matching_datasets": {"type": "array"}}}
)
def handle_datasets_containing_video(args: Dict[str, Any]) -> Dict[str, Any]:
    v_uuid = args.get("video_uuid")
    matching = []
    for ds in database.get_dataset_list():
        try:
            ds_video_uuids = json.loads(ds.get("video_uuids") or "[]")
            if v_uuid in ds_video_uuids:
                matching.append(ds)
        except Exception:
            pass
    return {"video_uuid": v_uuid, "matching_datasets": matching}



# ==============================================================================
# UNIFIED TOOL DISPATCHER & EXECUTOR
# ==============================================================================

def execute_tool(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute any registered tool with standardized error boundary and JSON output.
    """
    if arguments is None:
        arguments = {}

    if tool_name not in TOOL_REGISTRY:
        return {
            "success": False,
            "tool": tool_name,
            "error": f"Unknown tool: '{tool_name}'. Run 'z2yy_list_tools' to see available tools.",
            "message": "Tool execution rejected."
        }

    tool_def = TOOL_REGISTRY[tool_name]
    required = tool_def["parameters"].get("required", [])
    missing = [req for req in required if req not in arguments or arguments[req] is None]
    if missing:
        return {
            "success": False,
            "tool": tool_name,
            "error": f"Missing required arguments: {missing}",
            "message": "Validation failed."
        }

    try:
        result = tool_def["handler"](arguments)
        return {
            "success": True,
            "tool": tool_name,
            "result": result,
            "message": "Tool executed successfully."
        }
    except Exception as e:
        logging.error(f"[AgentHub] Execution error in {tool_name}: {e}", exc_info=True)
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e),
            "message": "Tool execution encountered an exception."
        }


def get_all_tools_schema() -> List[Dict[str, Any]]:
    """
    Return all tool declarations formatted for OpenAI / Anthropic / MCP tool definitions.
    """
    schemas = []
    for name, tool in TOOL_REGISTRY.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"[{tool['category'].upper()}] {tool['description']}",
                "parameters": tool["parameters"]
            }
        })
    return schemas


# ==============================================================================
# AI-FACING SYSTEM PROMPT & DOCUMENTATION GENERATOR
# ==============================================================================

def generate_agent_prompt_markdown() -> str:
    """
    Generates the complete, pristine system prompt written strictly for AI agents
    to autonomously operate Zero2YoloYard.
    """
    tools_list = []
    for cat in ["system", "models", "videos", "classes", "annotation", "tracking", "annotations", "classification", "tasks", "datasets"]:
        cat_tools = [t for t in TOOL_REGISTRY.values() if t["category"] == cat]
        if not cat_tools:
            continue
        tools_list.append(f"### {cat.upper()} TOOLS")
        for t in cat_tools:
            params_desc = ", ".join(t["parameters"].get("properties", {}).keys()) or "none"
            req_desc = ", ".join(t["parameters"].get("required", [])) or "none"
            tools_list.append(f"- **`{t['name']}`**: {t['description']}")
            tools_list.append(f"  - Parameters: `{params_desc}` (Required: `{req_desc}`)")

    tools_markdown = "\n".join(tools_list)

    doc = f"""# ZERO2YOLOYARD (Z2YY) AUTONOMOUS AI AGENT OPERATIONAL SPECIFICATION
> **Target Audience**: Autonomous LLM Agents (Claude, GPT-4, DeepSeek, local Llama/Qwen).
> **Operating Mode**: Fully Autonomous Headless Computer Vision Data Operations.
> **Human Intervention Required**: NONE. The human user issues goals in natural language; you execute the entire pipeline end-to-end.

---

## 1. MISSION DIRECTIVE
You are the Autonomous Vision AI Engineering Agent controlling **Zero2YoloYard (Z2YY)** on the local workstation.
Your objective is to ingest raw video streams, configure classes and natural language prompts, execute open-vocabulary zero-shot auto-labeling, track objects, interpolate keyframes, build training datasets, and output production-ready training packages (YOLOv8, YOLOv11, COCO, etc.) **without requiring the human to supervise or look at the screen**.

> [!IMPORTANT]
> **Strict Constraints**:
> 1. **Zero Internet Download Waste**: Do not re-download models if local weights exist.
> 2. **QC / Review Exclusion**: Data model review / dataset audit functions are excluded from your toolset. Do not attempt to call review endpoints.
> 3. **Executable Compatibility**: All operations must function identically whether running via Python (`python app.py`) or packaged standalone executable (`Zero2YoloYard.exe`).

---

## 2. DUAL-CHANNEL EXECUTION ARCHITECTURE

You can communicate with Z2YY through two primary channels:

### Channel A: HTTP JSON API (Preferred when server is running)
- Base URL: `http://127.0.0.1:5000`
- **Universal Tool Dispatcher**: `POST /api/agent/execute`
  ```json
  {{
    "tool": "<tool_name>",
    "arguments": {{ "<arg_name>": "<value>" }}
  }}
  ```
- **Discovery Endpoints**:
  - `GET /api/agent/tools` -> Returns full JSON Schema of all tools.
  - `GET /api/agent/prompt` -> Returns this specification.

### Channel B: Native CLI Subcommand Dispatcher (Zero-Dependency)
- Directly executable in Windows Terminal / Subprocess:
  ```powershell
  Zero2YoloYard.exe agent execute <tool_name> --key value
  # Or with JSON payload:
  Zero2YoloYard.exe agent execute <tool_name> '{{"key": "value"}}'
  # List tools:
  Zero2YoloYard.exe agent list-tools
  ```

---

## 3. AUTONOMOUS PLAYBOOKS (RECIPES)

### Playbook 1: Natural Language Request -> YOLOv8 Training Dataset
**User Prompt**: *"Label all red traffic cones in C:/videos/traffic.mp4 and generate a YOLOv8 dataset ready for training."*

**Your Execution Sequence**:
1. `z2yy_upload_video`:
   `arguments = {{"file_path": "C:/videos/traffic.mp4", "description": "traffic_stream", "target_fps": 5}}`
   -> Receive `video_uuid`.
2. `z2yy_get_task_status`:
   Poll until frame extraction is complete and video status is `READY`.
3. `z2yy_add_class`:
   `arguments = {{"class_name": "traffic_cone"}}`
4. `z2yy_set_class_prompt`:
   `arguments = {{"class_name": "traffic_cone", "sam3_prompt": "bright red or orange traffic safety cone"}}`
5. `z2yy_batch_apply_class_to_videos`:
   `arguments = {{"class_name": "traffic_cone", "video_uuids": [video_uuid], "confidence_threshold": 0.5}}`
   -> Receive `task_uuid`.
6. Poll `z2yy_get_task_status(task_uuid)` until status is `completed`.
7. `z2yy_create_dataset`:
   `arguments = {{"description": "traffic_cone_dataset_yolov8", "video_uuids": [video_uuid], "export_format": "yolo_v8_detect", "eval_percent": 15.0, "test_percent": 5.0}}`
   -> Receive `dataset_uuid`.
8. Poll `z2yy_get_dataset_info(dataset_uuid)` until dataset status is `READY`.
9. Retrieve `zip_path` from dataset metadata and report completion with download path to the user!

---

### Playbook 2: Single Keyframe Detection -> Full Video SAM 2 Tracking
**User Prompt**: *"Find the drone in frame 0 of video X and track it throughout the whole video."*

1. `z2yy_sam3_text_predict(video_uuid=X, frame_number=0, text_prompt="flying quadcopter drone")`.
2. Extract the highest confidence bounding box `[x1, y1, x2, y2]`.
3. `z2yy_start_tracking(video_uuid=X, start_frame=0, bbox=..., class_name="drone")` -> `tracker_uuid`.
4. Poll `z2yy_retrieve_tracked_bboxes(tracker_uuid)` until video sequence concludes.
5. `z2yy_stop_tracking(tracker_uuid)` -> auto-commits annotations to video frames.

---

### Playbook 3: Custom Model Pre-Annotation
**User Prompt**: *"Annotate warehouse.mp4 using my trained model weights C:/models/yolo_box.pt."*

1. `z2yy_import_trained_model(description="yolo_box_v1", model_path="C:/models/yolo_box.pt", model_type="yolo_detect")` -> `model_uuid`.
2. `z2yy_upload_video(file_path="C:/videos/warehouse.mp4", description="warehouse")` -> `video_uuid`.
3. Wait for video extraction to complete.
4. `z2yy_pre_annotate_with_model(video_uuid=video_uuid, model_uuid=model_uuid, confidence=0.45)`.
5. Check task completion and notify user.

---

## 4. ERROR HANDLING & SELF-HEALING PROTOCOL

- **Missing Class Prompts**: If `z2yy_batch_apply_class_to_videos` returns 0 detections, call `z2yy_list_class_prompts` to inspect the prompt. Refine the text description (e.g. from `"cone"` to `"orange conical plastic traffic marker"`).
- **Task Busy / Lock Conflict**: If an endpoint returns HTTP 409 or `busy_videos`, query `z2yy_list_tasks` to inspect ongoing tasks, wait 5 seconds, or call `z2yy_cancel_task` if a previous task hung.
- **Hardware Fallback**: Query `z2yy_get_system_status`. If `cuda_available` is false, adjust `batch_tracking_chunk_size` or decrease `target_fps` via `z2yy_update_settings` to ensure CPU throughput.

---

## 5. COMPLETE REGISTERED TOOL CATALOG

{tools_markdown}
"""
    return doc


# ==============================================================================
# CLI DISPATCHER FOR AGENT EXECUTION & EXE COMPATIBILITY
# ==============================================================================

def run_cli(args: List[str]) -> None:
    """
    Entry point for CLI agent interactions (both Python and compiled EXE).
    Usage:
        Zero2YoloYard.exe agent list-tools
        Zero2YoloYard.exe agent describe <tool_name>
        Zero2YoloYard.exe agent execute <tool_name> [args_json or --k v]
        Zero2YoloYard.exe agent dump-prompt
        Zero2YoloYard.exe agent dump-schema
    """
    if not args or args[0] in ("-h", "--help", "help"):
        print("Zero2YoloYard Agent CLI - Autonomous Tool Subsystem")
        print("Commands:")
        print("  list-tools               List all available tools and categories")
        print("  describe <tool>          Print detailed schema for a specific tool")
        print("  execute <tool> [args]    Execute a tool locally or via server")
        print("  dump-prompt              Print complete AI System Prompt in Markdown")
        print("  dump-schema              Print complete JSON Schema of all tools")
        sys.exit(0)

    cmd = args[0]

    if cmd == "list-tools":
        print(json.dumps([{"name": k, "category": v["category"], "description": v["description"]} for k, v in TOOL_REGISTRY.items()], indent=2))
        sys.exit(0)

    elif cmd == "describe":
        if len(args) < 2:
            print("Error: Specify tool name.")
            sys.exit(1)
        name = args[1]
        if name in TOOL_REGISTRY:
            print(json.dumps(TOOL_REGISTRY[name], default=str, indent=2))
        else:
            print(f"Error: Unknown tool '{name}'")
            sys.exit(1)
        sys.exit(0)

    elif cmd == "dump-prompt":
        print(generate_agent_prompt_markdown())
        sys.exit(0)

    elif cmd == "dump-schema":
        print(json.dumps(get_all_tools_schema(), indent=2))
        sys.exit(0)

    elif cmd == "execute":
        if len(args) < 2:
            print("Error: Specify tool name to execute.")
            sys.exit(1)
        tool_name = args[1]
        tool_args = {}

        if len(args) > 2:
            raw_arg = args[2]
            if raw_arg.startswith("{"):
                try:
                    tool_args = json.loads(raw_arg)
                except Exception as e:
                    print(f"Error: Invalid JSON arguments: {e}")
                    sys.exit(1)
            else:
                # Parse --key value pairs
                i = 2
                while i < len(args):
                    token = args[i]
                    if token.startswith("--"):
                        key = token[2:]
                        val = args[i + 1] if i + 1 < len(args) else True
                        try:
                            val = json.loads(val)
                        except Exception:
                            pass
                        tool_args[key] = val
                        i += 2
                    else:
                        i += 1

        # Check if local HTTP server is running on 5000; if so, forward to maintain shared process state
        use_http = False
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:5000/api/detect_hardware",
                headers={"User-Agent": "Z2YY-Agent-CLI"}
            )
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status == 200:
                    use_http = True
        except Exception:
            use_http = False

        if use_http:
            try:
                payload = json.dumps({"tool": tool_name, "arguments": tool_args}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:5000/api/agent/execute",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    output = resp.read().decode("utf-8")
                    print(output)
                    sys.exit(0)
            except urllib.error.HTTPError as he:
                print(he.read().decode("utf-8"))
                sys.exit(1)
            except Exception as ex:
                logging.warning(f"HTTP forward failed ({ex}), falling back to direct in-process execution.")

        # In-process execution fallback
        result = execute_tool(tool_name, tool_args)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("success") else 1)

    else:
        print(f"Error: Unknown agent command '{cmd}'")
        sys.exit(1)


if __name__ == '__main__':
    run_cli(sys.argv[1:])
