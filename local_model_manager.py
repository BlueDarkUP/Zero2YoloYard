import os
import threading
import subprocess
import requests
import time
import logging
from config import BASE_DIR

# 记录当前下载任务状态
# 格式: { model_id: {"status": "downloading"|"ready"|"error", "progress": "...", "message": "..."} }
DOWNLOAD_TASKS = {}

def get_model_registry():
    return [
        # --- SAM 2.1 Series ---
        {
            "id": "sam2_tiny",
            "name": "SAM 2.1 Tiny",
            "engine": "SAM 2.1",
            "path": os.path.join("checkpoints", "sam2.1_t.pt"),
            "ext": ".pt",
            "purpose": "Point / Tracking (Low Latency)",
            "type": "file",
            "download_type": "file",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt"
        },
        {
            "id": "sam2_small",
            "name": "SAM 2.1 Small",
            "engine": "SAM 2.1",
            "path": os.path.join("checkpoints", "sam2.1_s.pt"),
            "ext": ".pt",
            "purpose": "Point / Tracking (Balanced)",
            "type": "file",
            "download_type": "file",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_s.pt"
        },
        {
            "id": "sam2_base",
            "name": "SAM 2.1 Base+",
            "engine": "SAM 2.1",
            "path": os.path.join("checkpoints", "sam2.1_b.pt"),
            "ext": ".pt",
            "purpose": "Point / Tracking (Standard)",
            "type": "file",
            "download_type": "file",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt"
        },
        {
            "id": "sam2_large",
            "name": "SAM 2.1 Large",
            "engine": "SAM 2.1",
            "path": os.path.join("checkpoints", "sam2.1_l.pt"),
            "ext": ".pt",
            "purpose": "Point / Tracking (High Precision)",
            "type": "file",
            "download_type": "file",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_l.pt"
        },
        # --- SAM 3 Series ---
        {
            "id": "sam3_image",
            "name": "SAM 3 (Image)",
            "engine": "SAM 3",
            "path": os.path.join("checkpoints", "sam3", "sam3.pt"),
            "ext": ".pt",
            "purpose": "Open-Vocabulary Retrieval / Smart Select / LAM / Batch Apply",
            "type": "file",
            "download_type": "file",
            "url": "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
        },
        # --- CLIP Series ---
        {
            "id": "clip_b32",
            "name": "CLIP ViT-B/32",
            "engine": "CLIP",
            "path": os.path.join("checkpoints", "clip", "clip-vit-base-patch32"),
            "ext": "Directory",
            "purpose": "Zero-Shot Classification / Consistency Check (Recommended)",
            "type": "dir",
            "download_type": "hf_cli",
            "repo_id": "openai/clip-vit-base-patch32"
        },
        {
            "id": "clip_b16",
            "name": "CLIP ViT-B/16",
            "engine": "CLIP",
            "path": os.path.join("checkpoints", "clip", "clip-vit-base-patch16"),
            "ext": "Directory",
            "purpose": "Zero-Shot Classification / Consistency Check (Higher Precision)",
            "type": "dir",
            "download_type": "hf_cli",
            "repo_id": "openai/clip-vit-base-patch16"
        },
        {
            "id": "clip_l14",
            "name": "CLIP ViT-L/14",
            "engine": "CLIP",
            "path": os.path.join("checkpoints", "clip", "clip-vit-large-patch14"),
            "ext": "Directory",
            "purpose": "Zero-Shot Classification / Consistency Check (Highest Precision)",
            "type": "dir",
            "download_type": "hf_cli",
            "repo_id": "openai/clip-vit-large-patch14"
        },
        # --- GKDT Pose Estimation ---
        {
            "id": "gkdt_l",
            "name": "GKDT-L",
            "engine": "GKDT",
            "path": os.path.join("gkdt_engine", "output", "GKDT-L_for_app", "model", "gkd_fullset.best"),
            "ext": ".best",
            "purpose": "Pose Keypoint Estimation (Lightweight, ~6 GB)",
            "type": "file",
            "rare_format": True,
            "download_type": "hf_cli",
            "repo_id": "changshenglu/GKDT-L_for_App"
        },
        {
            "id": "gkdt_h",
            "name": "GKDT-H",
            "engine": "GKDT",
            "path": os.path.join("gkdt_engine", "output", "GKDT-H_for_app", "model", "gkd_fullset.best"),
            "ext": ".best",
            "purpose": "Pose Keypoint Estimation (High Precision, ~12 GB)",
            "type": "file",
            "rare_format": True,
            "download_type": "hf_cli",
            "repo_id": "changshenglu/GKDT-H_for_App"
        },
        # --- GKDT Object Detector Dependency ---
        {
            "id": "grounding_dino",
            "name": "Grounding DINO",
            "engine": "Object Detector",
            "path": os.path.join("gkdt_engine", "test_real_world", "object_detector_lib", "weights", "groundingdino_swint_ogc.pth"),
            "ext": ".pth",
            "purpose": "GKDT Multi-Object Detection Dependency (Open-Set Text Object Detection)",
            "type": "file",
            "download_type": "file_ghproxy",
            "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        },
        {
            "id": "locate_anything",
            "name": "Locate Anything 3B",
            "engine": "Object Detector",
            "path": os.path.join("gkdt_engine", "test_real_world", "object_detector_lib", "weights", "LocateAnything-3B"),
            "ext": "Directory",
            "purpose": "GKDT Multi-Object Detection Dependency (High-Precision Visual Grounding)",
            "type": "dir",
            "download_type": "hf_cli",
            "repo_id": "nvidia/LocateAnything-3B"
        },
    ]


def _download_file(url, dest_path, model_id):
    """ 流式下载单文件，并报告进度 """
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # 如果启用国内镜像加速 (ghproxy / hf-mirror)
        if DOWNLOAD_TASKS[model_id].get("use_mirror"):
            if url.startswith("https://github.com"):
                url = url.replace("https://github.com", "https://ghproxy.net/https://github.com")
            elif url.startswith("https://huggingface.co"):
                url = url.replace("https://huggingface.co", "https://hf-mirror.com")

        logging.info(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()
        last_update_time = start_time

        with open(dest_path + ".tmp", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_update_time > 0.5:  # 每半秒更新一次状态
                        speed = downloaded / (now - start_time)
                        if total_size > 0:
                            prog_str = f"{downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB - {speed/(1024*1024):.1f}MB/s"
                        else:
                            prog_str = f"{downloaded/(1024*1024):.1f}MB - {speed/(1024*1024):.1f}MB/s"
                        
                        DOWNLOAD_TASKS[model_id]["progress"] = prog_str
                        last_update_time = now

        os.rename(dest_path + ".tmp", dest_path)
        DOWNLOAD_TASKS[model_id]["status"] = "ready"
        DOWNLOAD_TASKS[model_id]["progress"] = "DOWNLOAD COMPLETE"
        logging.info(f"Download complete: {dest_path}")
    except Exception as e:
        logging.error(f"Download failed for {model_id}: {e}")
        DOWNLOAD_TASKS[model_id]["status"] = "error"
        DOWNLOAD_TASKS[model_id]["message"] = str(e)


def _download_hf_cli(repo_id, dest_path, model_id):
    """ 调用 huggingface-cli 下载仓库 """
    try:
        model_def = next(m for m in get_model_registry() if m["id"] == model_id)
        if model_def["type"] == "file":
            local_dir = os.path.dirname(dest_path) 
        else:
            local_dir = dest_path

        os.makedirs(local_dir, exist_ok=True)
        env = os.environ.copy()
        if DOWNLOAD_TASKS[model_id].get("use_mirror"):
            env["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        logging.info(f"Running huggingface-cli to download {repo_id} to {local_dir}")
        DOWNLOAD_TASKS[model_id]["progress"] = "DOWNLOADING (HF CLI)..."

        # 使用 huggingface-cli
        process = subprocess.Popen(
            ["huggingface-cli", "download", repo_id, "--local-dir", local_dir],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            pass

        process.wait()
        if process.returncode == 0:
            DOWNLOAD_TASKS[model_id]["status"] = "ready"
            DOWNLOAD_TASKS[model_id]["progress"] = "DOWNLOAD COMPLETE"
            logging.info(f"HF Download complete for {model_id}")
        else:
            raise RuntimeError(f"huggingface-cli exited with code {process.returncode}")

    except Exception as e:
        logging.error(f"HF Download failed for {model_id}: {e}")
        DOWNLOAD_TASKS[model_id]["status"] = "error"
        DOWNLOAD_TASKS[model_id]["message"] = str(e)


def start_download(model_id, use_mirror=True):
    if model_id in DOWNLOAD_TASKS and DOWNLOAD_TASKS[model_id]["status"] == "downloading":
        return False, "Task already running"

    registry = get_model_registry()
    model_def = next((m for m in registry if m["id"] == model_id), None)
    if not model_def:
        return False, "Model ID not found"

    full_path = os.path.join(BASE_DIR, model_def["path"])
    
    DOWNLOAD_TASKS[model_id] = {
        "status": "downloading",
        "progress": "0%",
        "message": "",
        "use_mirror": use_mirror
    }

    dtype = model_def.get("download_type")
    if dtype in ["file", "file_ghproxy"]:
        t = threading.Thread(target=_download_file, args=(model_def["url"], full_path, model_id))
        t.daemon = True
        t.start()
    elif dtype == "hf_cli":
        t = threading.Thread(target=_download_hf_cli, args=(model_def["repo_id"], full_path, model_id))
        t.daemon = True
        t.start()
    else:
        DOWNLOAD_TASKS[model_id]["status"] = "error"
        DOWNLOAD_TASKS[model_id]["message"] = "Unsupported download type"
        return False, "Unsupported download type"

    return True, "Download started"

def get_download_status(model_id):
    return DOWNLOAD_TASKS.get(model_id, None)
