import os
import sys
import threading
import requests
import urllib3
import time
import logging
from urllib.parse import urlparse
from config import BASE_DIR

# 禁用未验证 HTTPS 请求的警告信息，避免控制台刷屏
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 记录当前下载任务状态
# 格式: { model_id: {"status": "downloading"|"ready"|"error", "progress": "...", "percent": 0, "speed_mb": 0, "message": "...", "use_mirror": True} }
DOWNLOAD_TASKS = {}

# 常用高质量 GitHub 镜像前缀（按测速及稳定性排序）
GITHUB_MIRRORS = [
    "https://ghfast.top/",
    "https://ghproxy.cn/",
    "https://ghproxy.net/",
]

# HuggingFace 官方与国内镜像域名
HF_OFFICIAL = "https://huggingface.co"
HF_MIRROR = "https://hf-mirror.com"




def get_model_registry():
    """
    获取全局模型注册表。
    包含 SAM 2.1、SAM 3、CLIP、GKDT 全系姿态估计模型以及开集检测器依赖（Grounding DINO、LocateAnything-3B）。
    严格对照论文与官方 ECCV 2026 GKDT 开源规范定义。
    """
    return [
        # --- SAM 2.1 Series ---
        {
            "id": "sam2_tiny",
            "name": "SAM 2.1 Tiny",
            "engine": "SAM 2.1",
            "path": os.path.join("checkpoints", "sam2.1_t.pt"),
            "ext": ".pt",
            "purpose": "Point / Tracking (Low Latency, ~75 MB)",
            "type": "file",
            "min_size_mb": 70,
            "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt"
        },
        {
            "id": "sam2_small",
            "name": "SAM 2.1 Small",
            "engine": "SAM 2.1",
            "path": os.path.join("checkpoints", "sam2.1_s.pt"),
            "ext": ".pt",
            "purpose": "Point / Tracking (Balanced, ~88 MB)",
            "type": "file",
            "min_size_mb": 80,
            "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_s.pt"
        },
        {
            "id": "sam2_base",
            "name": "SAM 2.1 Base+",
            "engine": "SAM 2.1",
            "path": os.path.join("checkpoints", "sam2.1_b.pt"),
            "ext": ".pt",
            "purpose": "Point / Tracking (Standard, ~154 MB)",
            "type": "file",
            "min_size_mb": 140,
            "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt"
        },
        {
            "id": "sam2_large",
            "name": "SAM 2.1 Large",
            "engine": "SAM 2.1",
            "path": os.path.join("checkpoints", "sam2.1_l.pt"),
            "ext": ".pt",
            "purpose": "Point / Tracking (High Precision, ~428 MB)",
            "type": "file",
            "min_size_mb": 400,
            "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_l.pt"
        },
        # --- SAM 3 Series ---
        {
            "id": "sam3_image",
            "name": "SAM 3 (Image)",
            "engine": "SAM 3",
            "path": os.path.join("checkpoints", "sam3", "sam3.pt"),
            "ext": ".pt",
            "purpose": "Open-Vocabulary Retrieval / Smart Select / TrueLAM (~3.3 GB)",
            "type": "file",
            "min_size_mb": 3000,
            "url": "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
        },
        # --- CLIP Series ---
        {
            "id": "clip_b32",
            "name": "CLIP ViT-B/32",
            "engine": "CLIP",
            "path": os.path.join("checkpoints", "clip", "clip-vit-base-patch32"),
            "ext": "Directory",
            "purpose": "Zero-Shot Classification / Consistency Check (Recommended, ~605 MB)",
            "type": "dir",
            "repo_id": "openai/clip-vit-base-patch32",
            "min_size_mb": 500,
            "check_files": ["config.json"]
        },
        {
            "id": "clip_b16",
            "name": "CLIP ViT-B/16",
            "engine": "CLIP",
            "path": os.path.join("checkpoints", "clip", "clip-vit-base-patch16"),
            "ext": "Directory",
            "purpose": "Zero-Shot Classification / Consistency Check (Higher Precision, ~599 MB)",
            "type": "dir",
            "repo_id": "openai/clip-vit-base-patch16",
            "min_size_mb": 500,
            "check_files": ["config.json"]
        },
        {
            "id": "clip_l14",
            "name": "CLIP ViT-L/14",
            "engine": "CLIP",
            "path": os.path.join("checkpoints", "clip", "clip-vit-large-patch14"),
            "ext": "Directory",
            "purpose": "Zero-Shot Classification / Consistency Check (Highest Precision, ~1.7 GB)",
            "type": "dir",
            "repo_id": "openai/clip-vit-large-patch14",
            "min_size_mb": 1500,
            "check_files": ["config.json"]
        },
        # --- GKDT Pose Estimation (ECCV 2026 Foundation Models) ---
        {
            "id": "gkdt_l",
            "name": "GKDT-L (App)",
            "engine": "GKDT",
            "path": os.path.join("gkdt_engine", "output", "GKDT-L_for_app", "model", "gkd_fullset.best"),
            "ext": ".best",
            "purpose": "General Keypoint Detection (Application Default / DINOv3-L, ~6.3 GB)",
            "type": "file",
            "rare_format": True,
            "min_size_mb": 5800,
            "url": "https://huggingface.co/changshenglu/GKDT-L_for_App/resolve/main/gkd_fullset.best"
        },
        {
            "id": "gkdt_h",
            "name": "GKDT-H (App)",
            "engine": "GKDT",
            "path": os.path.join("gkdt_engine", "output", "GKDT-H_for_app", "model", "gkd_fullset.best"),
            "ext": ".best",
            "purpose": "General Keypoint Detection (High Precision / DINOv3-H, ~12.9 GB)",
            "type": "file",
            "rare_format": True,
            "min_size_mb": 12000,
            "url": "https://huggingface.co/changshenglu/GKDT-H_for_App/resolve/main/gkd_fullset.best"
        },
        {
            "id": "gkdt_l_research",
            "name": "GKDT-L (Research)",
            "engine": "GKDT",
            "path": os.path.join("gkdt_engine", "output", "GKDT-L_for_research", "model", "gkd.best"),
            "ext": ".best",
            "purpose": "General Keypoint Detection (Research Benchmark / DINOv3-L, ~6.3 GB)",
            "type": "file",
            "rare_format": True,
            "min_size_mb": 5800,
            "url": "https://huggingface.co/changshenglu/GKDT-L_for_Research/resolve/main/gkd.best"
        },
        {
            "id": "gkdt_h_research",
            "name": "GKDT-H (Research)",
            "engine": "GKDT",
            "path": os.path.join("gkdt_engine", "output", "GKDT-H_for_research", "model", "gkd.best"),
            "ext": ".best",
            "purpose": "General Keypoint Detection (Research Benchmark / DINOv3-H, ~12.9 GB)",
            "type": "file",
            "rare_format": True,
            "min_size_mb": 12000,
            "url": "https://huggingface.co/changshenglu/GKDT-H_for_Research/resolve/main/gkd.best"
        },
        # --- GKDT Multi-Object Detection Dependencies ---
        {
            "id": "grounding_dino",
            "name": "Grounding DINO",
            "engine": "Object Detector",
            "path": os.path.join("gkdt_engine", "test_real_world", "object_detector_lib", "weights", "groundingdino_swint_ogc.pth"),
            "ext": ".pth",
            "purpose": "GKDT Multi-Object Dependency: Open-Set Text Detector (~694 MB)",
            "type": "file",
            "min_size_mb": 600,
            "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        },
        {
            "id": "locate_anything",
            "name": "Locate Anything 3B",
            "engine": "Object Detector",
            "path": os.path.join("gkdt_engine", "test_real_world", "object_detector_lib", "weights", "LocateAnything-3B"),
            "ext": "Directory",
            "purpose": "GKDT Multi-Object Dependency: High-Precision Visual Grounding (~7.6 GB)",
            "type": "dir",
            "repo_id": "nvidia/LocateAnything-3B",
            "min_size_mb": 7000,
            "check_files": [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        },
    ]


def check_model_presence(model_def, base_dir=None):
    """
    严格检查模型权重是否存在。
    解决了刚 clone 下仓库时由于 git 跟踪了代码/配置文件导致误报 READY 的严重 BUG。
    返回: (is_present: bool, size_mb: float)
    """
    if base_dir is None:
        base_dir = BASE_DIR

    if isinstance(model_def, str):
        model_id = model_def
        model_def = next((m for m in get_model_registry() if m["id"] == model_id), None)
        if not model_def:
            return False, 0.0

    rel_path = model_def["path"]
    full_path = os.path.join(base_dir, rel_path)
    min_size_mb = model_def.get("min_size_mb", 1)

    if model_def["type"] == "file":
        if not os.path.isfile(full_path):
            # 检查是否有下载中的临时文件
            tmp_path = full_path + ".tmp"
            if os.path.isfile(tmp_path):
                return False, round(os.path.getsize(tmp_path) / (1024 * 1024), 1)
            return False, 0.0

        actual_bytes = os.path.getsize(full_path)
        actual_mb = round(actual_bytes / (1024 * 1024), 1)
        if actual_bytes < min_size_mb * 1024 * 1024:
            # 文件存在但小于预期阈值（例如下载中断残留的残损文件）
            return False, actual_mb
        return True, actual_mb

    elif model_def["type"] == "dir":
        if not os.path.isdir(full_path):
            return False, 0.0

        # 计算目录现有总大小
        total_bytes = 0
        for root_d, _, files in os.walk(full_path):
            for f in files:
                try:
                    total_bytes += os.path.getsize(os.path.join(root_d, f))
                except OSError:
                    pass
        total_mb = round(total_bytes / (1024 * 1024), 1)

        # 核心逻辑：检查关键权重文件（如 safetensors）
        check_files = model_def.get("check_files", [])
        if model_def["id"] == "locate_anything":
            # LocateAnything-3B 必须完整具有分卷权重或单卷权重
            f1 = os.path.join(full_path, "model-00001-of-00002.safetensors")
            f2 = os.path.join(full_path, "model-00002-of-00002.safetensors")
            f_single = os.path.join(full_path, "model.safetensors")
            has_split = (os.path.isfile(f1) and os.path.getsize(f1) > 1024 * 1024 * 1024 and
                         os.path.isfile(f2) and os.path.getsize(f2) > 1024 * 1024 * 1024)
            has_single = os.path.isfile(f_single) and os.path.getsize(f_single) > 4000 * 1024 * 1024

            if not (has_split or has_single) or total_mb < min_size_mb:
                return False, total_mb
            return True, total_mb

        elif model_def["id"].startswith("clip_"):
            # CLIP 必须含有 config.json 以及至少一个真实的权重文件 (model.safetensors 或 pytorch_model.bin > 100MB)
            cfg = os.path.join(full_path, "config.json")
            sf = os.path.join(full_path, "model.safetensors")
            bin_f = os.path.join(full_path, "pytorch_model.bin")
            has_cfg = os.path.isfile(cfg)
            has_weights = ((os.path.isfile(sf) and os.path.getsize(sf) > 100 * 1024 * 1024) or
                           (os.path.isfile(bin_f) and os.path.getsize(bin_f) > 100 * 1024 * 1024))
            if not (has_cfg and has_weights) or total_mb < min_size_mb:
                return False, total_mb
            return True, total_mb

        # 通用检查
        for req in check_files:
            target_req = os.path.join(full_path, req)
            if not os.path.isfile(target_req) or os.path.getsize(target_req) == 0:
                return False, total_mb

        if total_mb < min_size_mb:
            return False, total_mb

        return True, total_mb

    return False, 0.0


def _build_url_candidates(base_url, use_mirror=True):
    """
    根据启用镜像开关生成按优先级排序的下载候选 URL 列表。
    无论开关开还是关，均包含主地址与备选镜像，并提供自动重试降级保护：
    - 开镜像：先试高质量国内镜像节点，失败或超速限制时自动回退官方源；
    - 关镜像：先连官方源，因国内断网/墙失败时自动降级尝试镜像节点，绝不抛弃用户。
    """
    candidates = []
    if base_url.startswith("https://github.com"):
        mirror_urls = [m + base_url for m in GITHUB_MIRRORS]
        if use_mirror:
            candidates = mirror_urls + [base_url]
        else:
            candidates = [base_url] + mirror_urls
    elif base_url.startswith("https://huggingface.co"):
        mirror_url = base_url.replace("https://huggingface.co", HF_MIRROR)
        if use_mirror:
            candidates = [mirror_url, base_url]
        else:
            candidates = [base_url, mirror_url]
    else:
        candidates = [base_url]

    return candidates


def _download_stream_with_resume(candidate_urls, dest_path, model_id, min_size_mb=1,
                                 progress_prefix="", overall_callback=None):
    """
    核心流式断点续传下载器。
    特性：
    1. 支持 HTTP Range 范围请求，中断后重新下载自动从已下载字节追加，不浪费任何流量与时间；
    2. 多节点自动故障转移：某一镜像节点失败/断连/403/502，立即无损切换到下一个节点断点续传；
    3. 支持 verify=False 与 Session 会话复用，彻底规避 Windows 代理/梯子环境下的 SSL 证书校验失败；
    4. 实时计算精准下载速度（MB/s）、已下载量、总大小、百分比及剩余时间（ETA）；
    5. 先写入 .tmp 临时文件，校验成功后原子替换，防止生成残损文件。
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".tmp"
    min_bytes = max(1, int(min_size_mb * 1024 * 1024))

    session = requests.Session()
    # 彻底关闭 SSL 证书校验，确保代理或国内镜像跳转时不因根证书缺失而阻断模型权重下载
    session.verify = False
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    max_rounds = 2
    last_error = None

    for round_idx in range(max_rounds):
        for url in candidate_urls:
            logging.info(f"[{model_id}] Trying download (round {round_idx+1}/{max_rounds}) from: {url}")
            response = None
            try:
                existing_bytes = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
                headers = {}
                if existing_bytes > 0:
                    headers["Range"] = f"bytes={existing_bytes}-"

                # 连接超时 10 秒，读取流超时 35 秒
                response = session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(10, 35))

                if response.status_code == 416:
                    response.close()
                    # 范围请求超出（可能已有完整文件或服务器未支持）
                    if existing_bytes >= min_bytes:
                        os.replace(tmp_path, dest_path)
                        logging.info(f"[{model_id}] File already fully downloaded: {dest_path}")
                        return True
                    else:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                        existing_bytes = 0
                        headers.pop("Range", None)
                        response = session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(10, 35))

                response.raise_for_status()

                if response.status_code == 206:
                    # 支持断点续传
                    cr = response.headers.get("Content-Range", "")
                    if "/" in cr:
                        total_size = int(cr.split("/")[-1])
                    else:
                        total_size = existing_bytes + int(response.headers.get("Content-Length", 0))
                    file_mode = "ab"
                else:
                    # 不支持断点续传或从头请求
                    existing_bytes = 0
                    total_size = int(response.headers.get("Content-Length", 0))
                    file_mode = "wb"

                downloaded_this_session = 0
                start_time = time.time()
                last_report_time = start_time

                with open(tmp_path, file_mode) as f:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded_this_session += len(chunk)
                        now = time.time()
                        if now - last_report_time >= 0.35:
                            total_downloaded = existing_bytes + downloaded_this_session
                            elapsed = max(now - start_time, 0.001)
                            speed = downloaded_this_session / elapsed

                            if total_size > 0:
                                percent = min(99.9, round(total_downloaded / total_size * 100, 1))
                                rem_bytes = max(0, total_size - total_downloaded)
                                eta_sec = int(rem_bytes / max(speed, 1))
                                eta_str = f" ETA: {eta_sec // 60}m{eta_sec % 60:02d}s" if eta_sec < 7200 else ""
                                prog_str = f"{progress_prefix}{total_downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB ({percent}%){eta_str} - {speed/(1024*1024):.1f}MB/s"
                            else:
                                percent = 50.0
                                prog_str = f"{progress_prefix}{total_downloaded/(1024*1024):.1f}MB - {speed/(1024*1024):.1f}MB/s"

                            if overall_callback:
                                overall_callback(total_downloaded, total_size, speed)
                            else:
                                DOWNLOAD_TASKS[model_id].update({
                                    "status": "downloading",
                                    "progress": prog_str,
                                    "percent": percent,
                                    "speed_mb": round(speed / (1024 * 1024), 2),
                                    "downloaded_mb": round(total_downloaded / (1024 * 1024), 1),
                                    "total_mb": round(total_size / (1024 * 1024), 1) if total_size > 0 else 0.0,
                                })
                            last_report_time = now

                # 下载完成检验
                final_size = os.path.getsize(tmp_path)
                if final_size >= min_bytes:
                    os.replace(tmp_path, dest_path)
                    logging.info(f"[{model_id}] Successfully downloaded and saved to: {dest_path}")
                    return True
                else:
                    raise RuntimeError(f"Downloaded file size ({final_size} bytes) below threshold ({min_bytes} bytes).")

            except Exception as e:
                logging.warning(f"[{model_id}] Download attempt failed for {url}: {e}")
                last_error = e
                time.sleep(1)
                continue
            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass

    raise RuntimeError(f"All candidate mirrors failed. Last error: {last_error}")



def _download_locate_anything_weights(model_def, dest_dir, model_id, use_mirror=True):
    """
    专门针对 LocateAnything-3B 的高可靠性下载逻辑。
    LocateAnything 的代码、模型架构与文本配置已随 git 仓库内置。
    此处精准下载所需的两个 safetensors 大权重分卷及核心索引，
    完全绕过 snapshot_download 无法断点续传、不显示百分比与无用媒体视频文件的缺点。
    """
    os.makedirs(dest_dir, exist_ok=True)

    weight_files = [
        {"name": "model-00001-of-00002.safetensors", "min_mb": 4500, "approx_mb": 4730},
        {"name": "model-00002-of-00002.safetensors", "min_mb": 2500, "approx_mb": 2576}
    ]
    total_approx_mb = 7306.0

    # 1. 确保必要的 json 索引配置存在
    aux_files = ["model.safetensors.index.json", "config.json"]
    for af in aux_files:
        af_path = os.path.join(dest_dir, af)
        if not os.path.isfile(af_path) or os.path.getsize(af_path) == 0:
            direct_u = f"https://huggingface.co/nvidia/LocateAnything-3B/resolve/main/{af}"
            urls = _build_url_candidates(direct_u, use_mirror=use_mirror)
            try:
                _download_stream_with_resume(urls, af_path, model_id, min_size_mb=0, progress_prefix=f"Config [{af}]: ")
            except Exception as e:
                logging.warning(f"Failed to fetch auxiliary file {af}: {e}")

    # 2. 依次断点续传下载两大分卷权重
    for idx, wf in enumerate(weight_files, 1):
        file_name = wf["name"]
        file_path = os.path.join(dest_dir, file_name)

        if os.path.isfile(file_path) and os.path.getsize(file_path) >= wf["min_mb"] * 1024 * 1024:
            logging.info(f"[{model_id}] {file_name} already present and valid.")
            continue

        direct_url = f"https://huggingface.co/nvidia/LocateAnything-3B/resolve/main/{file_name}"
        candidate_urls = _build_url_candidates(direct_url, use_mirror=use_mirror)

        prev_done_mb = (weight_files[0]["approx_mb"] if idx == 2 else 0.0)

        def make_overall_callback(file_idx=idx, base_done_mb=prev_done_mb):
            def cb(curr_downloaded, curr_total, speed):
                overall_downloaded_mb = base_done_mb + (curr_downloaded / (1024 * 1024))
                overall_percent = min(99.9, round(overall_downloaded_mb / total_approx_mb * 100, 1))
                rem_mb = max(0, total_approx_mb - overall_downloaded_mb)
                speed_mb = speed / (1024 * 1024)
                eta_sec = int((rem_mb / max(speed_mb, 0.001))) if speed_mb > 0.05 else 0
                eta_str = f" ETA: {eta_sec // 60}m{eta_sec % 60:02d}s" if 0 < eta_sec < 7200 else ""
                prog = f"[{file_idx}/2] {file_name[:12]}..: {curr_downloaded/(1024*1024):.1f}MB/{curr_total/(1024*1024):.1f}MB (Total: {overall_percent}%){eta_str} - {speed_mb:.1f}MB/s"
                DOWNLOAD_TASKS[model_id].update({
                    "status": "downloading",
                    "progress": prog,
                    "percent": overall_percent,
                    "speed_mb": round(speed_mb, 2),
                    "downloaded_mb": round(overall_downloaded_mb, 1),
                    "total_mb": total_approx_mb,
                })
            return cb

        _download_stream_with_resume(
            candidate_urls, file_path, model_id,
            min_size_mb=wf["min_mb"],
            progress_prefix=f"[{idx}/2] ",
            overall_callback=make_overall_callback()
        )

    DOWNLOAD_TASKS[model_id].update({
        "status": "ready",
        "progress": "DOWNLOAD COMPLETE",
        "percent": 100,
        "message": ""
    })
    logging.info(f"[{model_id}] LocateAnything-3B weights download complete.")


def _download_clip_model(model_def, dest_dir, model_id, use_mirror=True):
    """
    专门针对 CLIP 系列模型的轻量精炼下载。
    仅下载 transformers 推理必备的 model.safetensors 及文本配置，
    避开官方 repo 中重复捆绑的无用 Flax 与 TF 格式权重（立省 3 倍多达 5GB 空间与下载时间）。
    """
    os.makedirs(dest_dir, exist_ok=True)
    repo_id = model_def["repo_id"]
    # openai/clip-vit-large-patch14 具备 model.safetensors；base-patch32 / base-patch16 仅提供 pytorch_model.bin
    weight_name = "model.safetensors" if repo_id == "openai/clip-vit-large-patch14" else "pytorch_model.bin"

    essential_files = [
        {"name": "config.json", "min_mb": 0},
        {"name": "preprocessor_config.json", "min_mb": 0},
        {"name": "tokenizer_config.json", "min_mb": 0},
        {"name": "special_tokens_map.json", "min_mb": 0},
        {"name": "vocab.json", "min_mb": 0},
        {"name": "merges.txt", "min_mb": 0},
        {"name": weight_name, "min_mb": model_def.get("min_size_mb", 400)},
    ]

    for item in essential_files:
        fn = item["name"]
        fp = os.path.join(dest_dir, fn)
        if os.path.isfile(fp) and (item["min_mb"] == 0 or os.path.getsize(fp) >= item["min_mb"] * 1024 * 1024):
            continue

        direct_u = f"https://huggingface.co/{repo_id}/resolve/main/{fn}"
        candidate_urls = _build_url_candidates(direct_u, use_mirror=use_mirror)
        _download_stream_with_resume(
            candidate_urls, fp, model_id,
            min_size_mb=item["min_mb"],
            progress_prefix=f"[{fn}]: "
        )

    DOWNLOAD_TASKS[model_id].update({
        "status": "ready",
        "progress": "DOWNLOAD COMPLETE",
        "percent": 100,
        "message": ""
    })
    logging.info(f"[{model_id}] CLIP model download complete.")


def _task_worker(model_def, full_path, model_id, use_mirror):
    """通用后台下载工作线程"""
    try:
        if model_def["id"] == "locate_anything":
            _download_locate_anything_weights(model_def, full_path, model_id, use_mirror=use_mirror)
        elif model_def["id"].startswith("clip_"):
            _download_clip_model(model_def, full_path, model_id, use_mirror=use_mirror)
        else:
            # 单文件直接断点续传（SAM 2.1、SAM 3、GKDT-L/H、Grounding DINO 等）
            base_url = model_def["url"]
            candidate_urls = _build_url_candidates(base_url, use_mirror=use_mirror)
            _download_stream_with_resume(
                candidate_urls, full_path, model_id,
                min_size_mb=model_def.get("min_size_mb", 1)
            )
            DOWNLOAD_TASKS[model_id].update({
                "status": "ready",
                "progress": "DOWNLOAD COMPLETE",
                "percent": 100,
                "message": ""
            })
    except Exception as e:
        logging.error(f"[ModelManager] Download failed for {model_id}: {e}", exc_info=True)
        DOWNLOAD_TASKS[model_id].update({
            "status": "error",
            "message": str(e),
            "progress": "ERROR"
        })


def start_download(model_id, use_mirror=True):
    """
    触发下载任务。
    如果此前已有任务处于下载中，拒绝重复触发；若是失败任务则允许重新重试续传。
    """
    if model_id in DOWNLOAD_TASKS and DOWNLOAD_TASKS[model_id].get("status") == "downloading":
        return False, "Task already running"

    registry = get_model_registry()
    model_def = next((m for m in registry if m["id"] == model_id), None)
    if not model_def:
        return False, f"Model ID '{model_id}' not found in registry"

    full_path = os.path.join(BASE_DIR, model_def["path"])

    DOWNLOAD_TASKS[model_id] = {
        "status": "downloading",
        "progress": "STARTING...",
        "percent": 0,
        "speed_mb": 0.0,
        "downloaded_mb": 0.0,
        "total_mb": 0.0,
        "message": "",
        "use_mirror": use_mirror
    }

    t = threading.Thread(
        target=_task_worker,
        args=(model_def, full_path, model_id, use_mirror),
        name=f"DLWorker-{model_id}"
    )
    t.daemon = True
    t.start()

    return True, "Download started"


def get_download_status(model_id):
    """获取指定模型的下载状态字典"""
    return DOWNLOAD_TASKS.get(model_id, None)
