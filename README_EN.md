# Zero2YoloYard: Intelligent Vision Dataset Annotation & Data Engineering Platform

<p align="center">
  <strong>English Version</strong> | <a href="README.md"><strong>中文 Version</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-Waitress-000000.svg" alt="Flask Waitress">
  <img src="https://img.shields.io/badge/SAM-2.1%20%7C%20SAM%203-green.svg" alt="SAM Models">
  <img src="https://img.shields.io/badge/GKDT-ECCV%202026-orange.svg" alt="GKDT Model">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg" alt="Platform">
</p>

> **Developed by BlueDarkUP from FIRST Tech Challenge team 27570**  
> *Based on -- FIRST Machine Learning Toolchain --*

**Zero2YoloYard** is a modern, end-to-end, multi-foundation-model-driven platform for intelligent vision dataset annotation, data augmentation, data quality diagnostics, and multi-format dataset export.

Built with a high-performance Python (Flask + Waitress) backend and packaged as a native desktop application using `pywebview`, the platform deeply integrates **SAM 2.1 / SAM 3 (Segment Anything Model)**, **CLIP (Contrastive Language-Image Pretraining)**, **GKDT (Grounded Keypoint Detection Transformer)** alongside its underlying `gkdt_engine` neural network algorithms, and the **Albumentations** image transformation pipeline. It empowers four core computer vision tasks: **Object Detection**, **Instance/Semantic Segmentation**, **Keypoint/Pose Estimation**, and **Image Classification/Clustering**.

---

## Table of Contents

1. [Acknowledgements & Open-Source Attribution](#1-acknowledgements--open-source-attribution)
   - [1.6 Checkpoints Directory Layout & Precise Download Commands](#16-checkpoints-directory-layout--precise-download-commands)
2. [Project Architecture & Design Philosophy](#2-project-architecture--design-philosophy)
3. [Underlying Algorithms & Neural Network Principles](#3-underlying-algorithms--neural-network-principles)
   - [3.1 GKDT (General Keypoint Detection Transformer) Core Engine](#31-gkdt-general-keypoint-detection-transformer-core-engine)
   - [3.2 SAM3 Open-Vocabulary & Bounding Box IoU Matching](#32-sam3-open-vocabulary--bounding-box-iou-matching)
   - [3.3 SAM2 Video Memory Mechanism & LRU Frame State Cache](#33-sam2-video-memory-mechanism--lru-frame-state-cache)
   - [3.4 CLIP Zero-Shot Classification & 8-Template Prompt Ensembling](#34-clip-zero-shot-classification--8-template-prompt-ensembling)
   - [3.5 Deep Semantic & HSV Dual-Stream Data Audit Model](#35-deep-semantic--hsv-dual-stream-data-audit-model)
   - [3.6 Cross-Frame BBox & Keypoint Linear Interpolation](#36-cross-frame-bbox--keypoint-linear-interpolation)
   - [3.7 4-in-1 Mosaic Augmentation & YOLO Coordinate Clamping](#37-4-in-1-mosaic-augmentation--yolo-coordinate-clamping)
4. [Full Directory & Subfolder Architecture Panorama](#4-full-directory--subfolder-architecture-panorama)
   - [4.1 `gkdt_engine/` Subsystem](#41-gkdt_engine-subsystem)
   - [4.2 `sam2/` & `sam3/` Model Predictors & Architectures](#42-sam2--sam3-model-predictors--architectures)
   - [4.3 Python Backend Core Logic & Scheduling Modules](#43-python-backend-core-logic--scheduling-modules)
   - [4.4 `exporters/` Multi-Format Dataset Export Engines](#44-exporters-multi-format-dataset-export-engines)
   - [4.5 `templates/` Frontend Templates & Inference Controls](#45-templates-frontend-templates--inference-controls)
   - [4.6 `static/js/annotation/` Canvas Annotation Engines](#46-staticjsannotation-canvas-annotation-engines)
5. [Database Schema & Entity-Relationship Diagram (SQLite ERD)](#5-database-schema--entity-relationship-diagram-sqlite-erd)
6. [Complete RESTful API & SSE Real-time Stream Manual](#6-complete-restful-api--sse-real-time-stream-manual)
7. [Data Augmentation & Real-Time Preview Workflow](#7-data-augmentation--real-time-preview-workflow)
8. [System Configuration (`settings.json`)](#8-system-configuration-settingsjson)
9. [Environment Setup & Native Desktop Packaging](#9-environment-setup--native-desktop-packaging)

---

## 1. Acknowledgements & Open-Source Attribution

The birth and evolution of Zero2YoloYard stand on the shoulders of the academic community and open-source contributions. We express our highest gratitude to the following foundation models, frameworks, and datasets:

### 1.1 Foundation AI Models & Academic Research

- **GKDT (General Keypoint Detection Transformer) & MegaKPT Dataset**  
  Special acknowledgements to **AlanLuSun** and his research team for their paper at ECCV 2026:
  - **Repository**: [AlanLuSun/General-Keypoint-Detection](https://github.com/AlanLuSun/General-Keypoint-Detection)
  - **Contribution**: Proposed the powerful General Keypoint Detection Transformer (GKDT) model and **MegaKPT**, a unified high-quality dataset combining 29 public benchmarks with over 1.3 million instances. Our internal `gkdt_engine` is built directly upon this research, enabling open-vocabulary pose estimation across diverse object categories.
- **SAM 2.1 & SAM 3 (Segment Anything Model 2 & 3)**  
  Thanks to **Meta AI Research** for open-sourcing the revolutionary foundation segmentation models:
  - **Repository**: [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
  - **Contribution**: Provides state-of-the-art single-frame mask generation and Memory Bank-based cross-frame spatio-temporal tracking in video sequences.
- **CLIP (Contrastive Language-Image Pretraining)**  
  Thanks to the **OpenAI** team and the **HuggingFace** Transformers community:
  - **Repository**: [openai/CLIP](https://github.com/openai/CLIP) / [huggingface/transformers](https://github.com/huggingface/transformers)
  - **Contribution**: Provides cross-modal joint text-visual embedding spaces for zero-shot image classification and unsupervised K-Means feature clustering.
- **DINOv3 (Self-Supervised Vision Transformer)**  
  Thanks to **Meta AI** for the DINOv3 self-supervised vision backbone, providing rich visual representations for GKDT and semantic feature retrieval.
- **Ultralytics YOLO Ecosystem**  
  Thanks to the **Ultralytics** team for standardizing object detection, instance segmentation, and pose estimation formats across YOLOv5 / YOLOv8 / YOLOv11.

### 1.2 Deep Learning & Computer Vision Libraries

- **PyTorch & Torchvision**: Core infrastructure for GPU tensor computing and neural network automatic differentiation.
- **OpenCV (Open Source Computer Vision Library)**: High-performance RGB/HSV color space conversions, histogram calculations, image slicing, and real-time canvas rendering.
- **Albumentations**: [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations) provides industrial-grade image augmentation pipelines with coordinate alignment for keypoints and polygon vertices.
- **NumPy & Pillow (PIL)**: Fundamental numerical computation and image manipulation.
- **scikit-learn**: Efficient K-Means clustering and statistical evaluation.

### 1.3 Backend & Native Desktop Engineering

- **Flask**: Agile RESTful web routing and controller layer.
- **Waitress**: Production-ready, multi-threaded WSGI web server.
- **pywebview**: Python bindings for rendering native webview windows, providing a seamless desktop GUI experience.
- **SQLAlchemy & SQLite3**: Reliable ORM layer and embedded relational database with Write-Ahead Logging (WAL) enabled.
- **PyInstaller**: Toolchain for compiling Python applications into standalone executable binaries for Windows/Linux.

### 1.4 Frontend Engineering & Design

- **Bootstrap 4 & Bootstrap Icons**: Modern glassmorphic layout components and icon set.
- **jQuery**: Client-side DOM manipulation and event handling core.
- **Google Fonts (Inter & JetBrains Mono)**: Minimalist typography for UI and code formatting.

### 1.5 Heritage

- Originating from early paradigms of the **FIRST Machine Learning Toolchain**, heavily refactored and extended by **BlueDarkUP from FIRST Tech Challenge Team 27570**.

### 1.6 Checkpoints Directory Layout & Precise Download Commands

Large binary model checkpoints are decoupled from this Git repository. Place pretrained weights into the designated subdirectories under `checkpoints/` and `gkdt_engine/`:

#### 1.6.1 Standard `checkpoints/` Directory Tree Layout

```text
checkpoints/
├── clip/
│   ├── clip-vit-base-patch16/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── clip-vit-base-patch32/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   └── clip-vit-large-patch14/
│       └── ...
├── sam3/
│   └── sam3.pt
├── sam2.1_t.pt
├── sam2.1_s.pt
├── sam2.1_b.pt
└── sam2.1_l.pt
```

Subsystem model weights under `gkdt_engine/`:

```text
gkdt_engine/
├── output/
│   ├── GKDT-L_for_app/model/
│   └── GKDT-H_for_app/model/
└── test_real_world/
    └── object_detector_lib/
        └── weights/
            ├── groundingdino_swint_ogc.pth
            └── LocateAnything-3B/
```

#### 1.6.2 Precise Terminal Commands

```bash
# Install HuggingFace Hub CLI
pip install -U huggingface_hub

# 1. Download SAM 2.1 Checkpoints (Placed directly under checkpoints/)
hf download facebook/sam2.1-hiera-tiny sam2.1_hiera_tiny.pt --local-dir checkpoints --local-dir-use-symlinks False
mv checkpoints/sam2.1_hiera_tiny.pt checkpoints/sam2.1_t.pt

hf download facebook/sam2.1-hiera-small sam2.1_hiera_small.pt --local-dir checkpoints --local-dir-use-symlinks False
mv checkpoints/sam2.1_hiera_small.pt checkpoints/sam2.1_s.pt

hf download facebook/sam2.1-hiera-base-plus sam2.1_hiera_base_plus.pt --local-dir checkpoints --local-dir-use-symlinks False
mv checkpoints/sam2.1_hiera_base_plus.pt checkpoints/sam2.1_b.pt

hf download facebook/sam2.1-hiera-large sam2.1_hiera_large.pt --local-dir checkpoints --local-dir-use-symlinks False
mv checkpoints/sam2.1_hiera_large.pt checkpoints/sam2.1_l.pt

# 2. Download SAM 3 Checkpoint (Placed under checkpoints/sam3/sam3.pt)
hf download facebook/sam3 --local-dir checkpoints/sam3

# 3. Download CLIP Ensembling Models (Placed under checkpoints/clip/)
hf download openai/clip-vit-base-patch16 --local-dir checkpoints/clip/clip-vit-base-patch16
hf download openai/clip-vit-base-patch32 --local-dir checkpoints/clip/clip-vit-base-patch32
hf download openai/clip-vit-large-patch14 --local-dir checkpoints/clip/clip-vit-large-patch14

# 4. Download GKDT Pose Models (Placed under gkdt_engine/output/)
cd gkdt_engine
hf download changshenglu/GKDT-L_for_App --local-dir output/GKDT-L_for_app/model
hf download changshenglu/GKDT-H_for_App --local-dir output/GKDT-H_for_app/model

# 5. Download GroundingDINO & LocateAnything Models
curl.exe -L "https://ghproxy.net/https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -o "test_real_world/object_detector_lib/weights/groundingdino_swint_ogc.pth"
hf download nvidia/LocateAnything-3B --local-dir test_real_world/object_detector_lib/weights/LocateAnything-3B
cd ..
```

---

## 2. Project Architecture & Design Philosophy

- **Hybrid Native Desktop & Web Architecture**: The backend runs as a Flask app served by Waitress (`host=127.0.0.1, port=5000`) and rendered inside a native 1920x1080 desktop window via `pywebview`, combining Web UI flexibility with native desktop performance.
- **Zero-Shot Open-Vocabulary Annotation**: Eliminates the need to train custom dataset models. Users can input natural language text prompts or category names, leveraging SAM3 and GKDT to automatically generate bounding boxes, polygon masks, and 17-point pose skeletons.
- **Asynchronous Long-Running Background Scheduler**: Frame extraction, SAM2 video tracking, batch labeling, and dataset exports run on dedicated asynchronous background threads with millisecond-level progress streaming via Server-Sent Events (SSE).
- **Dependency-Free Frontend Canvas Engines**: Built with vanilla ES6 classes (`AnnotationCore`, `AnnotatorDetection`, `AnnotatorSegmentation`, `AnnotatorPose`, `AnnotatorClassification`) utilizing HTML5 Canvas 2D Matrix Transforms for zoom and pan interactions.

---

## 3. Underlying Algorithms & Neural Network Principles

### 3.1 GKDT (General Keypoint Detection Transformer) Core Engine
Located inside `gkdt_engine/`, the end-to-end keypoint detection pipeline operates as follows:

1. **DINOv3 Visual & Text Backbones (`network/dino_vit.py`, `gkd_model.py`)**:
   - Extracts visual feature embeddings and prompt text token embeddings via `load_dinov3_visual_encoder` and `load_dinov3_text_encoder` to map them into a shared cross-modal latent space.
2. **AdaptationNet & KGTransformer (`network/kg_transformer.py`)**:
   - Features undergo projection in `AdaptationNet` (comprising multi-head self-attention and residual bottlenecks) before `KGTransformer` computes the keypoint affinity matrix.
3. **Deconv Upsampling & SoftArgmax (`network/upsampler.py`, `utils/utils.py`)**:
   - `DetectionHead` utilizes `UpsamplerByDeconv` to upsample low-resolution features into high-resolution heatmaps ($H$).
   - Sub-pixel continuous coordinates are regressed via **SoftArgmax**:
     $$\hat{x} = \sum_{u, v} u \cdot \text{Softmax}(H_{u,v} / \tau), \quad \hat{y} = \sum_{u, v} v \cdot \text{Softmax}(H_{u,v} / \tau)$$
     enabling sub-pixel keypoint localization accuracy.

### 3.2 SAM3 Open-Vocabulary & Bounding Box IoU Matching
In `ai_models.py` (`_best_iou_match`), candidate bounding boxes predicted by SAM3 are matched against a reference target box:

$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\max(0, x_2^i - x_1^i) \times \max(0, y_2^i - y_1^i)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

If $\max(\text{IoU}) < \text{min\_iou}$ (threshold set to 0.1), candidate boxes are rejected and assigned a score of `0.0` to eliminate false positives.

### 3.3 SAM2 Video Memory Mechanism & LRU Frame State Cache
SAM 2.1 introduces a Memory Bank architecture. In `ultralytics_sam_tasks.py`, `sam3_frame_state_cache` caches preprocessed frame features in memory using an LRU eviction strategy:

```python
def _sam3_frame_cache_put(key, value):
    if len(sam3_frame_state_cache) >= max_cache_size:
        oldest_key = next(iter(sam3_frame_state_cache))
        del sam3_frame_state_cache[oldest_key]
    sam3_frame_state_cache[key] = value
```
Chunking prevents GPU Out-Of-Memory (OOM) errors during long video tracking tasks:

$$\text{chunk\_end} = \min(\text{chunk\_start} + \text{chunk\_size}, \text{end\_frame} + 1)$$

### 3.4 CLIP Zero-Shot Classification & 8-Template Prompt Ensembling
`clip_model.py` implements prompt ensembling standard in OpenAI and FiftyOne benchmarks:
```python
templates = [
    "a photo of a {}.", "a close-up photo of the {}.", "a cropped photo of a {}.",
    "a bright photo of the {}.", "a good photo of a {}.", "a photo of the {}.",
    "a small photo of a {}.", "a picture of the {}."
]
```
Text prompt vectors for class $c$ are averaged and normalized:

$$\vec{w}_c = \text{Normalize}\left( \frac{1}{T} \sum_{t=1}^T \text{CLIP}_{\text{text}}(\text{Template}_t(c)) \right)$$

Softmax probabilities are computed as:

$$\text{Probability}_c = \frac{\exp(\text{logit\_scale} \cdot \vec{f}_{\text{img}} \cdot \vec{w}_c^T)}{\sum_{k} \exp(\text{logit\_scale} \cdot \vec{f}_{\text{img}} \cdot \vec{w}_k^T)}$$

When single-class $N=1$ classification is requested, a negative background prompt `"other background, floor or irrelevant object"` is automatically appended.

### 3.5 Deep Semantic & HSV Dual-Stream Data Audit Model
`ai_models.py` (`check_dataset_consistency`) computes dual-stream feature fusion vectors for dataset auditing:
- **Combined Feature Vector**:
  $$\vec{v}_{\text{combined}} = \text{Normalize}\Big( \text{Concat}\big[ 0.7 \cdot \frac{\vec{v}_{\text{semantic}}}{\|\vec{v}_{\text{semantic}}\|_2}, \; 0.3 \cdot \frac{\vec{v}_{\text{HSV}}}{\|\vec{v}_{\text{HSV}}\|_2} \big] \Big)$$
- **Class Centroid & 2-Sigma Lower Bound**:
  $$\vec{C}_k = \text{Normalize}\Big( \frac{1}{|S_k|} \sum_{i \in S_k} \vec{v}_i \Big), \quad \text{Thresh}_k = \max(0.50, \; \mu_{\text{sim}} - 2.0 \times \sigma_{\text{sim}})$$
- **Cross-Class Confusion Condition**: Triggers an audit alert if $\exists j \neq k$ such that $\vec{v}_i \cdot \vec{C}_j > \vec{v}_i \cdot \vec{C}_k + 0.06$ and $\vec{v}_i \cdot \vec{C}_j > 0.65$.

### 3.6 Cross-Frame BBox & Keypoint Linear Interpolation
For matched objects (via `object_id`) across keyframes $F_{\text{start}}$ and $F_{\text{end}}$:

$$t = \frac{i}{F_{\text{end}} - F_{\text{start}}}, \quad i \in [1, F_{\text{end}} - F_{\text{start}} - 1]$$
$$X(t) = X_{\text{start}} + t \cdot (X_{\text{end}} - X_{\text{start}})$$
$$Y(t) = Y_{\text{start}} + t \cdot (Y_{\text{end}} - Y_{\text{start}})$$

Keypoint visibility flags $v$ and skeleton topology connections remain invariant.

### 3.7 4-in-1 Mosaic Augmentation & YOLO Coordinate Clamping
In `file_storage.py` (`_clip_yolo_bbox`), normalized coordinates $(cx, cy, w, h)$ are clamped to maintain valid bounding box geometries:

$$\begin{aligned}
x_{\min} &= \max(0.0, \min(1.0 - \epsilon, cx - w/2)) \\
x_{\max} &= \max(\epsilon, \min(1.0, cx + w/2)) \\
new\_cx &= x_{\min} + (x_{\max} - x_{\min}) / 2 \\
new\_w &= x_{\max} - x_{\min}
\end{aligned}$$
where $\epsilon = 10^{-6}$ prevents bounding box coordinates from overflowing image boundaries.

---

## 4. Full Directory & Subfolder Architecture Panorama

### 4.1 `gkdt_engine/` Subsystem

- `gkdt_engine/config.py`: Defines network dimensions, downsize factors, Gaussian sigma, and batch configurations for GKDT.
- `gkdt_engine/main_gkd.py`: Training, inference, and evaluation pipeline engine.
- `gkdt_engine/network/gkd_model.py`: **Main GKDT model definition**, containing `FeatureProjector`, `visual_prompt_extraction`, and `DetectionHead`.
- `gkdt_engine/network/kg_transformer.py`: `KGTransformer` self-attention and cross-attention modules.
- `gkdt_engine/network/upsampler.py`: `UpsamplerByDeconv` multi-stage transposed convolution network.
- `gkdt_engine/network/dino_vit.py` & `dino_utils.py`: DINOv3 Vision Transformer backbone wrapper.
- `gkdt_engine/core/loss_lw.py`: Heatmap loss calculation and Object Keypoint Similarity (OKS) metric evaluation.
- `gkdt_engine/utils/heatmap.py` & `sample_keypoints.py`: 2D Gaussian heatmap generators and random keypoint samplers.
- `gkdt_engine/utils/utils.py`: `SoftArgmax` sub-pixel coordinate regressor and coordinate transformation matrices.

### 4.2 `sam2/` & `sam3/` Model Predictors & Architectures

- `sam2/sam2_image_predictor.py`: `SAM2ImagePredictor` for single-frame interactive point and box prompting.
- `sam2/sam2_video_predictor.py`: `SAM2VideoPredictor` for video tracking, managing Memory Encoders, Memory Attention, and Occlusion Heads.
- `sam2/automatic_mask_generator.py`: Grid-based automatic full-image mask generator (`SAM2AutomaticMaskGenerator`).
- `sam2/build_sam.py`: Builder for SAM 2.1 Hiera models (Tiny, Small, Base+, Large).
- `sam3/model_builder.py`: SAM3 model builder constructing cross-modal vision backbones, multi-modal prompt heads, and mask decoders.
- `sam3/visualization_utils.py`: Visualization utilities for bounding boxes and polygon masks.

### 4.3 Python Backend Core Logic & Scheduling Modules

1. `app.py` (130.8 KB, 2946 lines)
   - **Controller & Router Hub**: Defines 50+ HTTP endpoints and 1 SSE long-connection route. Handles video uploads, frame read/writes, real-time augmentation previews (`preview_augmentations`), SAM2/3 scheduling, and `pywebview` window launch.
2. `ai_models.py` (33.5 KB, 759 lines)
   - **AI Coordinator**: Implements `lam_predict` (Click-to-Label), `predict_from_one_shot` (global visual exemplar matching), `predict_by_class_text` (open-vocabulary detection), and `check_dataset_consistency` (dual-stream audit).
3. `ultralytics_sam_tasks.py` (23.0 KB, 522 lines)
   - **SAM Engine Wrappers**: Houses `_load_sam2_models`, `_load_sam3_models`, `sam3_query_frame`, and `track_video_ultralytics` with LRU state caching.
4. `gkdt_tasks.py` (13.3 KB, 306 lines)
   - **Grounded Keypoint Transformer Drivers**: Applies DINOv3 BPE path patches; provides text-guided pose (`predict_pose_from_text`), SAM point-guided pose (`predict_pose_from_sam_point`), and SAM3 + GKDT batch pose (`predict_sam3_gkdt_batch_pose`).
5. `clip_model.py` (8.9 KB, 215 lines)
   - **CLIP Deep Feature Extractor**: Manages HuggingFace model weights, 512/768-dim feature vector extractions, and 8-template prompt ensembling.
6. `background_tasks.py` (48.8 KB, 894 lines)
   - **Asynchronous Task Center**: Multi-threaded worker for video frame extraction (`extract_frames_task`), SAM3 batch labeling (`apply_class_to_videos_task`), batch pose inference (`apply_pose_class_to_videos_task`), SAM2 tracking (`start_sam2_tracking_task`), and dataset generation (`create_dataset_task`).
7. `database.py` (46.9 KB, 1026 lines)
   - **SQLAlchemy SQLite ORM**: Manages 7 database tables with WAL mode, transaction rollbacks, cascade frame deletions, and JSON schema parsing.
8. `file_storage.py` (12.8 KB, 318 lines)
   - **Storage & Image Utilities**: Locates video and frame files, generates 4-in-1 Mosaic images (`create_mosaic_image`), and performs coordinate clamping.
9. `settings_manager.py` (6.2 KB, 147 lines)
   - **System Config & Hardware Detection**: Manages `settings.json` reads/writes and auto-detects CUDA hardware to allocate PyTorch devices.
10. `annotation_model.py` (5.2 KB, 139 lines)
    - **Data Schemas**: Defines `AnnotationObject` (BBox/Polygon/Keypoints) and `AnnotationData` (Frame model) alongside default COCO 17-keypoint schemas (`COCO_POSE_17_SCHEMA`).
11. `bbox_writer.py` (5.3 KB, 160 lines)
    - **Bounding Box Text Parser**: Parses `x1,y1,x2,y2,label,object_id` strings, converts to normalized YOLO coordinates, and validates format syntax.
12. `config.py` (1.5 KB, 41 lines)
    - **Global Constants**: Defines `BASE_DIR`, `DATABASE_FILE`, `STORAGE_DIR`, and limits for file sizes, frame rates, and resolutions.

### 4.4 `exporters/` Multi-Format Dataset Export Engines

- `exporters/base.py`: Abstract base class `BaseExporter` and exporter registry `ExporterRegistry`.
- `exporters/detection/yolo_detect.py`: YOLOv5/v8/v11 detection format exporter featuring Albumentations coordinate transformations and bounding box tight-fitting algorithms (`tight_fit_bbox`).
- `exporters/detection/coco_detect.py`: Standard COCO JSON detection exporter.
- `exporters/detection/pascal_voc.py`: Pascal VOC XML format exporter.
- `exporters/segmentation/yolo_seg.py`: YOLO instance segmentation exporter using `multiprocessing.Pool` for parallel polygon processing.
- `exporters/segmentation/semantic_mask.py`: Semantic segmentation PNG mask exporter (per-pixel category color mapping).
- `exporters/pose/yolo_pose.py`: YOLO Pose txt format exporter (`cls cx cy w h x1 y1 v1 ...`).
- `exporters/pose/coco_pose.py`: COCO Keypoints JSON format exporter.
- `exporters/classification/yolo_cls.py`: YOLO classification folder layout and tabular dataset exporter.
- `exporters/classification/folder_class.py`: Directory-per-class dataset exporter.

### 4.5 `templates/` Frontend Templates & Inference Controls

- `templates/labelVideo.html` (180.6 KB): **Main Annotation UI**, featuring HTML5 Canvas, a dark glassmorphic sidebar, and video playback controls.
- `templates/root.html` (100.9 KB): Project & Video Management Dashboard for video uploads, frame extraction management, dataset imports, and model weight injections.
- `templates/dataset_analysis.html` (50.1 KB): **Dataset Quality Dashboard**, presenting class co-occurrence matrices, annotation density charts, brightness distribution plots, and anomaly tables.
- `templates/_augmentation_controls.html` (24.5 KB): Advanced data augmentation configuration panel for geometric, color/HSV, blur, and 4-in-1 Mosaic transformations.
- `templates/setup.html` (20.8 KB): First-run system wizard and hardware initialization interface.
- `templates/_label_inference_tools.html` (6.7 KB): Inference sidebar containing SAM Point, LAM Text, Smart Select, True LAM Zero-Shot, and SAM2 Tracking controls.
- `templates/_label_pose.html` (7.8 KB): Pose estimation sidebar with GKDT Click Mode, TrueLAM Pose Auto-Detect, and Keypoint Interpolation.
- `templates/_label_detection.html`, `_label_segmentation.html`, `_label_classification.html`: Task-specific canvas control layouts.

### 4.6 `static/js/annotation/` Canvas Annotation Engines

- `annotation_core.js` (20.0 KB): Base canvas class managing HTML5 Canvas 2D matrix transformations, a 20-step undo/redo stack, global keyboard shortcuts, and REST API communication.
- `annotator_detection.js` (3.2 KB): Object detection mode plugin supporting bounding box dragging, vertex resizing, and label assignments.
- `annotator_segmentation.js` (24.0 KB): Instance segmentation plugin supporting point-and-click polygon drawing, brush/eraser mask painting, and Marching Squares contour fitting.
- `annotator_pose.js` (61.7 KB): Keypoint pose plugin managing COCO-17 / custom skeleton rendering, joint node dragging, and interactive GKDT inference.
- `annotator_classification.js` (8.0 KB): Image classification plugin managing image tagging, CLIP clustering display, and ambiguous sample review cards.

---

## 5. Database Schema & Entity-Relationship Diagram (SQLite ERD)

```mermaid
erDiagram
    videos ||--o{ video_frames : "1 : N (CASCADE)"
    videos ||--o{ annotation_tasks : "1 : N (CASCADE)"
    
    videos {
        string video_uuid PK "Primary Key UUID"
        string description "Description (UNIQUE)"
        string video_filename "Local video filename"
        int file_size "File size in bytes"
        int width "Frame width"
        int height "Frame height"
        float fps "Frames per second"
        int frame_count "Total video frames"
        int extracted_frame_count "Extracted frames count"
        int labeled_frame_count "Labeled frames count"
        string annotation_type "Task type (detection/segmentation/pose/classification)"
        string keypoint_schema "Associated pose keypoint schema JSON"
    }

    video_frames {
        int frame_id PK "Auto-increment ID"
        string video_uuid FK "Foreign key to videos"
        int frame_number "Frame index number"
        string bboxes_text "Bounding box text representation"
        string suggested_bboxes_text "AI suggested bounding boxes"
        string tags "Classification tag comma-separated string"
        int include_frame_in_dataset "Dataset inclusion flag (1/0)"
        string annotations_json "Full JSON containing Polygon/Keypoints/Bbox"
    }

    datasets {
        string dataset_uuid PK "Dataset UUID"
        string description "Name description (UNIQUE)"
        string video_uuids "Included video UUID array JSON"
        string zip_path "Flattened ZIP archive file path"
        float eval_percent "Validation split ratio"
        float test_percent "Test split ratio"
        string export_format "Export format key"
    }

    models {
        string model_uuid PK "Model UUID"
        string description "Model description"
        string label_filename "Class labels filename"
        string model_type "Precision type (float32/float16/uint8)"
    }

    annotation_tasks {
        string task_uuid PK "Task UUID"
        string video_uuid FK "Associated video UUID"
        string assigned_to "Assigned annotator ID"
        int start_frame "Start frame index"
        int end_frame "End frame index"
        string status "Task status"
    }

    class_labels {
        int label_id PK "Auto-increment ID"
        string label_name UNIQUE "Class name"
        string sam3_prompt "Text prompt for SAM3 open-vocabulary query"
        string keypoint_schema "Pose skeleton connection schema JSON"
    }

    class_tags {
        int tag_id PK "Auto-increment ID"
        string tag_name UNIQUE "Global classification tag name"
    }
```

---

## 6. Complete RESTful API & SSE Real-time Stream Manual

### System & Configuration APIs

| Route | Method | Description | Key Payload / Form |
| :--- | :--- | :--- | :--- |
| `/setup` | GET | Render initial setup wizard (`setup.html`) | None |
| `/api/detect_hardware` | GET | Detect PyTorch CUDA capability and VRAM | None |
| `/api/complete_setup` | POST | Persist initial wizard configuration | `{gpu_device, max_workers, ...}` |
| `/api/settings` | GET/POST | Read or update system `settings.json` | Configuration JSON |
| `/api/clear_cache` | POST | Flush in-memory SAM3 and frame feature LRU caches | None |

### Video & Annotation Management APIs

| Route | Method | Description | Key Payload / Form |
| :--- | :--- | :--- | :--- |
| `/uploadVideo` | POST | Upload MP4 video and trigger background frame extraction | `file`, `description`, `annotation_type` |
| `/importFrames` | POST | Batch import standalone image files as a sequence | `files[]`, `description` |
| `/retrieveVideoFrames` | POST | Retrieve frame annotation records for a video | `{video_uuid}` |
| `/saveFrameAnnotations` | POST | Save complex JSON annotations (Polygons/Keypoints) | `{video_uuid, frame_number, annotations_json}` |
| `/storeVideoFrameBboxesText` | POST | Store raw bounding box coordinate text | `{video_uuid, frame_number, bboxes_text}` |

### AI Inference & Interaction APIs

| Route | Method | Description | Key Payload / Form |
| :--- | :--- | :--- | :--- |
| `/samPredict` | POST | Interactive point/box SAM2/3 segmentation prediction | `{video_uuid, frame_number, point_coords}` |
| `/api/sam3_text_predict` | POST | SAM3 text-prompt open-vocabulary detection | `{video_uuid, frame_number, text_prompt}` |
| `/lam_predict` | POST | Parallel prompt evaluation across all classes | `{video_uuid, frame_number, point}` |
| `/api/interpolateBboxes` | POST | Linear interpolation of bounding boxes across frames | `{video_uuid, object_id, start_frame, end_frame}` |
| `/api/interpolatePoseKeypoints` | POST | Linear interpolation of pose keypoints across frames | `{video_uuid, object_id, start_frame_number, end_frame_number}` |
| `/api/gkdt_text_pose_predict` | POST | Text prompt + GKDT automated pose inference | `{video_uuid, frame_number, class_label, bbox}` |
| `/api/gkdt_sam_pose_predict` | POST | SAM click + GKDT automated pose inference | `{video_uuid, frame_number, class_label, point}` |
| `/api/gkdt_sam3_batch_pose_predict` | POST | Multi-target full-image SAM3 + GKDT batch pose inference | `{video_uuid, frame_number, class_label, confidence}` |
| `/api/clusterClassificationImages` | POST | CLIP/HSV feature extraction + K-Means clustering | `{video_uuids, num_clusters, unlabeled_only}` |
| `/api/applyClipZeroShot` | POST | CLIP 8-prompt ensembling zero-shot classification | `{video_uuids, candidate_classes}` |
| `/api/previewAugmentations` | POST | Real-time preview of augmentations (returns Base64 image) | `{video_uuid, frame_number, augmentation_options}` |

### Video Tracking & SSE Stream APIs

| Route | Method | Description | Key Payload / Form |
| :--- | :--- | :--- | :--- |
| `/startSam2Tracking` | POST | Start background SAM2 long-term video tracking task | `{video_uuid, start_frame, end_frame, init_bboxes_text}` |
| `/streamSam2Tracking/<uuid>` | GET (SSE)| **Server-Sent Events** streaming tracking progress and box data | SSE Event Stream |
| `/stopSam2Tracking` | POST | Request cancellation of an active tracking task | `{tracker_uuid}` |

### Data Engineering & Audit APIs

| Route | Method | Description | Key Payload / Form |
| :--- | :--- | :--- | :--- |
| `/createDataset` | POST | Launch multi-processing dataset packaging task | `{video_uuids, export_format, eval_percent, test_percent}` |
| `/downloadDataset/<uuid>` | GET | Download generated `.zip` dataset archive | None |
| `/api/datasetAnalysis/<uuid>` | GET | Fetch dataset statistics (density, co-occurrence, brightness) | None |
| `/api/datasetAnalysis/<uuid>/consistency_check` | POST | Run SAM3 + HSV dual-stream dataset quality audit | `{enable_color_check: bool}` |

---

## 7. Data Augmentation & Real-Time Preview Workflow

1. **Albumentations Pipeline**: In `exporters/`, keypoints and polygon vertices are flattened into 1D coordinate arrays (`flat_kpts`) before passing through Albumentations transforms to maintain geometric topology consistency.
2. **Multi-Processing Parallel Export (`multiprocessing.Pool`)**:
   ```python
   with Pool(processes=safe_workers) as pool:
       for result in pool.imap_unordered(process_seg_frame_worker, all_tasks):
           # Incrementally update database status and report progress
   ```
3. **4-in-1 Mosaic Generation**: Crops four training images around a random center ($0.5 \sim 1.5$) and stitches them into a unified, normalized canvas.

---

## 8. System Configuration (`settings.json`)

Key adjustable runtime parameters:

```json
{
    "initial_setup_done": true,              // Wizard completion status
    "gpu_device": "auto",                    // PyTorch device ("auto", "cuda:0", "cpu")
    "sam_model_checkpoint": "sam2.1_t.pt",   // SAM checkpoint variant (t/s/b/l)
    "sam_mask_confidence": 0.70,             // Mask threshold confidence
    "default_confidence": 0.5,               // Global AI inference confidence threshold
    "max_workers": 8,                        // Max worker threads in pool
    "max_cache_size": 30,                    // Max LRU frame cache size
    "use_autocast": true,                    // Mixed precision acceleration (FP16/BF16)
    "color_confusion_factor": 2.0,           // Sensitivity factor for color audit
    "consistency_semantic_threshold": 0.3,   // Absolute semantic similarity lower bound
    "consistency_confusion_margin": 0.15,    // Cross-class confusion margin
    "enable_sam_model": true,                // Enable SAM module
    "enable_feature_extractor": true,        // Enable SAM3 retrieval module
    "enable_cls_model": true,                // Enable CLIP module
    "enable_pose_model": true                // Enable GKDT pose module
}
```

---

## 9. Environment Setup & Native Desktop Packaging

### Installing Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch (CUDA 12.1 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip install -r requirements.txt
```

### Native Desktop Compilation (PyInstaller)

The project includes `Zero2YoloYard.spec` for building standalone desktop executables via PyInstaller:

```bash
pyinstaller Zero2YoloYard.spec
```

The compiled standalone executable, bundled with Python runtime environments, `pywebview` dependencies, and frontend HTML/JS static assets, will be located in the `dist/` directory.
