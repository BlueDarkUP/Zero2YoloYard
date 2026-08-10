# Zero2YoloYard 智能化视觉数据集标注与数据工程平台

<p align="center">
  <a href="README_EN.md"><strong>English Version</strong></a> | <strong>中文 Version</strong>
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

**Zero2YoloYard** 是一套现代化、全流程、多大模型驱动的智能化视觉数据集标注、数据增强、数据质量诊断与多格式导出一体化平台。

项目采用 Python (Flask + Waitress) 作为高性能后端，借助 `pywebview` 打包为桌面原生应用，深度整合 **SAM 2.1 / SAM 3 (Segment Anything Model)**、**CLIP (Contrastive Language-Image Pretraining)**、**GKDT (Grounded Keypoint Detection Transformer)** 及其底层 `gkdt_engine` 神经网络算法，以及 **Albumentations** 图像增强引擎，全方位赋能计算机视觉四大核心任务：**目标检测 (Detection)**、**实例/语义分割 (Segmentation)**、**关键点与姿态估计 (Pose Estimation)** 和 **图像分类与聚类 (Classification)**。

---

## 目录

1. [鸣谢与开源致谢 (Acknowledgements)](#1-鸣谢与开源致谢-acknowledgements)
   - [1.6 预训练模型 Checkpoints 存放目录结构与下载指南 (Checkpoints Directory Layout & Downloads)](#16-预训练模型-checkpoints-存放目录结构与下载指南-checkpoints-directory-layout--downloads)
2. [项目架构与设计哲学](#2-项目架构与设计哲学)
3. [底层核心算法与神经网络原理](#3-底层核心算法与神经网络原理)
   - [3.1 GKDT (General Keypoint Detection Transformer) 核心引擎](#31-gkdt-general-keypoint-detection-transformer-核心引擎)
   - [3.2 SAM3 开放词汇与提示框 IoU 匹配算法](#32-sam3-开放词汇与提示框-iou-匹配算法)
   - [3.3 SAM2 视频记忆机制与 LRU 帧状态缓存](#33-sam2-视频记忆机制与-lru-帧状态缓存)
   - [3.4 CLIP 零样本分类与 8 模板 Prompt Ensembling](#34-clip-零样本分类与-8-模板-prompt-ensembling)
   - [3.5 深度语义与 HSV 色彩双流数据审计模型](#35-深度语义与-hsv-色彩双流数据审计模型)
   - [3.6 跨帧 BBox 与姿态 Keypoints 线性插值公式](#36-跨帧-bbox-与姿态-keypoints-线性插值公式)
   - [3.7 4-in-1 Mosaic 拼图与 YOLO 坐标 Clamp 防越界机制](#37-4-in-1-mosaic-拼图与-yolo-坐标-clamp-防越界机制)
4. [全量目录与子文件夹代码全景拆解](#4-全量目录与子文件夹代码全景拆解)
   - [4.1 `gkdt_engine/` 底层算法子系统](#41-gkdt_engine-底层算法子系统)
   - [4.2 `sam2/` 与 `sam3/` 深度学习预测器与大模型架构](#42-sam2-与-sam3-深度学习预测器与大模型架构)
   - [4.3 后端核心逻辑与调度模块 (Python)](#43-后端核心逻辑与调度模块-python)
   - [4.4 `exporters/` 多格式数据集导出引擎](#44-exporters-多格式数据集导出引擎)
   - [4.5 `templates/` 前端 UI 模版与推理交互控件](#45-templates-前端-ui-模版与推理交互控件)
   - [4.6 `static/js/annotation/` Canvas 交互与标注引擎](#46-staticjsannotation-canvas-交互与标注引擎)
5. [数据库全表字典与实体关系 (SQLite ER Diagram)](#5-数据库全表字典与实体关系-sqlite-er-diagram)
6. [全量 RESTful API 与 SSE 实时接口手册](#6-全量-restful-api-与-sse-实时接口手册)
7. [数据增强参数控制与图像预览全流程](#7-数据增强参数控制与图像预览全流程)
8. [系统配置参数 (`settings.json`) 说明](#8-系统配置参数-settingsjson-说明)
9. [环境搭建与原生桌面打包部署](#9-环境搭建与原生桌面打包部署)

---

## 1. 鸣谢与开源致谢 (Acknowledgements)

Zero2YoloYard 的诞生与演进离不开开源社区与学术界的无私奉献。特别向以下开源技术、深度学习模型、基础框架及学术数据集致以最崇高的敬意：

### 1.1 前沿 AI 大模型与学术研究 (Foundation AI Models & Academic Research)

- **GKDT (General Keypoint Detection Transformer) & MegaKPT Dataset**  
  特别鸣谢 **AlanLuSun** 及其学术团队在 ECCV 2026 的杰出贡献：
  - **开源仓库**: [AlanLuSun/General-Keypoint-Detection](https://github.com/AlanLuSun/General-Keypoint-Detection)
  - **学术贡献**: 提出了强大的通用关键点 Transformer 架构 (GKDT) 与统一了 29 个公共数据集、包含 130 万+ 物体实例的大规模高质数据集 **MegaKPT**。系统内置的 `gkdt_engine` 算法正是基于该成果构建，赋予了平台跨类别的开放词汇姿态估计能力。
- **SAM 2.1 & SAM 3 (Segment Anything Model 2 & 3)**  
  感谢 **Meta AI Research** 团队开源的开创性通用分割大模型：
  - **开源仓库**: [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
  - **贡献说明**: 提供了卓越的单帧分割、Mask 解码与基于 Memory Bank 的视频跨时间步特征跟踪能力。
- **CLIP (Contrastive Language-Image Pretraining)**  
  感谢 **OpenAI** 团队与 **HuggingFace** Transformers 社区：
  - **开源仓库**: [openai/CLIP](https://github.com/openai/CLIP) / [huggingface/transformers](https://github.com/huggingface/transformers)
  - **贡献说明**: 提供了连接文本与视觉图层的特征空间表达，支撑了平台中的 Zero-Shot 图像分类与 K-Means 无监督特征聚类。
- **DINOv3 (Self-Supervised Vision Transformer)**  
  感谢 **Meta AI** 的自监督视觉基础模型 DINOv3 系列，为 GKDT 与语义检索提供了强大的底层视觉 Backbone 特征表达。
- **Ultralytics YOLO Ecosystem**  
  感谢 **Ultralytics** 团队在 YOLO 系列 (YOLOv5 / YOLOv8 / YOLOv11) 目标检测、实例分割与姿态估计格式标准上的贡献。

### 1.2 深度学习与图像处理基础库 (Deep Learning & CV Libraries)

- **PyTorch & Torchvision**: 提供了强大的 GPU 张量计算与神经网络自动求导基础设施。
- **OpenCV (Open Source Computer Vision Library)**: 提供了图像 RGB/HSV 空间转换、直方图计算、图像切块与实时渲染引擎。
- **Albumentations**: [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations) 提供了工业级高效率的图像数据增强转换管道（支持 Keypoints 与 Polygons 变换）。
- **NumPy & Pillow (PIL)**: 矩阵计算与基础图像操作。
- **scikit-learn**: 提供了高效的 K-Means 聚类与数据统计解算。

### 1.3 后端 Web & 桌面原生架构 (Backend & Desktop Engineering)

- **Flask**: 提供了敏捷且功能强大的 RESTful Web 路由与控制响应层。
- **Waitress**: 提供了生产级高并发多线程 WSGI Web 服务器支持。
- **pywebview**: 提供了 Python 绑定原生 Webview 渲染窗口跨平台桌面体验的能力。
- **SQLAlchemy & SQLite3**: 提供了高可靠性的 ORM 与开启 WAL 预写日志的内置关系型数据库支持。
- **PyInstaller**: 提供了将 Python 应用打包为独立 Windows / 桌面双击可执行文件的工具支持。

### 1.4 前端界面与工程体系 (Frontend Engineering & Design)

- **Bootstrap 4 & Bootstrap Icons**: 现代化 Glassmorphic 样式库与图标组件。
- **jQuery**: 客户端 DOM 操作与事件监听基石。
- **Google Fonts (Inter & JetBrains Mono)**: 现代极简审美 UI 字体与代码字体规范。

### 1.5 传承与起源 (Heritage)

- 本项目起源并基于 **FIRST Machine Learning Toolchain** 的早期范式，由 **FIRST Tech Challenge (FTC) 27570 战队成员 BlueDarkUP** 深度重构与拓展研发。

### 1.6 预训练模型 Checkpoints 存放目录结构与下载指南 (Checkpoints Directory Layout & Downloads)

项目采用模型解耦设计，所有预训练权重均保存在 `checkpoints/` 以及 `gkdt_engine/` 下的专属子目录中。

#### 1.6.1 官方标准 `checkpoints/` 目录树架构

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

以及 `gkdt_engine/` 底层依赖模型的规范路径：

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

#### 1.6.2 命令行精准下载指令 (Precise Download Commands)

```bash
# 安装 HuggingFace 极速下载 CLI 工具
pip install -U huggingface_hub

# 1. 下载 SAM 2.1 官方权重 (直接存放在 checkpoints/ 根目录下)
hf download facebook/sam2.1-hiera-tiny sam2.1_hiera_tiny.pt --local-dir checkpoints --local-dir-use-symlinks False
mv checkpoints/sam2.1_hiera_tiny.pt checkpoints/sam2.1_t.pt

hf download facebook/sam2.1-hiera-small sam2.1_hiera_small.pt --local-dir checkpoints --local-dir-use-symlinks False
mv checkpoints/sam2.1_hiera_small.pt checkpoints/sam2.1_s.pt

hf download facebook/sam2.1-hiera-base-plus sam2.1_hiera_base_plus.pt --local-dir checkpoints --local-dir-use-symlinks False
mv checkpoints/sam2.1_hiera_base_plus.pt checkpoints/sam2.1_b.pt

hf download facebook/sam2.1-hiera-large sam2.1_hiera_large.pt --local-dir checkpoints --local-dir-use-symlinks False
mv checkpoints/sam2.1_hiera_large.pt checkpoints/sam2.1_l.pt

# 2. 下载 SAM 3 官方权重 (存放在 checkpoints/sam3/sam3.pt)
hf download facebook/sam3 --local-dir checkpoints/sam3

# 3. 下载 CLIP 多尺寸语义模型 (存放在 checkpoints/clip/ 下)
hf download openai/clip-vit-base-patch16 --local-dir checkpoints/clip/clip-vit-base-patch16
hf download openai/clip-vit-base-patch32 --local-dir checkpoints/clip/clip-vit-base-patch32
hf download openai/clip-vit-large-patch14 --local-dir checkpoints/clip/clip-vit-large-patch14

# 4. 下载 GKDT 姿态预测模型 (存放在 gkdt_engine/output/ 下)
cd gkdt_engine
hf download changshenglu/GKDT-L_for_App --local-dir output/GKDT-L_for_app/model
hf download changshenglu/GKDT-H_for_App --local-dir output/GKDT-H_for_app/model

# 5. 下载 GroundingDINO 与 LocateAnything 辅助检测模型
curl.exe -L "https://ghproxy.net/https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -o "test_real_world/object_detector_lib/weights/groundingdino_swint_ogc.pth"
hf download nvidia/LocateAnything-3B --local-dir test_real_world/object_detector_lib/weights/LocateAnything-3B
cd ..
```

---

## 2. 项目架构与设计哲学

- **原生桌面体验与 Web 混合架构**：后端基于 Flask 部署在 Waitress 高并发多线程服务器 (`host=127.0.0.1, port=5000`) 上，前端由 `pywebview` 绑定为 Native 窗口 (1920x1080 默认分辨率)，兼具 Web UI 的灵活度与桌面程序的流畅感。
- **零样本 (Zero-Shot) 开放词汇标注**：无需训练数据集专属模型，直接输入自然语言 Prompt 或类名称即可利用 SAM3 和 GKDT 自动解算框、多边形掩码及 17 点姿态骨架。
- **多并发与后台长任务异步调度**：抽帧、SAM2 视频跟踪、批量打标及数据集导出均为异步线程任务，结合 SSE (Server-Sent Events) 提供毫秒级实时进度推送。
- **无依赖的前端模块化设计**：前端摒弃复杂的现代框架打包，直接采用 Vanilla ES6 类扩展 (`AnnotationCore`, `AnnotatorDetection`, `AnnotatorSegmentation`, `AnnotatorPose`, `AnnotatorClassification`) + HTML5 Canvas Matrix Transform 控制缩放与平移。

---

## 3. 底层核心算法与神经网络原理

### 3.1 GKDT (General Keypoint Detection Transformer) 核心引擎
内置于 `gkdt_engine/` 中的通用关键点检测 Transformer 包含了完整的端到端解算流程：

1. **DINOv3 Visual & Text Backbones (`network/dino_vit.py`, `gkd_model.py`)**：
   - 提取输入图像高阶特征与文本提示词的 Token Embedding，借由 `load_dinov3_visual_encoder` 和 `load_dinov3_text_encoder` 构建跨模态表征空间。
2. **AdaptationNet 与 KGTransformer (`network/kg_transformer.py`)**：
   - 使用包含多头自注意力 (`Transformer`) 与残差瓶颈块 (`Bottleneck`) 的 `AdaptationNet` 实施特征映射，配合 `KGTransformer` 解算关键点关联矩阵。
3. **Deconv 解置卷积上采样与 SoftArgmax (`network/upsampler.py`, `utils/utils.py`)**：
   - `DetectionHead` 利用 `UpsamplerByDeconv` 将低分辨率特征上采样为高清晰热力图 (Heatmap)；
   - 配合 **SoftArgmax** 亚像素回归公式求解连续概率期望点：
     $$\hat{x} = \sum_{u, v} u \cdot \text{Softmax}(H_{u,v} / \tau), \quad \hat{y} = \sum_{u, v} v \cdot \text{Softmax}(H_{u,v} / \tau)$$
     实现亚像素级的关键点坐标定位。

### 3.2 SAM3 开放词汇与提示框 IoU 匹配算法
在 `ai_models.py` 的 `_best_iou_match` 中，针对 SAM3 预测的候选框集与目标参考框进行重叠度匹配：

$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\max(0, x_2 - x_1) \times \max(0, y_2 - y_1)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

当 $\max(\text{IoU}) < \text{min-iou}$ (阈值设为 0.1) 时，认为候选框无实际匹配，系统强行输出匹配得分 `0.0`，杜绝假阳性。

### 3.3 SAM2 视频记忆机制与 LRU 帧状态缓存
SAM 2.1 引入了 Memory Bank 机制。在 `ultralytics_sam_tasks.py` 中，采用 `sam3_frame_state_cache` 保持预处理好的视频帧特征：

```python
def _sam3_frame_cache_put(key, value):
    if len(sam3_frame_state_cache) >= max_cache_size:
        oldest_key = next(iter(sam3_frame_state_cache))
        del sam3_frame_state_cache[oldest_key]
    sam3_frame_state_cache[key] = value
```
在视频跟踪任务中，利用拆分机制分块计算显存占用：

$$\text{chunk-end} = \min(\text{chunk-start} + \text{chunk-size}, \text{end-frame} + 1)$$

### 3.4 CLIP 零样本分类与 8 模板 Prompt Ensembling
`clip_model.py` 实现了 OpenAI/FiftyOne 标准的 Prompt 组装算法：
```python
templates = [
    "a photo of a {}.", "a close-up photo of the {}.", "a cropped photo of a {}.",
    "a bright photo of the {}.", "a good photo of a {}.", "a photo of the {}.",
    "a small photo of a {}.", "a picture of the {}."
]
```
对属于类别 $c$ 的文本向量求均值：

$$\vec{w}_c = \text{Normalize}\left( \frac{1}{T} \sum_{t=1}^T \text{CLIP}_{\text{text}}(\text{Template}_t(c)) \right)$$

计算最终 Softmax 概率矩阵：

$$\text{Probability}_c = \frac{\exp(\text{logit-scale} \cdot \vec{f}_{\text{img}} \cdot \vec{w}_c^T)}{\sum_{k} \exp(\text{logit-scale} \cdot \vec{f}_{\text{img}} \cdot \vec{w}_k^T)}$$

当类别数 $N=1$ 时，自动引入背景负类 `"other background, floor or irrelevant object"` 防止过拟合。

### 3.5 深度语义与 HSV 色彩双流数据审计模型
`ai_models.py` 中的 `check_dataset_consistency` 实现了双通道向量特征加权融合：
- **组合特征向量**：
  $$\vec{v}_{\text{combined}} = \text{Normalize}\Big( \text{Concat}\big[ 0.7 \cdot \frac{\vec{v}_{\text{semantic}}}{\|\vec{v}_{\text{semantic}}\|_2}, \; 0.3 \cdot \frac{\vec{v}_{\text{HSV}}}{\|\vec{v}_{\text{HSV}}\|_2} \big] \Big)$$
- **类别中枢 Centroid** 与 **2-Sigma 阈值下界**：
  $$\vec{C}_k = \text{Normalize}\Big( \frac{1}{|S_k|} \sum_{i \in S_k} \vec{v}_i \Big), \quad \text{Thresh}_k = \max(0.50, \; \mu_{\text{sim}} - 2.0 \times \sigma_{\text{sim}})$$
- **跨类混淆条件**：若存在 $j \neq k$ 满足 $\vec{v}_i \cdot \vec{C}_j > \vec{v}_i \cdot \vec{C}_k + 0.06$ 且 $\vec{v}_i \cdot \vec{C}_j > 0.65$，则触发警告。

### 3.6 跨帧 BBox 与姿态 Keypoints 线性插值公式
针对起始帧 $F_{\text{start}}$ 与结束帧 $F_{\text{end}}$ 中的特定对象（基于 `object_id` 强匹配）：

$$t = \frac{i}{F_{\text{end}} - F_{\text{start}}}, \quad i \in [1, F_{\text{end}} - F_{\text{start}} - 1]$$
$$X(t) = X_{\text{start}} + t \cdot (X_{\text{end}} - X_{\text{start}})$$
$$Y(t) = Y_{\text{start}} + t \cdot (Y_{\text{end}} - Y_{\text{start}})$$

针对姿态 keypoint，保持其可见度 $v$ 与骨架连接拓扑不变。

### 3.7 4-in-1 Mosaic 拼图与 YOLO 坐标 Clamp 防越界机制
在 `file_storage.py` 的 `_clip_yolo_bbox` 中，针对归一化坐标 $(cx, cy, w, h)$ 执行严格几何截断：

$$\begin{aligned}
x_{\min} &= \max(0.0, \min(1.0 - \epsilon, cx - w/2)) \\
x_{\max} &= \max(\epsilon, \min(1.0, cx + w/2)) \\
cx_{\text{new}} &= x_{\min} + (x_{\max} - x_{\min}) / 2 \\
w_{\text{new}} &= x_{\max} - x_{\min}
\end{aligned}$$
其中 $\epsilon = 10^{-6}$，防止因为数据增强产生溢出阻断后续训练。

---

## 4. 全量目录与子文件夹代码全景拆解

### 4.1 `gkdt_engine/` 底层算法子系统

- `gkdt_engine/config.py`: 定义 GKDT 网络结构、Downsize Factor、Sigma 与 Batch 运算配置。
- `gkdt_engine/main_gkd.py`: 训练、推理与验证全流程引擎。
- `gkdt_engine/network/gkd_model.py`: **GKDT 主模型定义**，包含 `FeatureProjector` (维度投影)、`visual_prompt_extraction` (视觉提示提炼) 与 `DetectionHead` (反卷积热图解码)。
- `gkdt_engine/network/kg_transformer.py`: `KGTransformer` 自注意力与交叉注意力解算模块。
- `gkdt_engine/network/upsampler.py`: `UpsamplerByDeconv` 多级反卷积上采样网络。
- `gkdt_engine/network/dino_vit.py` & `dino_utils.py`: DINOv3 视觉 Transformer 编码器提取器。
- `gkdt_engine/core/loss_lw.py`: 计算关键点 Heatmap 响应损失与 OKs 衡量指标。
- `gkdt_engine/utils/heatmap.py` & `sample_keypoints.py`: 生成 2D 高斯热力图分布与随机关键点采样器。
- `gkdt_engine/utils/utils.py`: 亚像素坐标回归器 `SoftArgmax`、相似度计算与坐标转换矩阵。

### 4.2 `sam2/` 与 `sam3/` 深度学习预测器与大模型架构

- `sam2/sam2_image_predictor.py`: 单帧多点/多框交互式分割预测器 (`SAM2ImagePredictor`)。
- `sam2/sam2_video_predictor.py`: 视频级分割与时间步记忆传播预测器 (`SAM2VideoPredictor`)，掌管 Memory Encoder、Memory Attention 与 Occlusion Head。
- `sam2/automatic_mask_generator.py`: 自动化网格 (Grid) 极速遮罩生成器 (`SAM2AutomaticMaskGenerator`)。
- `sam2/build_sam.py`: 根据配置文件构建 SAM 2.1 Hiera 架构 (Tiny, Small, Base+, Large)。
- `sam3/model_builder.py`: SAM3 大模型架构，构建跨模态 Vision Backbone、Text/Box 多模态检索 Head 及 Prompt Decoders。
- `sam3/visualization_utils.py`: SAM3 图像级/帧级检测框与掩码渲染绘制工具。

### 4.3 后端核心逻辑与调度模块 (Python)

1. `app.py` (130.8 KB, 2946 行)
   - **控制器路由中心**：定义了 50+ 个 HTTP 路由与 1 个 SSE 长连接。处理前端视频列表、图片上传、帧数据读写、实时数据增强预览 (`preview_augmentations`)、SAM2 / SAM3 预测调度及 PyWebView 宿主启动。
2. `ai_models.py` (33.5 KB, 759 行)
   - **AI 功能中枢**：封装 `lam_predict` (Click-to-Label)、`predict_from_one_shot` (框选样例全局匹配)、`predict_by_class_text` (开放词汇分类检测) 及数据集语义+HSV双通道质量审查 `check_dataset_consistency`。
3. `ultralytics_sam_tasks.py` (23.0 KB, 522 行)
   - **SAM 引擎底层包**：封装 `_load_sam2_models`, `_load_sam3_models`, `sam3_query_frame` 及视频序列追踪 `track_video_ultralytics`，管理帧特征 LRU 状态缓存。
4. `gkdt_tasks.py` (13.3 KB, 306 行)
   - **Grounded Keypoint Transformer 驱动**：应用 DINOv3 路径修复 Patch；提供基于文本指示 `predict_pose_from_text`、结合 SAM 点击 `predict_pose_from_sam_point` 及 SAM3 + GKDT 盲扫 `predict_sam3_gkdt_batch_pose` 三种姿态计算。
5. `clip_model.py` (8.9 KB, 215 行)
   - **CLIP 深度学习提取器**：管理 `Transformers` 模型权重加载、512/768 维特征向量提取与 Prompt Ensembling Zero-Shot 预测。
6. `background_tasks.py` (48.8 KB, 894 行)
   - **异步任务中心**：多线程管理抽帧 (`extract_frames_task`)、跨视频 SAM3 批量打标 (`apply_class_to_videos_task`)、跨视频姿态推导 (`apply_pose_class_to_videos_task`)、SAM2 视频分割跟踪 (`start_sam2_tracking_task`) 及导出解算打包 (`create_dataset_task`)。
7. `database.py` (46.9 KB, 1026 行)
   - **SQLAlchemy SQLite ORM 层**：配置 WAL 预写日志，管理 7 大数据表，处理事务回滚、帧索引级联删除与关键点 Schema JSON 字段解析。
8. `file_storage.py` (12.8 KB, 318 行)
   - **存储与图像处理**：负责视频、帧 JPEG 文件定位、路径格式化；提供 4-in-1 Mosaic 拼接图生成 `create_mosaic_image` 与边界 Clamp。
9. `settings_manager.py` (6.2 KB, 147 行)
   - **系统配置与硬件检测**：管理 `settings.json` 的自动补全与落盘；智能检测系统 CUDA 驱动并自动分配 `torch.device`。
10. `annotation_model.py` (5.2 KB, 139 行)
    - **标准数据模型定义**：提供 `AnnotationObject` (表示单个 BBox/Polygon/Keypoint) 与 `AnnotationData` (整帧模型)。内建 COCO 17 关节规范表 `COCO_POSE_17_SCHEMA`。
11. `bbox_writer.py` (5.3 KB, 160 行)
    - **检测框文本解析**：解析 `x1,y1,x2,y2,label,object_id` 格式字符串；提供 YOLO 归一化转换算法与格式校验 `validate_bboxes_text`。
12. `config.py` (1.5 KB, 41 行)
    - **全局常量配置**：定义 `BASE_DIR`, `DATABASE_FILE`, `STORAGE_DIR` 以及最大视频大小、分辨率、帧数限制字典。

### 4.4 `exporters/` 多格式数据集导出引擎

- `exporters/base.py`: 抽象基类 `BaseExporter` 与导出器注册中心 `ExporterRegistry`。
- `exporters/detection/yolo_detect.py`: YOLOv5/v8/v11 检测格式导出，提供基于 Albumentations 的关键点模式补丁 `build_augmentation_pipeline_for_keypoints` 与框按比例微调算法 `tight_fit_bbox`。
- `exporters/detection/coco_detect.py`: 标准 COCO JSON 检测格式导出。
- `exporters/detection/pascal_voc.py`: XML 节点格式导出。
- `exporters/segmentation/yolo_seg.py`: YOLO 实例分割多边形导出，结合 `multiprocessing.Pool` 多进程并行计算多边形坐标及映射。
- `exporters/segmentation/semantic_mask.py`: 语义分割图像 Mask（全像素语义类别填色 PNG）导出。
- `exporters/pose/yolo_pose.py`: YOLO Pose 格式 txt 导出 (`cls cx cy w h x1 y1 v1 ...`)。
- `exporters/pose/coco_pose.py`: COCO Keypoints JSON 格式导出。
- `exporters/classification/yolo_cls.py`: YOLO 架构图像分类目录及数据表导出。
- `exporters/classification/folder_class.py`: 按类别文件夹归档导出。

### 4.5 `templates/` 前端 UI 模版与推理交互控件

- `templates/labelVideo.html` (180.6 KB): **主标注系统界面**，集成了 HTML5 Canvas 画布、极简暗黑 Glassmorphic 侧边栏与视频控制播放条。
- `templates/root.html` (100.9 KB): 视频与数据集管理大厅，提供视频上传、切帧管理、导入数据集管理及模型文件注入。
- `templates/dataset_analysis.html` (50.1 KB): **数据集质量分析仪表盘**，包含共现矩阵图、标注密度图、亮度分布图、异常检测交互表。
- `templates/_augmentation_controls.html` (24.5 KB): 高级数据增强控制面板，提供几何平移/旋转、颜色/HSV/灰度、高斯/运动模糊、Mosaic 4合1 手风琴配置面板。
- `templates/setup.html` (20.8 KB): 首期初始化与硬件环境配置检测向导。
- `templates/_label_inference_tools.html` (6.7 KB): 推理工具侧边栏，包含 SAM Point, LAM Text, Smart Select, True LAM Zero-Shot 及 SAM2 Video Tracking 控制组。
- `templates/_label_pose.html` (7.8 KB): 姿态工具侧边栏，支持 GKDT 点选模式、TrueLAM 盲扫与姿态跨帧插值设置。
- `templates/_label_detection.html`, `_label_segmentation.html`, `_label_classification.html`: 各子标注任务专用的控件模版。

### 4.6 `static/js/annotation/` Canvas 交互与标注引擎

- `annotation_core.js` (20.0 KB): 核心基类，掌管 HTML5 Canvas MatrixTransform 矩阵缩放/平移，维持 20 步 Undo/Redo 历史栈，捕获全局键盘快捷键（快捷切类、保存、撤销），与 REST API 建立交互。
- `annotator_detection.js` (3.2 KB): 目标检测框模式插件，支持鼠标拖拽生成矩形框、顶点拉伸与标签分配。
- `annotator_segmentation.js` (24.0 KB): 分割多边形插件，支持点选闭合多边形、Brush 刷子/Eraser 橡皮擦涂抹与 Marching Squares 边缘轮廓点拟合算法。
- `annotator_pose.js` (61.7 KB): 关键点姿态插件，负责 COCO-17 / 自定义 Keypoint Schema 的节点连线高亮渲染、手势节点拖动与 GKDT 自动预测交互。
- `annotator_classification.js` (8.0 KB): 分类与标记插件，负责图像标签选择、CLIP 聚类结果展示与 Ambiguous 模糊样本判定卡片。

---

## 5. 数据库全表字典与实体关系 (SQLite ER Diagram)

```mermaid
erDiagram
    videos ||--o{ video_frames : "1 : N (CASCADE)"
    videos ||--o{ annotation_tasks : "1 : N (CASCADE)"
    
    videos {
        string video_uuid PK "主键 UUID"
        string description "描述 (UNIQUE)"
        string video_filename "视频文件本地名"
        int file_size "文件字节数"
        int width "帧宽度"
        int height "帧高度"
        float fps "帧率"
        int frame_count "总帧数"
        int extracted_frame_count "已抽取帧数"
        int labeled_frame_count "已标注帧数"
        string annotation_type "标注类型 (detection/segmentation/pose/classification)"
        string keypoint_schema "对应姿态的关键点配置 JSON"
    }

    video_frames {
        int frame_id PK "自增主键"
        string video_uuid FK "外键指向 videos"
        int frame_number "帧序号"
        string bboxes_text "文本框坐标与标签"
        string suggested_bboxes_text "AI 建议框坐标与标签"
        string tags "图像分类标签列"
        int include_frame_in_dataset "是否包含在导出库中 (1/0)"
        string annotations_json "包含 Polygon/Keypoints/Bbox 的全量 JSON"
    }

    datasets {
        string dataset_uuid PK "数据集 UUID"
        string description "名称描述 (UNIQUE)"
        string video_uuids "包含的视频 UUID 数组 JSON"
        string zip_path "打平 ZIP 压缩包路径"
        float eval_percent "验证集占比"
        float test_percent "测试集占比"
        string export_format "导出格式 key"
    }

    models {
        string model_uuid PK "模型 UUID"
        string description "模型描述"
        string label_filename "类别描述文件名"
        string model_type "模型精度类型 (float32/float16/uint8)"
    }

    annotation_tasks {
        string task_uuid PK "任务 UUID"
        string video_uuid FK "关联视频 UUID"
        string assigned_to "分配标注员账号"
        int start_frame "起止帧"
        int end_frame "结束帧"
        string status "任务状态"
    }

    class_labels {
        int label_id PK "自增主键"
        string label_name UNIQUE "类别名称"
        string sam3_prompt "SAM3 检索用 Text Prompt"
        string keypoint_schema "姿态骨架关联 JSON"
    }

    class_tags {
        int tag_id PK "自增主键"
        string tag_name UNIQUE "全局标签名称"
    }
```

---

## 6. 全量 RESTful API 与 SSE 实时接口手册

### 系统与交互配置 API

| API 路由 | 方法 | 功能说明 | 核心输入参数 JSON / Form |
| :--- | :--- | :--- | :--- |
| `/setup` | GET | 渲染向导页面 `setup.html` | 无 |
| `/api/detect_hardware` | GET | 检查 PyTorch 是否支持 CUDA 及显存 | 无 |
| `/api/complete_setup` | POST | 完成向导并持久化初始配置 | `{gpu_device, max_workers, ...}` |
| `/api/settings` | GET/POST | 读取或保存 `settings.json` 系统参数 | 参数字典 |
| `/api/clear_cache` | POST | 清空内存中的 SAM3 与解帧状态缓存 | 无 |

### 视频与标注管理 API

| API 路由 | 方法 | 功能说明 | 核心输入参数 |
| :--- | :--- | :--- | :--- |
| `/uploadVideo` | POST | 上传 MP4 视频并自动开启后台切帧 | `file`, `description`, `annotation_type` |
| `/importFrames` | POST | 批量导入图片文件作为帧序列 | `files[]`, `description` |
| `/retrieveVideoFrames` | POST | 分页/批量获取视频全帧数据 | `{video_uuid}` |
| `/saveFrameAnnotations` | POST | 保存特定帧的高阶 JSON 标注数据 | `{video_uuid, frame_number, annotations_json}` |
| `/storeVideoFrameBboxesText` | POST | 保存特定帧的文本矩形框数据 | `{video_uuid, frame_number, bboxes_text}` |

### AI 智能推导与交互 API

| API 路由 | 方法 | 功能说明 | 核心输入参数 |
| :--- | :--- | :--- | :--- |
| `/samPredict` | POST | SAM2/3 正负点击交互分割预测 | `{video_uuid, frame_number, point_coords}` |
| `/api/sam3_text_predict` | POST | 基于文本 Prompt 执行 SAM3 开放词汇检测 | `{video_uuid, frame_number, text_prompt}` |
| `/lam_predict` | POST | Look-At-Me 全类别 Prompt 并行检测 | `{video_uuid, frame_number, point}` |
| `/api/interpolateBboxes` | POST | BBox 跨帧坐标线性插值 | `{video_uuid, object_id, start_frame, end_frame}` |
| `/api/interpolatePoseKeypoints` | POST | Keypoint 姿态关键点跨帧线性插值 | `{video_uuid, object_id, start_frame_number, end_frame_number}` |
| `/api/gkdt_text_pose_predict` | POST | 基于文本 Prompt + GKDT 自动识别姿态 | `{video_uuid, frame_number, class_label, bbox}` |
| `/api/gkdt_sam_pose_predict` | POST | 基于 SAM 点击 + GKDT 自动识别姿态 | `{video_uuid, frame_number, class_label, point}` |
| `/api/gkdt_sam3_batch_pose_predict` | POST | SAM3 + GKDT 全图多目标姿态识别 | `{video_uuid, frame_number, class_label, confidence}` |
| `/api/clusterClassificationImages` | POST | CLIP/HSV 深度特征提取 + K-Means 聚类 | `{video_uuids, num_clusters, unlabeled_only}` |
| `/api/applyClipZeroShot` | POST | 执行 CLIP 8-Prompt Ensembling 分类 | `{video_uuids, candidate_classes}` |
| `/api/previewAugmentations` | POST | 实时呈现数据增强作用后的 Base64 图片 | `{video_uuid, frame_number, augmentation_options}` |

### 视频跟踪与 SSE 流 API

| API 路由 | 方法 | 功能说明 | 核心输入参数 |
| :--- | :--- | :--- | :--- |
| `/startSam2Tracking` | POST | 开启 SAM2 单流长时跟踪异步任务 | `{video_uuid, start_frame, end_frame, init_bboxes_text}` |
| `/streamSam2Tracking/<uuid>` | GET (SSE)| **Server-Sent Events** 实时向客户端流式推送跟踪进度与框数据 | SSE 数据流 |
| `/stopSam2Tracking` | POST | 请求中断正在执行的 SAM2 跟踪任务 | `{tracker_uuid}` |

### 数据工程与质量审计 API

| API 路由 | 方法 | 功能说明 | 核心输入参数 |
| :--- | :--- | :--- | :--- |
| `/createDataset` | POST | 启动多进程后台数据集解算打平压缩任务 | `{video_uuids, export_format, eval_percent, test_percent}` |
| `/downloadDataset/<uuid>` | GET | 下载解算好的 `.zip` 数据集包 | 无 |
| `/api/datasetAnalysis/<uuid>` | GET | 获取标注密集度、共现率、亮度及框重叠统计 | 无 |
| `/api/datasetAnalysis/<uuid>/consistency_check` | POST | 运行 SAM3 + HSV 色彩双通道数据集审查 | `{enable_color_check: bool}` |

---

## 7. 数据增强参数控制与图像预览全流程

1. **Albumentations 管道**：在 `exporters/` 中，系统将 Keypoints 和 Polygon 顶点打平为一维坐标集 `flat_kpts` 传入变换管道，保持其空间几何拓扑一致。
2. **多进程并发导出 (`multiprocessing.Pool`)**：
   ```python
   with Pool(processes=safe_workers) as pool:
       for result in pool.imap_unordered(process_seg_frame_worker, all_tasks):
           # 增量刷新数据库状态并返回进度
   ```
3. **Mosaic 拼图导出机制**：以 0.5~1.5 随机切割中心对 4 张训练集图像实施重新拼接与归一化缩放。

---

## 8. 系统配置参数 (`settings.json`) 说明

关键可调参数与运行含义：

```json
{
    "initial_setup_done": true,              // 是否已完成首期向导
    "gpu_device": "auto",                    // CUDA 规则 ("auto", "cuda:0", "cpu")
    "sam_model_checkpoint": "sam2.1_t.pt",   // SAM 基础权重 (t/s/b/l)
    "sam_mask_confidence": 0.70,             // 分割掩码掩膜置信度
    "default_confidence": 0.5,               // 全局 AI 推理置信度
    "max_workers": 8,                        // 后台多线程工作池最大线程数
    "max_cache_size": 30,                    // 帧 Backbone 显存 LRU 缓存最大帧数
    "use_autocast": true,                    // 开启 FP16 / BF16 混合精度加速
    "color_confusion_factor": 2.0,           // 颜色偏离警告敏感度
    "consistency_semantic_threshold": 0.3,   // 语义绝绝对下界阈值
    "consistency_confusion_margin": 0.15,    // 混淆判定 Margin 边界
    "enable_sam_model": true,                // SAM 模块总开关
    "enable_feature_extractor": true,        // SAM3 开放词汇检索总开关
    "enable_cls_model": true,                // CLIP 模块总开关
    "enable_pose_model": true                // GKDT 姿态预测总开关
}
```

---

## 9. 环境搭建与原生桌面打包部署

### 安装依赖

```bash
# 激活环境
python -m venv .venv
.venv\Scripts\activate

# 安装 PyTorch (以 CUDA 12.1 为例)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

### 桌面原生打包 (PyInstaller)

项目根目录提供了 `Zero2YoloYard.spec` 配置文件，使用 PyInstaller 可直接将整个程序打包为独立的桌面双击可执行文件：

```bash
pyinstaller Zero2YoloYard.spec
```

打包完成后将在 `dist/` 目录下生成包含独立的 Python 环境、PyWebView 依赖及全套 HTML/JS 静态资源的桌面可执行应用。
