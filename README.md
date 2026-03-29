# Zero-to-YOLO-Yard

**本地化、AI 驱动的下一代计算机视觉标注工作站**

**Zero-to-YOLO-Yard** 是一个专为本地化部署深度优化的开源计算机视觉数据处理平台。它不仅是一个标注工具，更是一个为您提供从原始视频导入、AI 辅助标注、数据深度质检到最终 YOLO 数据集导出的全流程桌面级工作站。

结合了 `pywebview` 与 `Waitress` 的全新架构，Zero2YoloYard 启动即享原生桌面应用体验。无论您是 FTC (FIRST Tech Challenge) 参赛队员、机器人开发者，还是计算机视觉研究者，本项目都能助您安全、高效地构建高质量的训练数据，且所有数据与模型运算均 100% 在您的本地计算机上完成，确保绝对的数据隐私。

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-迭代中-brightgreen.svg)]()

---

## ✨ 核心亮点：不止于标注，重塑数据流

### 🚀 新一代 AI 辅助标注引擎

彻底告别繁琐的手动拉框，将枯燥的标注工作转化为充满探索乐趣的体验：

* **SAM 2.1 像素级点选成框**: 深度集成 Meta 最新 SAM 2.1 模型 (`sam2.1_hiera_t/s/b+/l`)。只需在目标物体上轻轻一点，即可自动生成极高精度的像素级边界框。
* **LAM (Label Assignment Matching) 智能推荐**: 点击目标后，系统不仅调用 SAM 生成边界框，还会利用后台特征库进行语义匹配，自动为您推荐最匹配的类别标签（Top-5 建议），实现“点击即完成”的极致体验。
* **极速智能选择 (Smart Select)**: 独创的轻量化批量标注功能。只需框选一两个正例样本，AI 就能在整张图中找出所有相似对象。底层由极速的 **MobileNetV3** (Large/Small) 驱动，并结合 **OpenCV 颜色直方图 (Color Histogram)** 过滤干扰项，实现一键高精度批量标注。
* **高级时序对象跟踪 (SAM 2 Video Predictor)**:
* 在视频首帧完成标注后，一键开启基于官方 `SAM2VideoPredictor` 的高精度跟踪，支持实时反馈与离线高精批处理。
* **关键帧线性插值**: 针对长时段的简单线性运动，只需标注起点和终点，中间所有帧的边界框均可一键自动生成。



### 🕵️‍♂️ 独创的 AI 数据质检 (AI Quality Control)

数据集的质量决定了模型的上限。Zero2YoloYard 提供了强大的“Consistency Check”一致性审查功能：

* **语义与色彩双重校验**: 系统自动提取所有已标注目标的 MobileNetV3 高维语义特征与 HSV 色彩特征。
* **聚类异常检测**: 利用 `scikit-learn` 的 KMeans 聚类算法，自动对您的标注进行深度分析，精准揪出“标错类别”、“外观异常”或“颜色冲突”的孤立样本（Outliers），并直接在 UI 画廊中为您高亮显示，让低质量数据无处遁形。

### 📊 深度数据集分析与在线增强

* **交互式数据体检**: 导出前全面洞察您的数据集。提供类别分布、目标密度、尺寸与宽高比分析、空间位置热力图以及亮度分布等图表。
* **强大的所见即所得增强 (Albumentations)**: 在创建数据集时直接配置旋转、裁剪、色彩变换、噪声等策略。独家支持 **Mosaic 实时预览**，在网页端直观调整参数，告别盲目试错。

### 🧠 自动化模型预标注

* **TFLite 极速预打标**: 直接导入您已有的 `.tflite` 模型（支持 float32/uint8）与标签文件，自动遍历视频帧完成基础标注，您只需在此基础上进行微调。

### 💻 纯本地化极致性能

* **智能 LRU 显存管理**: 后端重构了 `LRUCache` 缓存机制，动态管理预处理数据 (SAM Masks & MobileNet Features) 的 CPU/GPU 驻留状态。在保证极速响应的同时，彻底告别显存溢出 (OOM) 烦恼。

---

## 🚀 快速上手

### 1. 环境配置

请确保您的系统中已安装 **Python 3.10** 及 `pip`。强烈推荐拥有 NVIDIA 显卡并配置 CUDA 环境以获得最佳的 AI 推理体验。

**克隆项目**

```bash
git clone https://github.com/BlueDarkUP/Zero2YoloYard.git
cd Zero-to-YOLO-Yard

```

**安装依赖**
我们强烈建议使用虚拟环境来隔离项目依赖。

```bash
# 创建并激活虚拟环境 (Linux/macOS)
python -m venv venv
source venv/bin/activate

# (Windows)
python -m venv venv
.\venv\Scripts\activate

# 核心提示：如果拥有 NVIDIA 显卡，请先根据您的 CUDA 版本安装 PyTorch！
# 例如 CUDA 12.6 环境:
# pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装项目所有必需的库
pip install -r requirements.txt

```

*(注：运行 SAM2 及智能标注功能需要下载相应的权重文件放置于 `checkpoints/` 目录下)*

### 2. 启动应用

在项目根目录下运行以下命令：

```bash
python app.py

```

终端将打印精美的 ASCII 启动画面，并在后台预热 AI 模型。启动完成后，系统会自动弹出一个分辨率自适应的原生桌面窗口，您可以立即开始您的 AI 标注之旅！

---

## 📖 标准工作流指南

1. **上传与管理数据 (Videos)**: 上传 `.mp4` 视频或导入本地图片文件夹。系统会自动完成抽帧与数据库初始化。
2. **创建标注任务 (Manage Tasks)**: 为团队成员分配不同的标注帧范围，轻松实现协同作业。
3. **多模式高效标注 (Labeling)**:
* 利用 SAM 2.1 (Point) + LAM 进行单体快速标注。
* 遇到同类密集目标，框选后使用 Smart Select 一键批量捕获。
* 在视频序列中，标注首帧后启动 SAM2 Track Objects 自动完成后续帧标注。


4. **AI 质检与分析 (Datasets -> Analyze)**:
* 运行 `Consistency Check` 进行 AI 数据审查。
* 通过 Image Gallery 筛选可疑的高重叠度包围盒或极小目标，并快速跳转修正。
* 使用 Augmentation Previewer 调试增强参数。


5. **一键导出 YOLO 数据集 (Create Dataset)**: 设定训练集/验证集比例，勾选所需的数据增强策略，生成 `.zip` 压缩包，直接送入 YOLOv8 等框架进行训练。

---

## 🛠️ 技术栈核心

* **前端与客户端**: `pywebview` (桌面化封装), Bootstrap, jQuery, Chart.js
* **后端引擎**: Flask, Waitress, SQLite
* **AI 大模型与视觉库**:
* 分割与时序跟踪: SAM 2.1 (`build_sam2_video_predictor`, `SAM2ImagePredictor`)
* 轻量级特征提取与 LAM: PyTorch `MobileNet_V3_Large` / `MobileNet_V3_Small`
* 传统视觉处理: OpenCV, Scikit-image


* **数据流管道**: Albumentations (数据增强), Scikit-learn (异常检测聚类)

## 📂 文件结构说明

* `local_storage/`: 自动生成的本地工作区，安全存储所有视频、帧图片、导出的数据集和导入的模型。
* `checkpoints/`: 存放 SAM 2.1 等 AI 模型的 `.pt` 权重文件。
* `ftc_ml.db`: SQLite 本地数据库文件，安全管理所有元数据、标注坐标和任务分配信息。

## 🤝 贡献与致谢

本项目是对 [FMLTC (FIRST Machine Learning Toolchain)](https://github.com/FIRST-Tech-Challenge/fmltc) 的一次彻底本地化重构与智能化颠覆性升级。

特别感谢来自 **FIRST Tech Challenge Team 27570** 的 **BlueDarkUP** 在本项目架构设计、核心 AI 算法集成及桌面化适配中的卓越贡献。

我们极其欢迎任何形式的贡献！无论是提交 Issue 报告 Bug、提供新功能建议，还是提交 Pull Request，让我们共同为开源计算机视觉社区打造最极致的本地化标注工具。
