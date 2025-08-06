# Zero-to-YOLO Yard IDE - YOLO 全流程开发套件

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![UI Framework](https://img.shields.io/badge/UI-PySide6-orange.svg)](https://www.qt.io/qt-for-python)
[![AI Frameworks](https://img.shields.io/badge/AI-PyTorch%20%7C%20Ultralytics-red.svg)](https://pytorch.org/)

**中文** | [English](README_EN.md) `(待添加)`

你是否曾为目标检测数据集的准备、标注、分析和训练流程繁琐而烦恼？**Zero-to-YOLO Yard IDE** (Z2Y-Yard) 旨在解决这一痛点，它提供了一个图形化的、一站式的解决方案，将您从零散的脚本和复杂的配置中解放出来，专注于数据和模型本身。

本工具最初是一个数据增强工具，现已演变为一个功能完备的IDE，覆盖了YOLO等目标检测项目的完整生命周期。

## ✨ 主要功能

Z2Y-Yard 将数据处理的各个环节无缝集成在一个统一的界面中：

- **📦 多格式数据集支持**
  - **无缝读写**：支持 **YOLO**、**COCO**、**Pascal VOC** 和 **TFRecord** 等主流数据集格式。
  - **格式转换**：轻松将您的数据集从一种格式转换为另一种。
  - **智能检测**：自动检测您导入的数据集格式。

- **✏️ 强大的标注与编辑工具**
  - **手动标注**：高效创建、修改和删除边界框。
  - **AI 辅助标注 (SAM)**：集成 **Segment-Anything Model**，只需点击一下，即可自动生成高质量的边界框。
  - **视频标注**：
    - 从视频文件导入帧，快速创建图像序列数据集。
    - 使用**多目标追踪算法** (如 CSRT, KCF) 对视频进行半自动标注，大幅提升效率。

- **📊 交互式数据洞察仪表盘**
  - **多维度可视化**：通过图表直观展示类别分布、图像尺寸、标注密度、边界框尺寸分布等关键信息。
  - **交互式筛选**：直接在图表上点击或框选，即可在主文件列表中筛选出感兴趣的数据子集。
  - **嵌入空间分析**：利用 `UMAP` 或 `t-SNE` 对图像特征进行降维可视化，发现数据中的潜在聚类或异常点。

- **🚀 丰富的在线数据增强**
  - **海量增强算子**：基于 `Albumentations` 库，提供几何、色彩、模糊、噪声、遮挡等数十种增强方法。
  - **高级复合增强**：支持 **Mosaic** 等高级增强策略。
  - **可视化配置**：所有参数均可通过滑块和输入框实时调节。
  - **预设管理**：保存和加载您的增强配置，方便复用。

- **🛠️ 实用的数据集管理工具**
  - **智能数据集分割**：一键将数据集按比例（如 8:1:1）分割为训练集、验证集和测试集，支持**随机**或**分层**采样。
  - **数据集健康检查**：自动扫描并报告缺失标签、空标签文件、有标签但无对应图像等常见问题。

- **🤖 AI 模型驱动工作流**
  - **模型诊断**：使用您自己的YOLOv8模型对数据集进行预标注或诊断，找出**潜在漏标 (Misses)**、**误检 (False Positives)** 和 **类别错误 (Mismatches)**。
  - **集成模型训练**：在IDE内直接配置并启动 **YOLOv8 模型训练**，无需编写任何代码。实时日志输出，训练完成后自动返回最佳模型路径。

## 📦 安装与启动

### 1. 先决条件
- Python 3.8 或更高版本
- `git` (用于克隆仓库)

### 2. 安装步骤

**a. 克隆仓库**
```bash
git clone https://github.com/BlueDarkUP/Zero2YoloYard.git
cd zero-to-yolo-yard-ide
```

**b. (推荐) 创建虚拟环境**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**c. 安装依赖**
我们提供了 `requirements.txt` 文件来简化安装。核心AI功能依赖于PyTorch，请确保您的环境中CUDA（如果可用）版本与PyTorch兼容。

```bash
pip install -r requirements.txt
```
**注意**:
- `requirements.txt` 文件内容大致如下，您可以根据需要进行调整：
  ```
  # GUI
  PySide6

  # Core & Image Processing
  numpy
  opencv-python-headless
  opencv-contrib-python-headless  # For trackers
  Pillow
  PyYAML
  matplotlib

  # AI & ML
  torch
  torchvision
  ultralytics
  segment-anything
  albumentations
  umap-learn
  scikit-learn
  sentence-transformers

  # Optional for TFRecord support
  # tensorflow
  ```
- 如果您不需要 **TFRecord** 格式的支持，可以忽略 `tensorflow` 的安装。
- 如果您不需要 **AI辅助** 功能（如SAM、诊断、训练），可以注释掉 `torch`, `ultralytics`, `segment-anything` 等相关库。

**d. 下载AI模型权重 (可选，但推荐)**
为了使用AI辅助功能，您需要预先下载模型权重文件，并将它们放置在项目根目录下：
- **Segment-Anything Model (SAM)**: 下载 [SAM ViT-H 模型](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 并重命名为 `sam_vit_h.pth`。

### 3. 启动应用
一切就绪后，运行 `main_app.py` 即可启动程序：
```bash
python main_app.py
```

## 🚀 快速开始

1.  **选择项目目录**: 启动程序后，点击“浏览…”选择一个数据集的根目录。如果是一个空目录，程序会将其视为一个新项目。
2.  **分析数据集**: 如果您导入的是一个现有数据集，点击“📊 分析”按钮，打开交互式仪表盘，洞察数据分布。
3.  **标注与编辑**: 点击“✏️ 标注”按钮进入标注界面。
    - 使用 `Q` 键切换到手动标注模式。
    - 使用 `E` 键切换到SAM模式，点击物体即可自动生成标注。
    - 导入视频后，选择一个框，使用 `D` 键逐帧跟踪。
4.  **数据增强**: 在主界面配置您需要的增强策略，然后点击“🚀 开始处理”生成增强后的数据集。
5.  **分割数据集**: 点击“🪓 数据集分割工具”，配置比例后，一键生成 `train/valid/test` 子集。
6.  **开始训练**: （待添加训练UI后）在训练面板配置参数，直接开始训练您的YOLOv8模型。

## 🤝 贡献

我们热烈欢迎任何形式的贡献！无论是报告Bug、提出功能建议，还是提交代码，都对项目的发展至关重要。

- **报告问题**: 如果您遇到问题，请通过 [GitHub Issues](https://github.com/your-username/zero-to-yolo-yard-ide/issues) 提交。
- **功能建议**: 有任何好点子，也欢迎通过 Issues 提出。
- **提交代码**:
  1. Fork 本仓库。
  2. 创建一个新的分支 (`git checkout -b feature/your-feature-name`)。
  3. 提交您的修改 (`git commit -am 'Add some feature'`)。
  4. 推送到您的分支 (`git push origin feature/your-feature-name`)。
  5. 创建一个新的 Pull Request。

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源。

## 🙏 致谢

本项目构建于以下优秀的开源库之上：
- [PySide6](https://www.qt.io/qt-for-python)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Ultralytics (YOLOv8)](https://github.com/ultralytics/ultralytics)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [Matplotlib](https://matplotlib.org/)
- 以及其他所有在 `requirements.txt` 中列出的库。

---
