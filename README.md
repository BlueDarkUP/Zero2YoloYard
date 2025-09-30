# Zero-to-YOLO-Yard: 本地化、AI驱动的下一代计算机视觉标注工具
---
<img width="1403" height="769" alt="image" src="https://github.com/user-attachments/assets/e2fab11b-6c08-4771-a752-5e0f81355dc3" />

---

**Zero-to-YOLO-Yard** 是一个专为本地化部署而深度优化的开源工具，旨在为您提供从原始视频/图像到可训练数据集的全流程解决方案。它集成了最前沿的 AI 技术，将繁琐的标注工作转变为简单、高效、甚至充满探索乐趣的体验。无论您是机器人开发者、无人机爱好者还是计算机视觉研究者，此工具都能极大地加速您的数据处理流程，且所有数据都安全地保留在您自己的计算机上。

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-迭代中-brightgreen.svg)]()


---

## ✨ 核心亮点：不止于标注

- **🚀 新一代 AI 辅助标注引擎**
    - **SAM 2.1 点选成框**: 集成Meta最新的SAM 2.1模型，只需在目标物体上轻轻一点，即可自动生成像素级精确的边界框。
    - **智能选择 (Smart Select)**: 独创的“智能选择”功能，由 **DINOv2** 强大的特征提取能力驱动。您只需框选一两个“正例”样本（甚至可以圈出“反例”来排除干扰），AI 就能在整张图中找出所有相似对象，实现一键批量标注。
    - **数据集驱动查找**: 充分利用您已有的标注数据！选择一个类别，AI 会学习该类别在整个数据集中的特征，并自动在新图像中找出所有潜在目标。
    - **高级对象跟踪**: 在视频首帧完成标注后，一键开启自动跟踪。提供两种模式：
        1.  **交互模式**: 实时跟踪并反馈，可随时暂停、修正，适合复杂多变的场景。
        2.  **高精度批处理**: 调用官方 `SAM2VideoPredictor`，一次性处理视频片段，在稳定场景下获得更高质量的跟踪结果。
    - **关键帧插值**: 对于长时段的简单运动，只需标注起点和终点两帧，中间所有帧的边界框将自动线性插值生成。

- **📊 深度数据集分析与可视化**
    - 在导出前，对您的数据集进行一次全面“体检”。通过交互式图表洞察：
        - **类别分布**: 检查是否存在数据不均衡问题。
        - **目标密度**: 分析每张图的目标数量分布。
        - **尺寸与宽高比**: 发现异常尺寸或比例的目标。
        - **空间位置热力图**: 查看目标在图像中的常见位置。
    - **智能筛选与浏览**: 内置“图像画廊”，可一键筛选出数据“异常点”，如面积最大/最小的目标、重叠度过高的重复标注等，帮助您快速定位并修正标注错误。

- **⚙️ 强大的在线数据增强**
    - 在创建数据集时，直接在网页上配置丰富的数据增强策略（旋转、裁剪、色彩变换、噪声、Cutout等）。
    - **所见即所得的增强预览器**: 实时预览增强效果，直观调整参数，确保增强策略符合您的预期，无需反复试错。

- **📦 完整的工作流闭环**
    - **多源数据导入**: 支持上传 `.mp4` 视频文件，或直接导入已有的图片文件夹。
    - **任务协同管理**: 为不同成员分配不同的标注帧范围，轻松实现团队协作。
    - **一键导出YOLO格式**: 所有标注数据可一键打包为与YOLOv8等主流训练框架兼容的 `.zip` 数据集。

- **💻 纯本地化，安全私密**
    - 无需联网，无需云服务，所有数据和模型运算均在您的本地计算机完成，确保数据绝对安全。

---

## 🚀 快速上手

### 1. 环境配置

请确保您的系统中已安装 **Python 3.10** 及 `pip`。

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

# 安装所有必需的库
# 如果您有NVIDIA显卡，建议先根据您的CUDA版本安装对应的PyTorch
# PyTorch官网: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### 2. 启动应用

在项目根目录下运行以下命令启动 Web 服务器：

```bash
python app.py
```

服务器成功启动后，您将在终端看到类似输出：
```
INFO:waitress:Serving on http://127.0.0.1:5000
```

现在，打开您的浏览器并访问 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** 即可开始您的 AI 标注之旅！

---

## 📖 工作流指南

### 步骤 1: 上传与管理数据

1.  在 **"Videos"** 选项卡中，点击 **"Upload Video"** 上传视频，或点击视频列表中的 **"Import"** 按钮（图标为 <i class="bi bi-images"></i>）导入本地图片文件夹。
2.  系统会自动处理数据，状态变为 `READY` 后即可开始下一步。

https://github.com/user-attachments/assets/11147a8c-e949-402a-ac81-0b914f7f47b5

### 步骤 2: 创建标注任务

1.  点击视频旁的 **"Manage Tasks"** <i class="bi bi-card-checklist"></i> 按钮。
2.  为任务分配一个负责人名称（如 `Alice`），并指定他/她需要标注的 **起始帧** 和 **结束帧**。

https://github.com/user-attachments/assets/b4a97574-9cce-4ee4-ba4a-dcbd8115c5a5

### 步骤 3: 高效标注

进入标注界面后，您可以组合使用以下多种方式，选择最高效的工具来完成工作：

- **基础操作**:
    - 在右侧 **"Classes"** 面板中创建或选择一个类别。
    - **手动绘制**: 按住鼠标左键拖拽即可绘制矩形框。
    - **快捷键**: `S` 保存, `A`/`D` 前后翻页, `Delete` 删除选中框, `Ctrl+Z` 撤销。

https://github.com/user-attachments/assets/00eaf98a-7e1c-47e9-957d-9e135a4024e1

- **AI 辅助**:
    1.  **点选标注**: 点击 **"Enable SAM (Point)"** <i class="bi bi-magic"></i>，然后在目标上单击，AI将自动为您生成边界框。

    https://github.com/user-attachments/assets/131f2b62-da8f-4d81-8d9f-7d32208c29fd

    2.  **智能选择**:
        - 点击 **"Enable Smart Select"** <i class="bi bi-stars"></i> 激活此模式。
        - 默认处于 **"Positive Sample"** 模式，绘制一两个您想找的目标。
        - （可选）切换到 **"Negative Sample"** 模式，框出您不想选择的背景或干扰物。
        - 点击 **"Find Similar Objects"** <i class="bi bi-search"></i>，AI将展示所有找到的相似目标。
        - 选择一个类别，然后点击蓝色的预览框即可将其采纳为正式标注。

    https://github.com/user-attachments/assets/e8d0bb47-3396-49dc-8cf3-1cba531f7c8f

    3.  **自动跟踪**:
        - 在视频的任意一帧，完成对所有目标的标注。
        - 点击 **"Track Objects with SAM2"** <i class="bi bi-play-circle"></i>。
        - 在弹出的窗口中选择 **"Interactive Tracking"** (实时修正) 或 **"High-Accuracy Batch Mode"** (更高质量，离线处理)。
        - 系统将自动处理后续帧。您可以在 **"Review Mode"** 中检查、修正并批量保存结果。

    https://github.com/user-attachments/assets/a5bb6cb5-5bf5-4648-b52e-fbf45e0d2e35

### 步骤 4: 分析与洞察 (新功能!)

1.  标注完成后，在主界面的 **"Datasets"** 选项卡创建一个数据集，并关联已标注的视频。
2.  当数据集状态变为 `READY` 后，点击 **"Analyze"** <i class="bi bi-bar-chart-line"></i> 按钮。
3.  在分析页面，您可以：
    - 查看各类统计图表，了解数据质量。
    - 使用 **"Augmentation Previewer"** 实时调试数据增强效果。
    - 在 **"Image Gallery"** 中使用筛选器快速找到并跳转到有问题的标注进行修正。


    https://github.com/user-attachments/assets/287ec74e-3a69-4504-bfe7-12f313540400


### 步骤 5: 创建与导出数据集

1.  在 **"Datasets"** 选项卡，点击 **"Create Dataset"**。
2.  选择需要打包的视频，设置训练/验证/测试集比例。
3.  **（可选）** 展开并启用 **"Data Augmentation Options"**，配置您需要的增强策略。
4.  创建成功后，点击 **"Download"** <i class="bi bi-download"></i> 即可获取用于模型训练的 YOLO 格式 `.zip` 文件。

> **提示**: 数据集一旦创建，其内容便已固定。如果您后续修改了视频标注，需要重新创建一个新的数据集版本以包含这些更新。

---

## 🛠️ 技术栈核心

- **后端**: Flask, Waitress
- **数据库**: SQLite
- **AI 模型**:
    - **分割与跟踪**: [Ultralytics (YOLOv8-SAM)](https://github.com/ultralytics/ultralytics) & [SAM 2.1](https://github.com/facebookresearch/segment-anything)
    - **特征提取 (智能选择)**: [DINOv2](https://github.com/facebookresearch/dinov2)
- **数据增强**: Albumentations
- **前端**: Bootstrap, jQuery, Chart.js

## 📂 文件结构

-   **`local_storage/`**: 存储所有用户数据，包括视频、帧、数据集和模型。
-   **`checkpoints/`**: 存放SAM等AI模型的权重文件（需自行下载）。
-   **`ftc_ml.db`**: SQLite 数据库文件，管理所有元数据，如项目描述、标注信息、任务等。

## 🤝 贡献与致谢

本项目是对 [FMLTC (FIRST Machine Learning Toolchain)](https://github.com/FIRST-Tech-Challenge/fmltc) 的一次重大功能扩展和本地化重构。

特别感谢 **BlueDarkUP** 在项目开发中的卓越贡献。

我们欢迎任何形式的贡献，无论是功能建议、代码提交还是问题反馈。请通过 Pull Request 或 Issues 与我们交流！
