# Zero-to-YOLO-Yard

**Localized, AI-Driven Next-Generation Computer Vision Annotation Workstation**

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-Active_Development-brightgreen.svg)]()
[![FTC](https://img.shields.io/badge/FTC-Team_27570-orange.svg)]()

**Zero-to-YOLO-Yard** is an open-source computer vision data processing platform heavily optimized for local deployment. It is not just an annotation tool, but a complete desktop-grade workstation that covers the entire pipeline: from raw video import and AI-assisted annotation, to deep data quality control and final YOLO dataset export.

Powered by a brand-new architecture combining `pywebview` and `Waitress`, Zero2YoloYard delivers a native desktop application experience right out of the box. Whether you are an FTC (FIRST Tech Challenge) team member, a robotics developer, or a computer vision researcher, this project empowers you to build high-quality training data securely and efficiently. All data and model computations are performed 100% locally on your machine, ensuring absolute data privacy.

---

## ✨ Core Highlights: Redefining the Data Pipeline

### 🚀 Next-Generation AI-Assisted Annotation Engine
Say goodbye to tedious manual bounding boxes and turn the boring annotation process into a joyful exploration:
* **SAM 2.1 Pixel-Level Point-to-Box**: Deeply integrated with Meta's latest SAM 2.1 models (`sam2.1_hiera_t/s/b+/l`). Just one click on the target object, and the AI automatically generates a pixel-perfect bounding box.
* **LAM (Label Assignment Matching) Smart Recommendation**: After you click on an object, the system not only calls SAM to generate the box, but also uses the backend feature library for semantic matching. It automatically recommends the most likely category labels (Top-5), achieving an ultimate "click-and-done" experience.
* **Lightning-Fast Smart Select**: An original, lightweight batch annotation feature. Simply box one or two positive samples, and the AI will find all similar objects in the entire image. Driven by the ultra-fast **MobileNetV3** (Large/Small) and combined with **OpenCV Color Histograms** to filter out distractions, it enables one-click, high-precision batch annotation.
* **Advanced Temporal Object Tracking (SAM 2 Video Predictor)**: 
    * Annotate the first frame and activate high-precision tracking based on the official `SAM2VideoPredictor` with a single click. Supports both interactive real-time feedback and offline high-accuracy batch processing.
    * **Keyframe Linear Interpolation**: For simple linear movements over a long period, simply annotate the start and end frames. The bounding boxes for all intermediate frames will be generated automatically.

### 🕵️‍♂️ Original AI Quality Control
The quality of your dataset determines the upper limit of your model. Zero2YoloYard provides a powerful "Consistency Check" feature:
* **Dual Semantic & Color Verification**: The system automatically extracts high-dimensional semantic features (MobileNetV3) and HSV color features of all annotated targets.
* **Clustering Anomaly Detection**: Utilizing `scikit-learn`'s KMeans clustering algorithm, it deeply analyzes your annotations to pinpoint isolated samples (Outliers) that are "misclassified", "visually abnormal", or "color-conflicted". These are highlighted directly in the UI Gallery, leaving low-quality data nowhere to hide.

### 📊 Deep Dataset Analysis & Online Augmentation
* **Interactive Data Dashboard**: Gain comprehensive insights into your dataset before exporting. Features include category distribution, object density, size and aspect ratio analysis, spatial heatmap, and brightness distribution charts.
* **WYSIWYG Augmentation (Albumentations)**: Configure rotation, cropping, color transformation, and noise strategies directly when creating your dataset. Exclusively supports **Mosaic Real-time Preview**, allowing you to adjust parameters visually on the web page and avoid blind trial-and-error.

### 🧠 Automated Model Pre-annotation
* **Blazing-Fast TFLite Pre-annotation**: Import your existing `.tflite` models (supports float32/uint8) along with label files. The system will automatically traverse video frames to complete the base annotation, leaving you with only minor tweaks to make.

### 💻 Pure Local & Extreme Performance
* **Smart LRU VRAM Management**: The backend features a refactored `LRUCache` mechanism that dynamically manages the CPU/GPU residency states of preprocessed data (SAM Masks & MobileNet Features). This completely eliminates Out-Of-Memory (OOM) issues while ensuring lightning-fast response times.

---

## 🚀 Quick Start

### 1. Environment Setup

Please ensure you have **Python 3.10** and `pip` installed on your system. We highly recommend having an NVIDIA GPU with a properly configured CUDA environment for the best AI inference experience.

**Clone the Project**
```bash
git clone [https://github.com/BlueDarkUP/Zero2YoloYard.git](https://github.com/BlueDarkUP/Zero2YoloYard.git)
cd Zero-to-YOLO-Yard

```

**Install Dependencies**
We strongly advise using a virtual environment to isolate project dependencies.

```bash
# Create and activate virtual environment (Linux/macOS)
python -m venv venv
source venv/bin/activate

# (Windows)
python -m venv venv
.\venv\Scripts\activate

# CRITICAL: If you have an NVIDIA GPU, install PyTorch matching your CUDA version first!
# Example for CUDA 12.6:
# pip install torch==2.7.0 torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

# Install all required project libraries
pip install -r requirements.txt

```

*(Note: Running SAM2 and smart annotation features requires downloading the corresponding weight files and placing them in the `checkpoints/` directory.)*

### 2. Launch the Application

Run the following command in the project root directory:

```bash
python app.py

```

The terminal will print a beautiful ASCII startup screen and warm up the AI models in the background. Once initialized, the system will automatically pop up a responsive, native desktop window. You can start your AI annotation journey immediately!

---

## 📖 Standard Workflow Guide

1. **Upload & Manage Data (Videos)**: Upload `.mp4` videos or import local image folders. The system automatically handles frame extraction and database initialization.
2. **Create Tasks (Manage Tasks)**: Assign different frame ranges to team members for effortless collaborative work.
3. **Multi-Mode Efficient Labeling (Labeling)**:
* Use SAM 2.1 (Point) + LAM for rapid single-object annotation.
* When encountering dense similar targets, draw a box and use *Smart Select* to capture them all in one click.
* In video sequences, annotate the first frame and trigger *SAM2 Track Objects* to automatically label the rest.


4. **AI Quality Control & Analysis (Datasets -> Analyze)**:
* Run the `Consistency Check` for an AI-powered data audit.
* Use the Image Gallery to filter out suspicious high-overlap bounding boxes or extremely small targets, and jump directly to fix them.
* Use the Augmentation Previewer to fine-tune augmentation parameters.


5. **One-Click YOLO Export (Create Dataset)**: Set your Train/Val split, select your desired data augmentation strategies, and generate a `.zip` file ready to be fed directly into frameworks like YOLOv8 for training.

---

## 🛠️ Core Tech Stack

* **Frontend & Client**: `pywebview` (Desktop wrapper), Bootstrap, jQuery, Chart.js
* **Backend Engine**: Flask, Waitress, SQLite
* **AI Vision Foundation Models**:
* Segmentation & Tracking: SAM 2.1 (`build_sam2_video_predictor`, `SAM2ImagePredictor`)
* Lightweight Feature Extraction & LAM: PyTorch `MobileNet_V3_Large` / `MobileNet_V3_Small`
* Traditional Vision Processing: OpenCV, Scikit-image


* **Data Pipelines**: Albumentations (Data Augmentation), Scikit-learn (Clustering for Anomaly Detection)

## 📂 File Structure Overview

* `local_storage/`: Automatically generated local workspace. Safely stores all videos, extracted frames, exported datasets, and imported models.
* `checkpoints/`: Directory for storing `.pt` weight files for AI models like SAM 2.1.
* `ftc_ml.db`: Local SQLite database file. Securely manages all metadata, annotation coordinates, and task assignments.

## 🤝 Acknowledgments & Contributions

This project is a complete localized reconstruction and intelligent, disruptive upgrade of the [FMLTC (FIRST Machine Learning Toolchain)](https://github.com/FIRST-Tech-Challenge/fmltc).

Special thanks to **BlueDarkUP** from **FIRST Tech Challenge Team 27570** for outstanding contributions to the architectural design, core AI algorithm integration, and desktop adaptation of this project.

We warmly welcome contributions of any kind! Whether it's reporting a bug via Issues, suggesting new features, or submitting a Pull Request, let's build the ultimate localized annotation tool for the open-source computer vision community together.

```

```
