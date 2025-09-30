# Zero-to-YOLO-Yard: A Localized, AI-Powered, Next-Generation Computer Vision Annotation Tool
<img width="1629" height="527" alt="image" src="https://github.com/user-attachments/assets/4a4ec469-a1b9-48e4-8acf-2854d61fbc72" />

**Zero-to-YOLO-Yard** is an open-source tool deeply optimized for local deployment, designed to provide you with an end-to-end solution from raw video/images to a trainable dataset. It integrates cutting-edge AI technology to transform the tedious work of annotation into a simple, efficient, and even fun exploratory experience. Whether you are a robotics developer, a drone enthusiast, or a computer vision researcher, this tool will significantly accelerate your data processing workflow, with all data remaining securely on your own computer.

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-Iterating-brightgreen.svg)]()

---

## ✨ Core Highlights: More Than Just Annotation

- **🚀 Next-Generation AI-Assisted Annotation Engine**
    - **SAM 2.1 Point-to-Box**: Integrates Meta's latest SAM 2.1 model. A simple click on the target object automatically generates a pixel-perfect bounding box.
    - **Smart Select**: An original "Smart Select" feature powered by the robust feature extraction capabilities of **DINOv2**. Simply box one or two "positive" examples (you can even circle "negative" examples to exclude them), and the AI will find all similar objects in the entire image, enabling one-click batch annotation.
    - **Dataset-Driven Search**: Fully leverage your existing annotated data! Select a class, and the AI will learn its features across the entire dataset and automatically identify all potential targets in new images.
    - **Advanced Object Tracking**: After annotating the first frame of a video, enable automatic tracking with a single click. Two modes are available:
        1.  **Interactive Mode**: Real-time tracking and feedback, allowing you to pause and correct at any time, ideal for complex and dynamic scenes.
        2.  **High-Accuracy Batch Processing**: Utilizes the official `SAM2VideoPredictor` to process video clips in one go, achieving higher quality tracking results in stable scenes.
    - **Keyframe Interpolation**: For long-duration, simple movements, just annotate the start and end frames. The bounding boxes for all intermediate frames will be automatically generated via linear interpolation.

- **📊 In-depth Dataset Analysis and Visualization**
    - Perform a comprehensive "health check" on your dataset before exporting. Gain insights through interactive charts:
        - **Class Distribution**: Check for data imbalance issues.
        - **Object Density**: Analyze the distribution of the number of objects per image.
        - **Size and Aspect Ratio**: Discover objects with abnormal sizes or proportions.
        - **Spatial Heatmap**: See the common locations of objects within the images.
    - **Smart Filtering and Browsing**: The built-in "Image Gallery" allows you to filter for data "outliers" with one click, such as objects with the largest/smallest area or highly overlapping duplicate annotations, helping you quickly locate and correct labeling errors.

- **⚙️ Powerful Online Data Augmentation**
    - Configure a rich set of data augmentation strategies (rotation, cropping, color transformation, noise, Cutout, etc.) directly on the web interface when creating your dataset.
    - **WYSIWYG Augmentation Previewer**: See a live preview of augmentation effects and adjust parameters intuitively, ensuring your augmentation strategy meets expectations without trial and error.

- **📦 Complete Workflow Loop**
    - **Multi-Source Data Import**: Supports uploading `.mp4` video files or directly importing existing image folders.
    - **Collaborative Task Management**: Assign different frame ranges to different team members for easy team collaboration.
    - **One-Click Export to YOLO Format**: All annotated data can be packaged into a `.zip` dataset compatible with mainstream training frameworks like YOLOv8 with a single click.

- **💻 Purely Local, Secure, and Private**
    - No internet connection or cloud services required. All data and model computations are performed on your local machine, ensuring absolute data security.

---

## 🚀 Quick Start

### 1. Environment Setup

Please ensure you have **Python 3.10** and `pip` installed on your system.

**Clone the Project**
```bash
git clone https://github.com/BlueDarkUP/Zero2YoloYard.git
cd Zero-to-YOLO-Yard
```

**Install Dependencies**
We strongly recommend using a virtual environment to isolate project dependencies.

```bash
# Create and activate a virtual environment (Linux/macOS)
python -m venv venv
source venv/bin/activate

# (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install all required libraries
# If you have an NVIDIA GPU, it is recommended to first install the corresponding PyTorch version for your CUDA version
# PyTorch official website: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### 2. Launch the Application

Run the following command in the project's root directory to start the web server:

```bash
python app.py
```

Once the server starts successfully, you will see output in the terminal similar to this:
```
INFO:waitress:Serving on http://127.0.0.1:5000
```

Now, open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)** to begin your AI annotation journey!

---

## 📖 Workflow Guide

### Step 1: Upload and Manage Data

1.  In the **"Videos"** tab, click **"Upload Video"** to upload a video, or click the **"Import"** button (<i class="bi bi-images"></i> icon) in the video list to import a local image folder.
2.  The system will automatically process the data. Once the status changes to `READY`, you can proceed to the next step.

https://github.com/user-attachments/assets/11147a8c-e949-402a-ac81-0b914f7f47b5

### Step 2: Create Annotation Tasks

1.  Click the **"Manage Tasks"** <i class="bi bi-card-checklist"></i> button next to the video.
2.  Assign a name to the person responsible for the task (e.g., `Alice`) and specify the **Start Frame** and **End Frame** they need to annotate.

https://github.com/user-attachments/assets/b4a97574-9cce-4ee4-ba4a-dcbd8115c5a5

### Step 3: Efficient Annotation

Once in the annotation interface, you can combine the following methods, choosing the most efficient tool for the job:

- **Basic Operations**:
    - Create or select a class in the **"Classes"** panel on the right.
    - **Manual Drawing**: Click and drag the left mouse button to draw a rectangle.
    - **Hotkeys**: `S` to Save, `A`/`D` for Previous/Next Page, `Delete` to remove the selected box, `Ctrl+Z` to Undo.

https://github.com/user-attachments/assets/00eaf98a-7e1c-47e9-957d-9e135a4024e1

- **AI-Assisted**:
    1.  **Point-and-Click Annotation**: Click **"Enable SAM (Point)"** <i class="bi bi-magic"></i>, then click on the target object, and the AI will automatically generate a bounding box for you.

    https://github.com/user-attachments/assets/131f2b62-da8f-4d81-8d9f-7d32208c29fd

    2.  **Smart Select**:
        - Click **"Enable Smart Select"** <i class="bi bi-stars"></i> to activate this mode.
        - By default, it is in **"Positive Sample"** mode. Draw a box around one or two examples of the object you want to find.
        - (Optional) Switch to **"Negative Sample"** mode to box out backgrounds or distractors you don't want to select.
        - Click **"Find Similar Objects"** <i class="bi bi-search"></i>, and the AI will display all similar objects found.
        - Select a class, then click on the blue preview boxes to accept them as official annotations.

    https://github.com/user-attachments/assets/e8d0bb47-3396-49dc-8cf3-1cba531f7c8f

    3.  **Auto-Tracking**:
        - In any frame of the video, finish annotating all target objects.
        - Click **"Track Objects with SAM2"** <i class="bi bi-play-circle"></i>.
        - In the pop-up window, choose **"Interactive Tracking"** (for real-time correction) or **"High-Accuracy Batch Mode"** (for higher quality, offline processing).
        - The system will automatically process the subsequent frames. You can review, correct, and bulk-save the results in **"Review Mode"**.

    https://github.com/user-attachments/assets/a5bb6cb5-5bf5-4648-b52e-fbf45e0d2e35

### Step 4: Analyze and Gain Insights (New Feature!)

1.  After annotation is complete, create a dataset in the **"Datasets"** tab on the main interface and link the annotated videos.
2.  Once the dataset status changes to `READY`, click the **"Analyze"** <i class="bi bi-bar-chart-line"></i> button.
3.  On the analysis page, you can:
    - View various statistical charts to understand data quality.
    - Use the **"Augmentation Previewer"** to debug data augmentation effects in real-time.
    - Use filters in the **"Image Gallery"** to quickly find and navigate to problematic annotations for correction.

    https://github.com/user-attachments/assets/287ec74e-3a69-4504-bfe7-12f313540400

### Step 5: Create and Export Dataset

1.  In the **"Datasets"** tab, click **"Create Dataset"**.
2.  Select the videos to be packaged and set the train/validation/test split ratios.
3.  **(Optional)** Expand and enable **"Data Augmentation Options"** to configure your desired augmentation strategies.
4.  After successful creation, click **"Download"** <i class="bi bi-download"></i> to get the YOLO format `.zip` file ready for model training.

> **Tip**: Once a dataset is created, its contents are fixed. If you modify the video annotations later, you will need to create a new dataset version to include these updates.

---

## 🛠️ Core Tech Stack

- **Backend**: Flask, Waitress
- **Database**: SQLite
- **AI Models**:
    - **Segmentation & Tracking**: [Ultralytics (YOLOv8-SAM)](https://github.com/ultralytics/ultralytics) & [SAM 2.1](https://github.com/facebookresearch/segment-anything)
    - **Feature Extraction (Smart Select)**: [DINOv2](https://github.com/facebookresearch/dinov2)
- **Data Augmentation**: Albumentations
- **Frontend**: Bootstrap, jQuery, Chart.js

## 📂 File Structure

-   **`local_storage/`**: Stores all user data, including videos, frames, datasets, and models.
-   **`checkpoints/`**: Stores the weight files for AI models like SAM (needs to be downloaded by the user).
-   **`ftc_ml.db`**: The SQLite database file that manages all metadata, such as project descriptions, annotation information, tasks, etc.

## 🤝 Contribution and Acknowledgements

This project is a major functional extension and localization refactoring of [FMLTC (FIRST Machine Learning Toolchain)](https://github.com/FIRST-Tech-Challenge/fmltc).

Special thanks to **BlueDarkUP** for their outstanding contributions to the project's development.

We welcome contributions of any kind, whether it's feature suggestions, code submissions, or bug reports. Please feel free to communicate with us via Pull Requests or Issues