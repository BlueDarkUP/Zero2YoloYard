### Before you read: Already have a dataset?
- #### Easily convert YOLOv8 format datasets to TFRecord using [Yolo2TFRecord](https://github.com/BlueDarkUP/Yolo2TFRecord)
- #### Easily build a simple pipeline from environment setup to model export using [FTC-Easy-TFLITE](https://github.com/BlueDarkUP/FTC-Easy-TFLITE)

# Zero-to-YOLO-Yard: A Localized Machine Learning Annotation Tool

*   [中文 README](README.md)

**Zero-to-YOLO-Yard** is a deeply customized version of the [FMLTC (FIRST Machine Learning Toolchain)](https://github.com/FIRST-Tech-Challenge/fmltc), specifically designed to run efficiently on your local machine without any cloud dependencies. It focuses on providing a complete solution from video to dataset, making it an ideal local data processing tool for robotics, drones, or other computer vision projects.

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

---

## ✨ Core Features

- **🎬 Video Management**: Easily upload and manage your `.mp4` or `.mov` video files.
- **🖼️ Smart Frame Extraction**: Automatically decompose videos into image frames, or directly import existing image folders.
- **✍️ Precise Image Annotation**: Draw bounding boxes and assign class labels on image frames through an intuitive interface.
- **🤖 AI-Assisted Annotation and Tracking**:
    - **SAM 2.1 Integration**: Utilize the [Segment Anything Model 2.1](https://segment-anything.com/) to automatically generate high-quality bounding boxes with a simple click.
    - **Automatic Object Tracking**: Annotate an object in one frame, and it will be automatically tracked through all subsequent frames, significantly boosting efficiency.
- **📦 One-Click Dataset Export**: Export labeled frames in YOLO format, packaged as a `.zip` file, ready for direct use in model training.
- **🧠 Model Management**: Supports importing and managing your `.tflite` models trained on other platforms.

## 🚀 Quick Start

### 1. Environment Setup

Before starting, ensure you have **Python 3.10** installed on your system.

**Clone the Project**
```bash
git clone https://github.com/BlueDarkUP/Zero2YoloYard.git
cd Zero-to-YOLO-Yard
```

**Install Dependencies**
We recommend using a virtual environment to manage project dependencies.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`

# Install all required libraries
pip install -r requirements.txt
```

### 2. Start the Application

Once everything is set up, run the following command in the project's root directory to start the web server:

```bash
python app.py
```

After the server starts successfully, you will see the following output in your terminal:
```
 * Running on http://127.0.0.1:5000
```

Now, open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to begin!

---

## 📖 Workflow Guide

### Step 1: Upload Data

1.  Navigate to the **"Videos"** tab.
2.  Click **"Upload Video"**, select your video file, and assign it a project name.
3.  The system will automatically start processing the video, and the status will change from `UPLOADING` -> `EXTRACTING` -> `READY`.
4.  You can also click **"Import Frame"** to directly import a folder of images as a dataset.

![img.png](assets/img.png)

### Step 2: Assign Labeling Tasks

For team collaboration, you can clearly define the scope of work for each member.

1.  When the video status changes to `READY`, click **"Manage Tasks"**.
2.  Enter the team member's name in the **"User"** field.
3.  Define the frame range for that member to label in the **"Start Frame"** and **"End Frame"** fields.

![img_1.png](assets/img_1.png)

### Step 3: Start Labeling

1.  Find your name in the task list and click **"Start Labeling"** to enter the annotation interface.
2.  In the right sidebar, **create or select a Class**.
3.  **Manual Annotation**:
    - Click and drag your mouse in the image to draw a bounding box (BBox).
    - After finishing, press the `S` key or click **"Save BBoxes"** to save the annotations for the current frame.
    - Use the `A` / `D` keys or drag the progress bar below to switch between frames.
    - A newly drawn BBox is automatically selected; you can delete it by pressing `Delete` or `Backspace`.

### Step 4: AI-Assisted Labeling

To further improve efficiency, we have integrated the powerful SAM 2.1 model.

**Model Configuration**
1.  Return to the main interface and click **"Settings"**.
2.  Here, you can select different SAM models (e.g., `Tiny`, `Small`, `Base`, `Large`), provided the model files are located in the `checkpoints` folder.
3.  The tracker defaults to **CSRT**, which offers balanced performance and usually does not need to be changed.

![img_2.png](assets/img_2.png)

**Smart Labeling and Tracking**
- **Assisted Labeling (SAM)**:
    1.  In the annotation interface, click **"Enable SAM"** in the right sidebar.
    2.  The mouse cursor will change to a pointer. Simply click on the object you want to label, and SAM will automatically generate a precise bounding box for you.
- **Automatic Tracking (SAM Tracker)**:
    1.  After annotating all objects in a frame, click **"Track Object with SAM2"**.
    2.  The system will start from the current frame and automatically track all your annotated objects until the end of the video. You can pause the process at any time to make manual corrections.

### Step 5: Generate and Export Dataset

1.  Return to the main interface and go to the **"Dataset"** tab.
2.  Click **"Create Dataset"** and fill in the dataset name and other information.
3.  Associate one or more completed annotation projects via **"Select Videos"**.
4.  Set the percentage of the total data to be used as the validation set (e.g., `20` for 20%).
5.  After successful creation, click **"Download"** next to the dataset to get a YOLO-formatted `.zip` archive.

> **Important Note**: If you modify any annotations after linking them to a dataset, you will need to create a new dataset to include these changes.

---

## 📂 File Structure

All project data is stored locally, ensuring data privacy and security.

-   **`local_storage`**: Contains all uploaded videos, extracted frames, datasets, and model files.
-   **`ftc_ml.db`**: A SQLite database file that stores all metadata, such as video descriptions, annotation information, and task assignments.

## 🤝 Contribution and Acknowledgements

This project is an improvement and localized implementation of [FIRST-Tech-Challenge/fmltc](https://github.com/FIRST-Tech-Challenge/fmltc).

Special thanks to **BlueDarkUP** for their contributions to the project's development.

Contributions to this project via Pull Requests or Issues are welcome
