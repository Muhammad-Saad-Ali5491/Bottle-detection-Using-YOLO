<div align="center">

# 🥤 Bottle Detection with YOLOv8

**Detect Pepsi bottles in images with a custom-trained YOLOv8 model.**

[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-7B68EE?logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Roboflow-6706CE)](https://universe.roboflow.com/muhammad-saad-ali/my-first-project-g8ifu/dataset/1)

Upload images in the web app to view detections, confidence scores, bounding-box coordinates, and download annotated results.

</div>

---

## Contents

- [Project overview](#project-overview)
- [Preview](#preview)
- [Features](#features)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Training and evaluation](#training-and-evaluation)
- [Dataset](#dataset)
- [Model details](#model-details)
- [Troubleshooting](#troubleshooting)
- [Credits and licensing](#credits-and-licensing)

## Project overview

This project combines a YOLOv8 object-detection workflow with a simple Streamlit interface. The model is trained to recognize one class, **Pepsi Bottle**. The repository includes the training notebook, dataset configuration and images, pretrained YOLOv8 nano weights, recorded training artifacts, and the trained model weights.

The app accepts multiple images and runs inference locally with the selected weights. It displays the original and annotated images side by side, lists each detection, and provides a PNG download.

## Preview

Example image included in the repository:

<div align="center">
<img src="multiple%20bottle%20test.jpeg" alt="Example image for bottle detection" width="640">
</div>

## Features

- **Single-class detection:** identifies Pepsi Bottles and draws bounding boxes.
- **Multi-image inference:** upload JPG, JPEG, PNG, BMP, and WebP files in one session.
- **Adjustable inference thresholds:** set confidence and IoU thresholds in the sidebar.
- **Detection details:** view the predicted class, confidence, and pixel coordinates for each box.
- **Download results:** save each annotated image as a PNG.
- **Training workflow:** explore data preparation, augmentation, YOLOv8 training, test-set evaluation, and ONNX export in the notebook.

## Repository layout

```text
.
├── app.py                                  # Streamlit inference app
├── YOLO_Implementation.ipynb               # Training, evaluation, and export workflow
├── yolov8n.pt                              # Pretrained YOLOv8 nano starting weights
├── multiple bottle test.jpeg               # Example image
├── Dataset/
│   ├── data.yaml                           # Dataset and class configuration
│   ├── train/                              # Training images and YOLO labels
│   ├── valid/                              # Validation images and labels
│   └── test/                               # Test images and labels
└── runs/detect/runs/bottle_detect/exp3/
    ├── results.csv                         # Recorded training metrics
    ├── results.png                         # Training metric plots
    └── weights/
        ├── best.pt                         # Best checkpoint used by the app
        └── last.pt                          # Final checkpoint
```

## Quick start

### Requirements

- Python and `pip`
- The repository's model weights (included in the clone)

### Install and launch

Run these commands from the repository root:

```bash
git clone https://github.com/Muhammad-Saad-Ali5491/Bottle-detection-Using-YOLO.git
cd Bottle-detection-Using-YOLO

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install streamlit ultralytics numpy pillow
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser. The app's default checkpoint path points to the trained `best.pt` included in this repository. If you move the weights, choose **Use a path on disk** in the sidebar and enter the new path, or select **Upload a weights file** to provide a `.pt` or `.onnx` checkpoint.

## Using the app

1. Start the app with `streamlit run app.py`.
2. Confirm the weights path in the sidebar, or upload a compatible `.pt` / `.onnx` file.
3. Adjust the confidence and IoU thresholds if needed. Higher confidence thresholds generally filter out more low-confidence detections.
4. Upload one or more images.
5. Review the annotated image and detection table, then download the PNG if desired.

The app reports coordinates as `(x1, y1, x2, y2)` in pixels, measured from the top-left of the image.

## Training and evaluation

The notebook `YOLO_Implementation.ipynb` documents the model workflow. It uses `yolov8n.pt` as a starting checkpoint, applies training-time augmentation, trains at 640-pixel image size, evaluates against the test split, and includes an optional ONNX export step.

To prepare a notebook environment, install the app dependencies plus the notebook's additional packages:

```bash
python -m pip install pyyaml opencv-python albumentations matplotlib
```

Then open the notebook in Jupyter or Google Colab and run the cells in order. **Before running it locally, update the notebook's `DATASET_DIR` from `Path("dataset")` to `Path("Dataset")`** (the checked-in directory uses an uppercase `D`). Review the output paths in the notebook after training; if your generated checkpoint is not at the app's default path, update the model path in the Streamlit sidebar.

Training can be compute-intensive. The notebook sets the batch size to 16 and includes a note to reduce it if GPU memory is insufficient. YOLO training can also run on CPU, but will take longer.

## Dataset

The dataset configuration in `Dataset/data.yaml` declares one class:

| ID | Class |
|---:|---|
| 0 | Pepsi Bottle |

The dataset is organized into `train`, `valid`, and `test` splits, each with `images/` and `labels/` directories. Labels use the YOLO bounding-box format: `class_id x_center y_center width height`, with normalized coordinates. The Roboflow export notes 64 source images and describes its preprocessing and augmentations; consult the included `Dataset/README.roboflow.txt` for export details.

## Model details

| Item | Value |
|---|---|
| Task | Object detection |
| Model family | Ultralytics YOLOv8 |
| Starting checkpoint | YOLOv8 Nano (`yolov8n.pt`) |
| Target class | Pepsi Bottle |
| Inference interface | Streamlit (`app.py`) |
| Included trained checkpoint | `runs/detect/runs/bottle_detect/exp3/weights/best.pt` |

The training run's plots and metrics are stored alongside the checkpoint under `runs/detect/runs/bottle_detect/exp3/`.

## Troubleshooting

- **Weights file not found:** launch Streamlit from the repository root, or set the correct checkpoint path in the sidebar.
- **Model does not load:** make sure the selected file is a supported Ultralytics `.pt` or `.onnx` model and that `ultralytics` is installed.
- **Notebook cannot find the dataset:** use `Path("Dataset")` for `DATASET_DIR`, matching the checked-in folder name.
- **CUDA out-of-memory during training:** lower the batch size in the notebook (for example, from 16 to 8) or train on CPU.
- **No detections:** check the selected weights and thresholds; the model is trained for Pepsi Bottles, not general-purpose bottle detection.

## Credits and licensing

- The dataset was exported from [Roboflow Universe](https://universe.roboflow.com/muhammad-saad-ali/my-first-project-g8ifu) and its included metadata identifies it as **CC BY 4.0**. Follow the attribution requirements when redistributing the dataset.
- Model training and inference use [Ultralytics](https://github.com/ultralytics/ultralytics). Review the applicable Ultralytics licensing terms for your use case.
- No separate license file for this repository is included; do not assume the dataset's license applies to the source code or model weights.
