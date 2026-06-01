#  DL_XVIEW • Object Detection in Satellite Imagery
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-success.svg)
![Stars](https://img.shields.io/github/stars/Yigtwxx/dl_xview_yolo?style=social)


![CI](https://img.shields.io/github/actions/workflow/status/Yigtwxx/FiratUniversityChatbot/ci.yml?branch=main)
![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Issues](https://img.shields.io/github/issues/Yigtwxx/dl_xview_yolo)

**Deep Learning–based Object Detection** system for satellite imagery using **YOLOv8**.  
This project applies **state-of-the-art computer vision models** to detect objects such as airplanes, ships, vehicles, bridges, and more from aerial or satellite images.  
Data sources include **xView** and **DOTA** datasets.

>  Goal: Detect and classify multiple objects in high-resolution satellite imagery  
>  Model: YOLOv8 (with Oriented Bounding Box / OBB support)

---

##  Features

-  Object detection using YOLOv8 (Ultralytics)
-  Dataset conversion utilities for xView / DOTA → YOLO format (`convert_all_to_yolo.py`)
-  Interactive web-based prediction UI (`ui/index.html`)
-  Utility scripts: `train_yolo.py`, `predict_yolo.py`, `summarize_run.py`
-  Easy path and parameter customization


##  Project Structure
```text
dl_xview_yolo/
├── scripts/
│   ├── convert_all_to_yolo.py     # Converts datasets to YOLO format
│   ├── train_yolo.py              # Training script
│   ├── predict_yolo.py            # Inference script
│   ├── summarize_run.py           # Summarizes training metrics
│   └── prepare_dota_v1_to_yolo_obb.py
│
├── ui/
│   └── index.html                 # Interactive front-end for predictions
│
├── YOLOv8/                        # YOLOv8 models / configs
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt


## Installation & Setup

### Environment Preparation

> Recommended: Python 3.10 or higher

```bash
git clone https://github.com/Yigtwxx/dl_xview_yolo.git
cd dl_xview_yolo
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` file, install manually:

```bash
pip install ultralytics opencv-python pillow tqdm numpy torch torchvision torchaudio matplotlib
```

---

##  Dataset Preparation

### Supported Datasets

* [xView Dataset](https://challenge.xviewdataset.org/)
* [DOTA Dataset](https://captain-whu.github.io/DOTA/)

### Convert to YOLO Format

```bash
python scripts/convert_all_to_yolo.py --src "data/raw" --out "data/yolo_data" --copy
```

**Explanation:**

* Converts raw `.txt` annotations into YOLO-style labels
* Saves output to `data/yolo_data`
* Copies images when `--copy` flag is used

---

##  Training

```bash
python scripts/train_yolo.py --data "configs/data.yaml" --model "yolov8n.pt" --epochs 40 --imgsz 1024
```

* Model checkpoints are saved under `runs/train/...`
* You can monitor metrics via `results.png` (mAP, loss curves, etc.)

---

##  Inference

### Command Line

```bash
python scripts/predict_yolo.py --weights "runs/train/xview-yolo/weights/best.pt" --source "data/test_images"
```

### Web UI

```bash
# If integrated with FastAPI / Streamlit UI:
python scripts/predict_yolo.py --ui
```

Then open your browser at
 **[http://127.0.0.1:7860/](http://127.0.0.1:7860/)**

---

## User Interface (UI)

Open `ui/index.html` directly in your browser to visualize predictions.

> 💫 Modern design: world-themed background & glass-panel overlay
> 📂 Upload: drag-and-drop or select image
> 🧠 Backend: connects directly to YOLOv8 model for real-time inference

---

## 📊 Example Results

| Metric       | Value |
| ------------ | ----- |
| mAP@0.5      | 0.54  |
| mAP@0.5-0.95 | 0.36  |
| Precision    | 0.67  |
| Recall       | 0.71  |

> *Metrics may vary depending on dataset and training configuration.*
> See `runs/train/.../results.png` for detailed training graphs.

---

## 📦 Model Weights

If your trained model (`best.pt`) is smaller than 100 MB, you can store it in:

```
runs/train/xview-yolo/weights/best.pt
```

Otherwise, upload to Google Drive and link it here:

```
[📁 Download best.pt](https://drive.google.com/your-model-link)
```

---

## 🧩 Requirements

* 🧠 Python ≥ 3.10
* ⚡ PyTorch ≥ 2.0
* 🎯 Ultralytics YOLOv8
* 📸 OpenCV, Pillow
* 📊 Matplotlib

---

## ⚠️ Notes

* Training datasets (`data/`, `runs/`, `venv/`) are excluded from Git using `.gitignore`.
* To resume training from the last checkpoint:

```bash
python scripts/train_yolo.py --resume
```
---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## 💬 Author

**Yiğit Erdoğan**

📧 Mail: yigiterdogan6@icloud.com

🧠 Focus Areas: Deep Learning • Computer Vision • Data Science

---

## Contact
LinkedIn: www.linkedin.com/in/yiğit-erdoğan-ba7a64294

---

## ⭐ Support

If you find this project useful, please consider giving it a **⭐ Star** on GitHub!

## Contact
LinkedIn: www.linkedin.com/in/yiğit-erdoğan-ba7a64294

```bash
git clone https://github.com/Yigtwxx/dl_xview_yolo.git
```
