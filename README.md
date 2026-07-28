# DL_XVIEW • Object Detection in Satellite Imagery

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-success.svg)
![CI](https://img.shields.io/github/actions/workflow/status/Yigtwxx/dl_xview_yolo/lint.yml?branch=main)
![Issues](https://img.shields.io/github/issues/Yigtwxx/dl_xview_yolo)
![Stars](https://img.shields.io/github/stars/Yigtwxx/dl_xview_yolo?style=social)

**Deep Learning–based Object Detection** system for satellite imagery using **YOLOv8**.
This project applies **state-of-the-art computer vision models** to detect objects such as airplanes, ships, vehicles, bridges, and more from aerial or satellite images.
Data sources include **xView** and **DOTA** datasets.

> Goal: Detect and classify multiple objects in high-resolution satellite imagery
> Model: YOLOv8 (axis-aligned detection; DOTA's oriented boxes are converted to horizontal boxes)

---

## Features

- Object detection using YOLOv8 (Ultralytics)
- Dataset conversion utilities for xView / DOTA → YOLO format (`convert_all_to_yolo.py`)
- Interactive web-based prediction UI (`ui/index.html`, served by FastAPI)
- Training, validation and metric-plotting scripts
- Device selection follows CUDA → MPS → CPU, so it runs on NVIDIA, Apple Silicon and CPU

## Project Structure

```text
dl_xview_yolo/
├── scripts/
│   ├── config.py                  # Shared paths + device selection
│   ├── convert_all_to_yolo.py     # Converts xView/DOTA to YOLO format
│   ├── train_yolo.py              # Training script
│   ├── val_yolo.py                # Validation / mAP evaluation
│   ├── predict_yolo.py            # Inference (CLI) + FastAPI UI
│   └── plot_training_metrics.py   # Plots results.csv into PNG charts
│
├── ui/
│   └── index.html                 # Front-end for predictions
│
├── requirements.txt
├── LICENSE
└── README.md
```

Data (`data/`), converted labels (`yolo_data/`) and training outputs (`runs/`) are
generated locally and excluded via `.gitignore`.

---

## Installation & Setup

> Recommended: Python 3.10 or higher

```bash
git clone https://github.com/Yigtwxx/dl_xview_yolo.git
cd dl_xview_yolo

python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### Where the data lives

By default every script reads and writes inside the repository (`data/`,
`yolo_data/`, `runs/`). To keep datasets elsewhere, set one environment variable:

```bash
export DL_XVIEW_ROOT=/mnt/data/dl_xview     # macOS / Linux
# set DL_XVIEW_ROOT=D:\dl_xview             # Windows
```

---

## Dataset Preparation

### Supported Datasets

* [xView Dataset](https://challenge.xviewdataset.org/)
* [DOTA Dataset](https://captain-whu.github.io/DOTA/)

Expected raw layout under `data/`:

```text
data/
├── xview_raw/
│   ├── images/train/           # or train_images/
│   └── xView_train.geojson     # .geojson or .geojson.gz
└── dota_raw/
    ├── images/train/
    └── labelTxt-v1.5/train/
```

### Convert to YOLO Format

```bash
python scripts/convert_all_to_yolo.py
```

Useful options:

```bash
# Custom source/output directories
python scripts/convert_all_to_yolo.py --src data --out yolo_data

# Convert a single dataset
python scripts/convert_all_to_yolo.py --datasets dota

# Regenerate data.yaml over an existing one
python scripts/convert_all_to_yolo.py --overwrite-yaml
```

**What it does:**

* Parses xView GeoJSON properties (`bounds_imcoords`, `bbox`, `xmin/ymin/...`)
* Converts DOTA's 8-point oriented boxes into axis-aligned YOLO boxes
* Normalizes 16-bit / single-channel GeoTIFFs into 8-bit JPEG
* Writes `images/{train,val}`, `labels/{train,val}` and `data.yaml`

**Class layout:** xView uses ids `0–59` (`type_id - 1`) and DOTA's 15 classes are
offset to `60–74`, so both datasets can share one `labels/` directory without
colliding. Total `nc: 75`.

**Split:** ~20% validation, chosen by an md5 hash of the filename so the split is
identical on every run.

---

## Training

```bash
python scripts/train_yolo.py
```

Options:

```bash
python scripts/train_yolo.py --epochs 100 --imgsz 1024 --batch 4 --name exp1
python scripts/train_yolo.py --weights runs/train/weights/last.pt --resume
```

Defaults: `yolov8m.pt`, AdamW at `lr0=1e-4`, cosine LR, `imgsz=1024`, `batch=4`,
mosaic closed for the final 30 epochs, early stopping at `patience=30`.
Checkpoints land in `runs/train/<name>/weights/`.

---

## Validation

```bash
python scripts/val_yolo.py                       # auto-finds the latest best.pt
python scripts/val_yolo.py --weights runs/train/exp1/weights/best.pt --split val
```

## Plot training metrics

```bash
python scripts/plot_training_metrics.py                        # auto-find results.csv
python scripts/plot_training_metrics.py --run-dir runs/train/exp1
```

Charts are written to `<run-dir>/plots_matplotlib/`.

---

## Inference

### Web UI (default)

```bash
python scripts/predict_yolo.py
```

Opens **[http://127.0.0.1:7860/](http://127.0.0.1:7860/)** with a drag-and-drop
panel that returns the annotated image. Weights are auto-discovered from
`runs/train/**/weights/best.pt`.

### Command line

```bash
python scripts/predict_yolo.py --no-ui --source data/test_images --conf 0.25
```

---

## Example Results

| Metric       | Value |
| ------------ | ----- |
| mAP@0.5      | 0.54  |
| mAP@0.5-0.95 | 0.36  |
| Precision    | 0.67  |
| Recall       | 0.71  |

> Measured on the validation split of an xView-only run. Metrics vary with the
> dataset mix and training configuration; see `runs/train/**/results.png`.

---

## Model Weights

Trained weights are not tracked in Git (`*.pt` is in `.gitignore`). After
training, `best.pt` lives at:

```text
runs/train/<name>/weights/best.pt
```

---

## Requirements

* Python ≥ 3.10
* PyTorch ≥ 2.0
* Ultralytics YOLOv8
* OpenCV, Pillow, NumPy, tqdm
* Matplotlib
* FastAPI + Uvicorn (only for the prediction UI)

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Author

**Yiğit Erdoğan**

Focus Areas: Deep Learning • Computer Vision • Data Science

* Mail: yigiterdogan6@icloud.com
* LinkedIn: [yigit-erdogan0](https://www.linkedin.com/in/yigit-erdogan0/)
* GitHub: [@Yigtwxx](https://github.com/Yigtwxx)

---

## Support

If you find this project useful, please consider giving it a ⭐ on GitHub!
