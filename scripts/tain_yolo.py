from ultralytics import YOLO
from pathlib import Path
import torch
import argparse

# To continue from where you left off: find the last.pt file in dir C:\Users\Asus\Desktop\dl_xview\runs\train\weights
# Then: python scripts\train_yolo.py --weights C:\Users\Asus\Desktop\dl_xview\runs\train\weights\last.pt --resume

# ==========================
# common paths
# ==========================
ROOT = Path(r"C:\Users\Asus\Desktop\dl_xview")
YOLO_DATA = ROOT / "yolo_data"
RUNS_DIR = ROOT / "runs"

# Current data file:
DATA_YAML = YOLO_DATA / "data.yaml"   # C:\Users\Asus\Desktop\dl_xview\yolo_data\data.yaml

# 🔴 IMPORTANT: WE ARE TRAINING DETECTION NOT OBB
# Previously: "yolov8m-obb.pt"
MODEL_PATH = "yolov8m.pt"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model with optional resume/weights")
    parser.add_argument(
        "--weights",
        type=str,
        default=MODEL_PATH,
        help="Weights file or pretrained model to start from (default: yolov8m.pt)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Try to resume training from checkpoint (Ultralytics resume=True)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Total number of epochs to run (default: 250)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name under runs/ (default: train)",
    )
    args = parser.parse_args()

    # device selection
    device = "0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n💻 Kullanılan cihaz: {device} ({device_name})")
    print(f"📦 Kullanılan ağırlık: {args.weights}    (resume={args.resume})")

    # 🔴 DETECTION MODEL LOADING
    model = YOLO(args.weights)

    # ==========================
    # TRAINING
    # ==========================
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1024,
        batch=4,
        device=device,
        workers=2,
        optimizer="AdamW",
        lr0=0.0001,
        dropout=0.05,
        hsv_h=0.015 ,hsv_s=0.75 ,hsv_v=0.45, 
        translate=0.15 ,scale=0.55 ,mixup=0.15 ,copy_paste=0.15,        
        shear=0.0,
        perspective=0.0,
        close_mosaic=30,
        patience=30,
        cos_lr=True,
        pretrained=True,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
    )

    print("\n✅ [EĞİTİM TAMAMLANDI]")
    print(f"📂 Sonuç klasörü: {results.save_dir}")

    if torch.cuda.is_available():
        print(f"🔥 GPU VRAM kullanımı: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
