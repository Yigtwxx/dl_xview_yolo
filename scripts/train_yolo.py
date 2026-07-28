from ultralytics import YOLO
import argparse
import torch

from config import DATA_YAML, RUNS, describe_device, select_device

# To resume from where you left off, point --weights at the last checkpoint:
#   python scripts/train_yolo.py --weights runs/train/weights/last.pt --resume

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
        default=100,
        help="Total number of epochs to run (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Training image size; small objects need resolution (default: 1024)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (default: 4, which fits 1024px on a single GPU)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name under runs/ (default: train)",
    )
    args = parser.parse_args()

    # device selection: CUDA -> MPS -> CPU
    device = select_device()
    print(f"\n💻 Kullanılan cihaz: {device} ({describe_device(device)})")
    print(f"📦 Kullanılan ağırlık: {args.weights}    (resume={args.resume})")

    # 🔴 DETECTION MODEL LOADING
    model = YOLO(args.weights)

    # ==========================
    # TRAINING
    # ==========================
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
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
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
    )

    print("\n✅ [EĞİTİM TAMAMLANDI]")
    print(f"📂 Sonuç klasörü: {results.save_dir}")

    if device.isdigit():
        print(f"🔥 GPU VRAM kullanımı: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
