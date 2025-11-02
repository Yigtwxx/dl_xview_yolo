# C:\Users\Asus\Desktop\dl_xview\scripts\val_yolo.py
from ultralytics import YOLO
from pathlib import Path
import argparse, glob, torch

# ---- common paths ----
ROOT = Path(r"C:\Users\Asus\Desktop\dl_xview")
RUNS = ROOT / "runs"
DATA_YAML = ROOT / "yolo_data" / "data.yaml"

def find_weights() -> Path:
    """
    runs/train/**/weights/{best,last}.pt içinde en son denemeyi (exp) bulur.
    Önce best.pt, yoksa last.pt.
    """
    bests = sorted(glob.glob(str(RUNS / "train" / "**" / "weights" / "best.pt"), recursive=True))
    lasts = sorted(glob.glob(str(RUNS / "train" / "**" / "weights" / "last.pt"), recursive=True))
    if bests:
        return Path(bests[-1])
    if lasts:
        return Path(lasts[-1])
    raise FileNotFoundError("runs/train/**/weights içinde best.pt veya last.pt bulunamadı.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=None,
                    help="Ağırlık dosyası (varsayılan: runs/train/**/weights içinden otomatik bul)")
    ap.add_argument("--data", type=str, default=str(DATA_YAML),
                    help="data.yaml yolu (varsayılan: yolo_data/data.yaml)")
    ap.add_argument("--imgsz", type=int, default=1024, help="giriş boyutu")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                    help="değerlendirilecek split")
    ap.add_argument("--device", type=str, default=None,
                    help="CUDA cihazı ('0' gibi) veya 'cpu'; boşsa otomatik algılar")
    ap.add_argument("--project", type=str, default=str(ROOT / "runs" / "val"),
                    help="Ultralytics çıktı projesi")
    ap.add_argument("--name", type=str, default="exp", help="deneme adı")
    args = ap.parse_args()

    # Device selection
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")

    # Select weights
    weights = Path(args.weights) if args.weights else find_weights()
    if not weights.exists():
        raise FileNotFoundError(f"Ağırlık bulunamadı: {weights}")

    print(f"Using weights : {weights}")
    print(f"Data YAML     : {args.data}")
    print(f"Split         : {args.split}")
    print(f"Image size    : {args.imgsz}")
    print(f"Device        : {device}")
    print(f"Save to       : {args.project}\\{args.name}")

    # Load model and evaluate
    model = YOLO(str(weights))
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        device=device,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    # Summary
    print("\n[VALIDATION COMPLETED]")
    try:
        print(f"mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")
    except Exception:
        print("mAP information could not be retrieved (label may be missing or empty).")

if __name__ == "__main__":
    main()
