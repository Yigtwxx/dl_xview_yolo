# C:\Users\Asus\Desktop\dl_xview\scripts\plot_training_metrics.py
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import argparse
import glob

ROOT = Path(r"C:\Users\Asus\Desktop\dl_xview")

def auto_find_results_csv(root: Path) -> Path:
    """
    Ultralytics çıktı yapılarında en uygun results.csv dosyasını bulur.
    Tercih sırası:
      1) runs/train/results.csv
      2) runs/train/**/results.csv (en son exp)
      3) runs/detect/train/results.csv (eski)
      4) runs/detect/train/**/results.csv (en son exp)
    """
    candidates = [
        root / "runs" / "train" / "results.csv",
        *[Path(p) for p in sorted(glob.glob(str(root / "runs" / "train" / "**" / "results.csv"), recursive=True))],
        root / "runs" / "detect" / "train" / "results.csv",
        *[Path(p) for p in sorted(glob.glob(str(root / "runs" / "detect" / "train" / "**" / "results.csv"), recursive=True))],
    ]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError(
            "results.csv bulunamadı. Eğitim koştunuz mu? "
            "Beklenen yer: runs/train/**/results.csv (veya eski runs/detect/**/results.csv)"
        )
    return candidates[-1]  # best 

def read_results_csv(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Boş results.csv: {csv_path}")
    return rows

# Bazı Ultralytics sürümlerinde kolon adları küçük farklarla değişebilir.
ALIAS = {
    # losses
    "train/box_loss": ["train/box_loss", "train/box_loss(B)"],
    "train/cls_loss": ["train/cls_loss", "train/cls_loss(B)"],
    "train/dfl_loss": ["train/dfl_loss", "train/dfl_loss(B)"],
    "val/box_loss":   ["val/box_loss", "val/box_loss(B)"],
    "val/cls_loss":   ["val/cls_loss", "val/cls_loss(B)"],
    "val/dfl_loss":   ["val/dfl_loss", "val/dfl_loss(B)"],

    # metrics
    "metrics/precision": ["metrics/precision(B)", "metrics/precision"],
    "metrics/recall":    ["metrics/recall(B)", "metrics/recall"],
    "metrics/mAP50":     ["metrics/mAP50(B)", "metrics/mAP50"],
    "metrics/mAP50-95":  ["metrics/mAP50-95(B)", "metrics/mAP50-95"],

    # lr
    "lr/pg0": ["lr/pg0"],
    "lr/pg1": ["lr/pg1"],
    "lr/pg2": ["lr/pg2"],

    # misc
    "gpu_mem": ["gpu_mem", "mem(GiB)", "mem"],
    "epoch": ["epoch"],
}

def extract_col(rows, keys, cast=float):
    """keys: olası kolon adları listesi. Bulduğu ilkini döndürür."""
    header = rows[0].keys()
    key_found = None
    for k in keys:
        if k in header:
            key_found = k
            break
    vals = []
    if key_found is None:
        # hiçbiri yoksa None dizisi döndür (grafikte boş olur)
        return [None for _ in rows]
    for r in rows:
        v = r.get(key_found, "")
        if v == "" or v is None:
            vals.append(None)
        else:
            try:
                vals.append(cast(v))
            except Exception:
                vals.append(None)
    return vals

def col(rows, logical_name, cast=float):
    return extract_col(rows, ALIAS.get(logical_name, [logical_name]), cast)

def simple_plot(xs, ys_list, labels, title, ylabel, out_path: Path):
    plt.figure(figsize=(9, 5))
    for ys, lab in zip(ys_list, labels):
        plt.plot(xs, ys, label=lab, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="results.csv'nin bulunduğu koşu klasörü (örn: runs/train/exp). "
                         "Boşsa otomatik bulunur.")
    args = ap.parse_args()

    # results.csv yolunu belirle
    if args.run_dir:
        run_dir = Path(args.run_dir)
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Bulunamadı: {csv_path}")
    else:
        csv_path = auto_find_results_csv(ROOT)
        run_dir = csv_path.parent

    out_dir = run_dir / "plots_matplotlib"
    print(f"Using results.csv : {csv_path}")
    print(f"Saving plots to  : {out_dir}")

    # CSV reading
    rows = read_results_csv(csv_path)

    # columns
    epoch   = col(rows, "epoch", int)
    tr_box  = col(rows, "train/box_loss")
    tr_cls  = col(rows, "train/cls_loss")
    tr_dfl  = col(rows, "train/dfl_loss")
    va_box  = col(rows, "val/box_loss")
    va_cls  = col(rows, "val/cls_loss")
    va_dfl  = col(rows, "val/dfl_loss")

    prec    = col(rows, "metrics/precision")
    rec     = col(rows, "metrics/recall")
    map50   = col(rows, "metrics/mAP50")
    map5095 = col(rows, "metrics/mAP50-95")

    lr0     = col(rows, "lr/pg0")
    lr1     = col(rows, "lr/pg1")
    lr2     = col(rows, "lr/pg2")
    tmem    = col(rows, "gpu_mem", float)

    # Graphics
    simple_plot(epoch, [tr_box, va_box], ["train/box_loss", "val/box_loss"],
                "Box Loss (train vs val)", "loss", out_dir / "loss_box.png")
    simple_plot(epoch, [tr_cls, va_cls], ["train/cls_loss", "val/cls_loss"],
                "Cls Loss (train vs val)", "loss", out_dir / "loss_cls.png")
    simple_plot(epoch, [tr_dfl, va_dfl], ["train/dfl_loss", "val/dfl_loss"],
                "DFL Loss (train vs val)", "loss", out_dir / "loss_dfl.png")

    simple_plot(epoch, [prec, rec], ["precision", "recall"],
                "Precision & Recall", "score", out_dir / "pr_over_epochs.png")
    simple_plot(epoch, [map50, map5095], ["mAP50", "mAP50-95"],
                "mAP Progress", "score", out_dir / "map_over_epochs.png")

    simple_plot(epoch, [lr0, lr1, lr2], ["lr/pg0", "lr/pg1", "lr/pg2"],
                "Learning Rate Schedule", "lr", out_dir / "lr_plan.png")

    if any(v is not None for v in tmem):
        simple_plot(epoch, [tmem], ["gpu_mem(GB)"],
                    "GPU Memory Usage", "GB", out_dir / "gpu_mem.png")

    print(f"[OK] Grafikler kaydedildi: {out_dir}")

if __name__ == "__main__":
    main()
