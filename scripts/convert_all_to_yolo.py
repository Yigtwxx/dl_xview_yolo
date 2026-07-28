from pathlib import Path
from tqdm import tqdm
import argparse, hashlib, json, gzip
import cv2
import numpy as np

from config import DATA, YOLO_DATA


def make_output_dirs(out_root: Path) -> None:
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)


# ===============================
# Ortak yardımcılar
# ===============================
def ensure_bgr_uint8(im):
    if im is None:
        return None
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def convert_to_jpg(src: Path, dst: Path):
    im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Görüntü açılamadı: {src}")
    im = ensure_bgr_uint8(im)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def is_val_split(name: str) -> bool:
    """Deterministic ~20% validation split.

    Uses md5 rather than the builtin hash(): CPython salts string hashes per
    process, so hash() would reshuffle the split on every run and leak training
    images into validation across re-conversions.
    """
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (int(digest, 16) % 10) < 2


# ===============================
# xView yardımcıları
# ===============================
def load_geojson_any(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON yok: {path}")
    with open(path, "rb") as f:
        head2 = f.read(2)
    is_gz = head2 == b"\x1f\x8b" or path.suffix == ".gz" or "".join(path.suffixes).endswith(".json.gz")

    def _parse_text(txt: str):
        txt = txt.lstrip("\ufeff \r\n\t")
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            feats = []
            for ln in lines:
                try:
                    feats.append(json.loads(ln))
                except Exception:
                    pass
            if feats:
                return {"type": "FeatureCollection", "features": feats}
            raise

    if is_gz:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return _parse_text(f.read())
    else:
        with open(path, "r", encoding="utf-8-sig") as f:
            return _parse_text(f.read())


def get_prop_ci(d: dict, *cands):
    if not d:
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    for c in cands:
        if c is None:
            continue
        k = str(c).lower()
        if k in low:
            return low[k]
    return None


def parse_bounds_string(s: str):
    if not s or str(s).strip().lower() in {"", "null", "none"}:
        return None
    try:
        s = str(s).replace(" ", "").replace("[", "").replace("]", "")
        nums = [float(x) for x in s.split(",") if x != ""]
        if len(nums) == 4:
            x1, y1, x2, y2 = nums
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        elif len(nums) == 8:
            xs = nums[0::2]
            ys = nums[1::2]
            return min(xs), min(ys), max(xs), max(ys)
        return None
    except Exception:
        return None


def extract_bbox_from_props(props: dict):
    # 1) bounds_imcoords
    for key in list(props.keys()):
        lk = key.lower()
        if "bounds" in lk and "imcoords" in lk:
            bb = parse_bounds_string(props[key])
            if bb:
                return bb
    # 2) bbox
    bbox = get_prop_ci(props, "bbox", "bound_box", "bounding_box")
    if bbox is not None:
        if isinstance(bbox, str):
            nums = [float(x) for x in bbox.replace(" ", "").split(",") if x != ""]
        elif isinstance(bbox, (list, tuple)):
            nums = [float(x) for x in bbox]
        else:
            nums = []
        if len(nums) >= 4:
            x, y, w, h = nums[:4]
            as_xywh = (x, y, x + w, y + h)
            as_xyxy = (x, y, nums[2], nums[3])
            if w > 0 and h > 0:
                return as_xywh
            if as_xyxy[2] > as_xyxy[0] and as_xyxy[3] > as_xyxy[1]:
                return as_xyxy
    # 3) xmin/xmax
    cand_names = [
        ("xmin", "ymin", "xmax", "ymax"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom"),
    ]
    low = {str(k).lower(): v for k, v in props.items()}
    for (xmin_k, ymin_k, xmax_k, ymax_k) in cand_names:
        if xmin_k in low and ymin_k in low and xmax_k in low and ymax_k in low:
            try:
                xmin = float(low[xmin_k]); ymin = float(low[ymin_k])
                xmax = float(low[xmax_k]); ymax = float(low[ymax_k])
                if xmax > xmin and ymax > ymin:
                    return xmin, ymin, xmax, ymax
            except Exception:
                pass
    return None


def auto_find_xview_geojson(data_dir: Path) -> Path:
    cands = [
        data_dir / "xview_raw" / "xView_train.geojson",
        data_dir / "xView_train.geojson",
        data_dir / "xview_raw" / "xView_train.geojson.gz",
        data_dir / "xView_train.geojson.gz",
        * (data_dir / "xview_raw").glob("*.geojson"),
        * data_dir.glob("*.geojson"),
    ]
    cands = [p for p in cands if p.exists()]
    if not cands:
        raise FileNotFoundError("xView geojson bulunamadı.")
    return max(cands, key=lambda p: p.stat().st_size)


def xview_to_yolo_cls(xv: int) -> int:
    return int(xv) - 1


def convert_xview(data_dir: Path, out_root: Path):
    xview_root = data_dir / "xview_raw"

    # görüntü klasörünü bul
    img_candidates = [
        xview_root / "images" / "train",
        xview_root / "train_images",
        data_dir / "train_images",
    ]
    img_dir = None
    for c in img_candidates:
        if c.exists():
            img_dir = c
            break
    if img_dir is None:
        print("ℹ️ xView görüntü klasörü bulunamadı, xView atlandı.")
        return

    print(f"📁 xView görüntü klasörü: {img_dir}")

    gj_path = auto_find_xview_geojson(data_dir)
    print(f"📄 xView GeoJSON: {gj_path}")
    gj = load_geojson_any(gj_path)
    feats = gj.get("features", [])
    print(f"📦 xView feature sayısı: {len(feats)}")

    # geojson → image_id -> bbox listesi
    records_by_image = {}
    miss_bbox = 0
    for feat in tqdm(feats, desc="xView özellikleri okunuyor"):
        props = feat.get("properties", {}) or {}
        img_id_raw = get_prop_ci(props, "image_id", "imageId", "image_filename", "image_name")
        if not img_id_raw:
            continue
        cls_id = get_prop_ci(props, "type_id", "class_id", "typeId")
        if cls_id is None:
            continue
        bb = extract_bbox_from_props(props)
        if not bb:
            miss_bbox += 1
            continue

        img_stem = Path(str(img_id_raw)).stem
        minx, miny, maxx, maxy = map(float, bb)
        rec = (minx, miny, maxx, maxy, int(cls_id))

        for k in {
            img_stem, img_stem + ".tif", img_stem + ".jpg",
            img_stem.upper(), (img_stem + ".tif").upper(), (img_stem + ".jpg").upper()
        }:
            records_by_image.setdefault(k, []).append(rec)

    print(f"✅ xView kullanılabilir bbox: {sum(len(v) for v in records_by_image.values())}")
    print(f"⚠️ xView atlanan bbox: {miss_bbox}")

    # görüntüleri dolaş
    img_list = []
    for ext in ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"):
        img_list.extend(img_dir.rglob(ext))
    print(f"🔎 xView görüntü sayısı: {len(img_list)}")

    for img_p in tqdm(img_list, desc="xView dönüştürülüyor"):
        stem = img_p.stem
        split = "val" if is_val_split(stem) else "train"
        out_img = out_root / "images" / split / f"{stem}.jpg"
        out_lbl = out_root / "labels" / split / f"{stem}.txt"

        convert_to_jpg(img_p, out_img)

        im = cv2.imread(str(out_img))
        if im is None:
            print(f"⚠️ xView: çıktı okunamadı {out_img}")
            continue
        h, w = im.shape[:2]

        # etiketleri eşleştir
        recs = []
        for k in {
            stem, stem + ".tif", stem + ".jpg",
            stem.upper(), (stem + ".tif").upper(), (stem + ".jpg").upper()
        }:
            if k in records_by_image:
                recs = records_by_image[k]
                break

        lines = []
        for (minx, miny, maxx, maxy, cls_id) in recs:
            minx = clip(minx, 0, w - 1); maxx = clip(maxx, 0, w - 1)
            miny = clip(miny, 0, h - 1); maxy = clip(maxy, 0, h - 1)
            if maxx <= minx or maxy <= miny:
                continue

            cx = (minx + maxx) / 2.0 / w
            cy = (miny + maxy) / 2.0 / h
            bw = (maxx - minx) / w
            bh = (maxy - miny) / h

            yolo_cls = xview_to_yolo_cls(cls_id)
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                continue
            if yolo_cls < 0:
                continue
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.write_text("\n".join(lines), encoding="utf-8")

    print("✅ xView dönüştürme tamamlandı.")


# ===============================
# DOTA bölümü
# ===============================
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]

# xView occupies ids 0..59 (type_id - 1), so DOTA classes are shifted above it.
# Without this offset, writing both datasets into the same labels/ directory
# would silently map DOTA's "plane" onto xView class 0.
XVIEW_NUM_CLASSES = 60
DOTA_CLASS_OFFSET = XVIEW_NUM_CLASSES
NUM_CLASSES = XVIEW_NUM_CLASSES + len(DOTA_CLASSES)


def parse_dota_line(line: str):
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    try:
        xs = list(map(float, parts[0:8:2]))
        ys = list(map(float, parts[1:8:2]))
        cls_name = parts[8]
    except Exception:
        return None
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    return xmin, ymin, xmax, ymax, cls_name


def convert_dota(data_dir: Path, out_root: Path):
    dota_root = data_dir / "dota_raw"
    img_dir = dota_root / "images" / "train"
    lbl_dir = dota_root / "labelTxt-v1.5" / "train"

    if not img_dir.exists() or not lbl_dir.exists():
        print("ℹ️ DOTA dizini yok veya eksik, DOTA atlandı.")
        return

    img_list = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        img_list.extend(img_dir.rglob(ext))
    print(f"🔎 DOTA görüntü sayısı: {len(img_list)}")

    for img_p in tqdm(img_list, desc="DOTA dönüştürülüyor"):
        stem = img_p.stem
        lbl_p = lbl_dir / f"{stem}.txt"
        if not lbl_p.exists():
            # büyük harf dene
            cand = list(lbl_dir.glob(f"{stem.upper()}.txt"))
            if cand:
                lbl_p = cand[0]
            else:
                # etiket yoksa boş label
                split = "val" if is_val_split(stem) else "train"
                out_img = out_root / "images" / split / f"{stem}.jpg"
                out_lbl = out_root / "labels" / split / f"{stem}.txt"
                convert_to_jpg(img_p, out_img)
                out_lbl.write_text("", encoding="utf-8")
                continue

        split = "val" if is_val_split(stem) else "train"
        out_img = out_root / "images" / split / f"{stem}.jpg"
        out_lbl = out_root / "labels" / split / f"{stem}.txt"
        convert_to_jpg(img_p, out_img)

        im = cv2.imread(str(out_img))
        if im is None:
            print(f"⚠️ DOTA çıktı okunamadı: {out_img}")
            continue
        h, w = im.shape[:2]

        lines = []
        with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                parsed = parse_dota_line(ln)
                if not parsed:
                    continue
                xmin, ymin, xmax, ymax, cls_name = parsed
                xmin = clip(xmin, 0, w - 1); xmax = clip(xmax, 0, w - 1)
                ymin = clip(ymin, 0, h - 1); ymax = clip(ymax, 0, h - 1)
                if xmax <= xmin or ymax <= ymin:
                    continue
                cx = (xmin + xmax) / 2.0 / w
                cy = (ymin + ymax) / 2.0 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h

                if cls_name in DOTA_CLASSES:
                    cls_id = DOTA_CLASSES.index(cls_name) + DOTA_CLASS_OFFSET
                else:
                    continue

                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.write_text("\n".join(lines), encoding="utf-8")

    print("✅ DOTA dönüştürme tamamlandı.")


# ===============================
# data.yaml
# ===============================
def write_data_yaml(out_root: Path, overwrite: bool = False):
    yaml_p = out_root / "data.yaml"
    if yaml_p.exists() and not overwrite:
        print("ℹ️ data.yaml zaten var (üzerine yazmak için --overwrite-yaml).")
        return

    # xView ships numeric type ids only, so those classes stay numbered.
    names = [f"xview_{i}" for i in range(XVIEW_NUM_CLASSES)] + DOTA_CLASSES
    yaml_p.write_text(
        "path: " + str(out_root) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {NUM_CLASSES}\n"
        "names: [" + ", ".join(names) + "]\n",
        encoding="utf-8"
    )
    print(f"📝 data.yaml yazıldı: {yaml_p} (nc={NUM_CLASSES})")


# ===============================
# MAIN
# ===============================
def main():
    ap = argparse.ArgumentParser(description="xView / DOTA → YOLO format dönüştürücü")
    ap.add_argument("--src", type=Path, default=DATA,
                    help="Ham veri klasörü; xview_raw/ ve dota_raw/ burada aranır")
    ap.add_argument("--out", type=Path, default=YOLO_DATA,
                    help="YOLO formatındaki çıktı klasörü")
    ap.add_argument("--datasets", nargs="+", choices=["xview", "dota"],
                    default=["xview", "dota"], help="Dönüştürülecek veri setleri")
    ap.add_argument("--overwrite-yaml", action="store_true",
                    help="Mevcut data.yaml dosyasının üzerine yaz")
    args = ap.parse_args()

    print(f"🚀 Başlıyoruz... kaynak={args.src} çıkış={args.out}")
    make_output_dirs(args.out)

    if "xview" in args.datasets:
        convert_xview(args.src, args.out)
    if "dota" in args.datasets:
        convert_dota(args.src, args.out)

    write_data_yaml(args.out, overwrite=args.overwrite_yaml)
    print("✅ Tüm işlemler bitti. Çıkış:", args.out)


if __name__ == "__main__":
    main()
