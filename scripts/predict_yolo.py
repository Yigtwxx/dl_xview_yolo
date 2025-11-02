from pathlib import Path
import argparse, glob, io, re, datetime, webbrowser, shutil
import numpy as np
from PIL import Image

import torch
from ultralytics import YOLO

try:
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    import uvicorn
    FASTAPI_OK = True
except Exception:
    FASTAPI_OK = False



ROOT      = Path(r"C:\Users\Asus\Desktop\dl_xview")
RUNS      = ROOT / "runs"
YOLO_DATA = ROOT / "yolo_data"
DATA_DIR  = ROOT / "data"
TEST_DIR  = DATA_DIR / "test_images"
TEST_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR = Path(r"C:\Users\Asus\Desktop\dl_xview\ui")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_weights() -> Path:
    bests = sorted(glob.glob(str(RUNS / "train" / "**" / "weights" / "best.pt"), recursive=True))
    lasts = sorted(glob.glob(str(RUNS / "train" / "**" / "weights" / "last.pt"),  recursive=True))
    if bests:
        return Path(bests[-1])
    if lasts:
        return Path(lasts[-1])
    raise FileNotFoundError("Ağırlık yok: runs/train/**/weights/{best.pt,last.pt} bulunamadı.")


def dir_has_images(p: Path) -> bool:
    return p.exists() and any((f.is_file() and f.suffix.lower() in IMG_EXTS) for f in p.iterdir())


def auto_find_source() -> Path:
    candidates = [
        TEST_DIR,
        YOLO_DATA / "images" / "val",
        YOLO_DATA / "images" / "train",
        ROOT / "data" / "val_images",
        ROOT / "data" / "train_images",
    ]
    for p in candidates:
        if dir_has_images(p):
            return p
    return TEST_DIR


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE)
    name = name.strip(" .")
    return name or "file"


def ndarray_to_png_bytes(arr_bgr_or_rgb: np.ndarray) -> bytes:
    if arr_bgr_or_rgb.ndim == 3 and arr_bgr_or_rgb.shape[2] == 3:
        img_rgb = arr_bgr_or_rgb[:, :, ::-1]
    else:
        img_rgb = arr_bgr_or_rgb
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def run_cli_predict(model: YOLO, source: str, imgsz: int, conf: float, device: str,
                    project: str, name: str, save_txt: bool, save_conf: bool, show: bool):
    print("\n======= YOLO PREDICT (CLI) =======")
    print(f"Source        : {source}")
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        project=project,
        name=name,
        verbose=True,
        show=show,
    )
    n_preds = sum((len(r.boxes) if hasattr(r, "boxes") else 0) for r in results)
    print(f"\n[PREDICT BİTTİ] Toplam kutu: {n_preds}")


def build_app(model: YOLO, imgsz: int, conf: float, device: str):
    app = FastAPI(title="dl_xview • YOLO Tahmin UI")

    ui_path = UI_DIR / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index():
        if ui_path.exists():
            return HTMLResponse(ui_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>UI dosyası bulunamadı: ui/index.html</h1>", status_code=500)


    @app.post("/predict")
    async def predict_endpoint(file: UploadFile = File(...)):
        try:
            raw = await file.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            orig = sanitize_filename(file.filename or "upload.jpg")
            fname = f"user_{ts}_{orig}"
            save_path = TEST_DIR / fname
            img.save(save_path)

            results = model.predict(
                source=np.array(img),
                imgsz=imgsz,
                conf=conf,
                device=device,
                save=True,
                project=str(RUNS / "predict"),
                name="exp",
                verbose=False,
            )

            plot_bgr = results[0].plot()
            png = ndarray_to_png_bytes(plot_bgr)
            return StreamingResponse(io.BytesIO(png), media_type="image/png")
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return app


def main():
    ap = argparse.ArgumentParser(description="YOLO tahmin + UI")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf",  type=float, default=0.25)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--project", type=str, default=str(RUNS / "predict"))
    ap.add_argument("--name",    type=str, default="exp")
    ap.add_argument("--save-txt", action="store_true")
    ap.add_argument("--save-conf", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--source", type=str, default=None)
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    weights = find_weights()
    model = YOLO(str(weights))

    if args.no_ui:
        src = args.source or str(auto_find_source())
        if src and Path(src).exists() and is_image_file(Path(src)):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = TEST_DIR / f"cli_{ts}_{Path(src).name}"
            shutil.copy2(src, dst)
        run_cli_predict(
            model=model,
            source=src,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            project=args.project,
            name=args.name,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            show=args.show,
        )
        return

    if not FASTAPI_OK:
        raise RuntimeError("UI için fastapi + uvicorn kur: pip install fastapi uvicorn")

    app = build_app(model, imgsz=args.imgsz, conf=args.conf, device=device)
    url = f"http://{args.host}:{args.port}"
    print(f"📦 Model: {weights}")
    print(f"🚀 UI açılıyor: {url}")
    try:
      webbrowser.open(url)
    except Exception:
      pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
