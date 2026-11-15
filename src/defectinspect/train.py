"""Training, evaluation and export with Ultralytics YOLO11 (needs a GPU for sensible speed).

    python -m defectinspect.train --data data/neu_yolo/data.yaml --model yolo11s.pt --epochs 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .data import read_yolo_labels
from .metrics import GT, Det, evaluate

# Augmentation suited to steel / PCB inspection: no colour jitter (grayscale texture), flips are
# physically valid, mosaic helps small defects, mild scale.
AUG = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, fliplr=0.5, flipud=0.5, degrees=0.0, scale=0.3, mosaic=1.0, mixup=0.0)


def train(data: str, model: str = "yolo11s.pt", epochs: int = 100, imgsz: int = 640, batch: int = 16,
          project: str = "runs/defects", name: str = "train", seed: int = 0, fraction: float = 1.0, **kw) -> Path:
    from ultralytics import YOLO

    m = YOLO(model)
    m.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, project=project, name=name, seed=seed,
            deterministic=True, fraction=fraction, exist_ok=True, patience=30, plots=True, **{**AUG, **kw})
    return Path(project) / name / "weights" / "best.pt"


def val(weights: str | Path, data: str, split: str = "test", imgsz: int = 640) -> dict:
    from ultralytics import YOLO

    r = YOLO(str(weights)).val(data=data, split=split, imgsz=imgsz, plots=True)
    names = r.names
    per = {names[int(c)]: round(float(ap), 4) for c, ap in zip(r.box.ap_class_index, r.box.ap50)}
    return {"map50": round(float(r.box.map50), 4), "map50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4), "ap50_per_class": per,
            "speed_ms": {k: round(v, 2) for k, v in r.speed.items()}}


def evaluate_detector(detector, dataset_dir: str | Path, split: str, class_names: list[str]) -> dict:
    """Score any detector with ``.predict(img) -> [Detection]`` (ONNX FP32, INT8...) with our own evaluator."""
    root = Path(dataset_dir)
    dets, gts = [], []
    for img_path in sorted((root / "images" / split).iterdir()):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for c, x1, y1, x2, y2 in read_yolo_labels(root / "labels" / split / f"{img_path.stem}.txt", w, h):
            gts.append(GT(img_path.stem, c, (x1, y1, x2, y2)))
        for d in detector.predict(img):
            dets.append(Det(img_path.stem, d.cls, d.score, d.box))
    return evaluate(dets, gts, class_names)


def export(weights: str | Path, data: str, imgsz: int = 640) -> dict[str, str]:
    """ONNX (FP32) for onnxruntime, OpenVINO INT8 for Intel CPUs / edge boxes."""
    from ultralytics import YOLO

    m = YOLO(str(weights))
    out = {"onnx": m.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)}
    try:
        out["openvino_int8"] = m.export(format="openvino", imgsz=imgsz, int8=True, data=data)
    except Exception as e:  # OpenVINO not available on every platform
        out["openvino_int8_error"] = str(e)
    return {k: str(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="train")
    a = ap.parse_args()
    best = train(a.data, a.model, a.epochs, a.imgsz, a.batch, name=a.name)
    res = {"weights": str(best), "test": val(best, a.data, "test", a.imgsz), "export": export(best, a.data, a.imgsz)}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{a.name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
