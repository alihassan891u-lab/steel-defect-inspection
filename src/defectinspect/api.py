"""REST API for a production line / MES integration.

    DEFECT_MODEL=weights/best.onnx uvicorn defectinspect.api:app --host 0.0.0.0 --port 8000
    curl -F "file=@part.jpg" localhost:8000/inspect
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .data import NEU_CLASSES
from .onnx_detector import OnnxDetector
from .quality import draw, judge, load_rules

app = FastAPI(title="Defect Inspection API", version="0.1.0")
_state: dict = {}


def detector():
    if "det" not in _state:
        path = os.getenv("DEFECT_MODEL", "weights/best.onnx")
        if not Path(path).exists():
            raise HTTPException(503, f"Model not found at {path}. Set DEFECT_MODEL to an exported .onnx file.")
        names = os.getenv("DEFECT_CLASSES", ",".join(NEU_CLASSES)).split(",")
        _state["det"] = OnnxDetector(path, names, imgsz=int(os.getenv("DEFECT_IMGSZ", "640")))
        rules = os.getenv("DEFECT_RULES", "configs/rules.yaml")
        _state["rules"] = load_rules(rules if Path(rules).exists() else None)
    return _state["det"], _state["rules"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "det" in _state}


@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...), annotate: bool = True):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Not a readable image")
    det, rules = detector()
    t0 = time.perf_counter()
    insp = judge(det.predict(img), img.shape[:2], rules)
    insp.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    out = insp.to_dict()
    if annotate:
        ok, png = cv2.imencode(".png", draw(img, insp))
        out["annotated_png_base64"] = base64.b64encode(png.tobytes()).decode() if ok else None
    return out
