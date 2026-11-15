"""YOLO11 inference with ONNX Runtime only (no PyTorch, no Ultralytics).

This is what runs on the edge device / factory PC: export once with
``scripts/export.py``, then deploy with just onnxruntime + OpenCV + NumPy.

YOLO11 detection ONNX output: (1, 4 + num_classes, N) with rows
cx, cy, w, h (letterboxed input pixels) followed by per-class scores.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    cls: int
    name: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    mask: np.ndarray | None = None          # optional HxW bool mask (SAM2)

    @property
    def area_px(self) -> float:
        if self.mask is not None:
            return float(self.mask.sum())
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


def letterbox(img: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to size x size. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[py:py + nh, px:px + nw] = resized if resized.ndim == 3 else resized[..., None]
    return out, r, (px, py)


def preprocess(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lb, r, pad = letterbox(img, size)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), r, pad


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        ious = inter / np.maximum(a_i + a_r - inter, 1e-9)
        order = rest[ious <= iou_thr]
    return keep


def postprocess(out: np.ndarray, r: float, pad: tuple[int, int], orig_shape: tuple[int, int], names: list[str],
                conf: float = 0.25, iou_thr: float = 0.45, max_det: int = 100) -> list[Detection]:
    nc = len(names)
    pred = out[0].T if out.shape[1] == 4 + nc else out[0]  # -> (N, 4 + nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(pred)), cls]
    keep = scores >= conf
    pred, cls, scores = pred[keep], cls[keep], scores[keep]
    if not len(pred):
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / r
    H, W = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    dets = []
    for c in np.unique(cls):  # class-wise NMS
        idx = np.where(cls == c)[0]
        for k in nms(boxes[idx], scores[idx], iou_thr):
            j = idx[k]
            dets.append(Detection(int(c), names[int(c)] if int(c) < len(names) else str(c), float(scores[j]),
                                  tuple(float(v) for v in boxes[j])))
    dets.sort(key=lambda d: -d.score)
    return dets[:max_det]


class OnnxDetector:
    def __init__(self, model_path: str | Path, names: list[str], imgsz: int = 640, conf: float = 0.25, iou: float = 0.45,
                 threads: int | None = None, session=None):
        if session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if threads:
                opts.intra_op_num_threads = threads
            session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.names, self.imgsz, self.conf, self.iou = names, imgsz, conf, iou

    def predict(self, img: np.ndarray) -> list[Detection]:
        x, r, pad = preprocess(img, self.imgsz)
        out = self.session.run(None, {self.input_name: x})[0]
        return postprocess(out, r, pad, img.shape[:2], self.names, self.conf, self.iou)


def benchmark(detector, images: list[np.ndarray], warmup: int = 5, runs: int = 50) -> dict:
    """End-to-end latency (preprocess + inference + NMS) per image."""
    for i in range(warmup):
        detector.predict(images[i % len(images)])
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        detector.predict(images[i % len(images)])
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"runs": runs, "mean_ms": round(float(t.mean()), 2), "p50_ms": round(float(np.percentile(t, 50)), 2),
            "p95_ms": round(float(np.percentile(t, 95)), 2), "fps": round(1000 / float(t.mean()), 1)}
