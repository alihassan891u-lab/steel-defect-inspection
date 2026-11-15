"""Detection metrics, implemented independently of Ultralytics.

Used to score ANY model output (PyTorch, ONNX FP32, OpenVINO INT8) with the
same code, so the accuracy cost of quantising for the edge is measured, not
assumed. COCO-style: 101-point interpolated AP, mAP@50 and mAP@50:95.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Det:
    image: str
    cls: int
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 pixels


@dataclass
class GT:
    image: str
    cls: int
    box: tuple[float, float, float, float]


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between Nx4 and Mx4 boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP, same definition as pycocotools.

    For each recall level r in {0, 0.01, ..., 1}: the best precision achieved at recall >= r (0 if never reached).
    """
    if len(recall) == 0:
        return 0.0
    env = np.maximum.accumulate(precision[::-1])[::-1]  # precision envelope
    x = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, x, side="left")
    q = np.where(idx < len(env), env[np.minimum(idx, len(env) - 1)], 0.0)
    return float(q.mean())


def ap_for_class(dets: list[Det], gts: list[GT], cls: int, thr: float) -> tuple[float, int, np.ndarray, np.ndarray]:
    g_by_img: dict[str, list[tuple]] = {}
    for g in gts:
        if g.cls == cls:
            g_by_img.setdefault(g.image, []).append(g.box)
    n_gt = sum(len(v) for v in g_by_img.values())
    d = sorted((x for x in dets if x.cls == cls), key=lambda x: -x.score)
    if n_gt == 0:
        return float("nan"), 0, np.array([]), np.array([])
    used = {k: np.zeros(len(v), bool) for k, v in g_by_img.items()}
    tp = np.zeros(len(d))
    for i, det in enumerate(d):
        cands = g_by_img.get(det.image, [])
        best, best_j = 0.0, -1
        for j, gb in enumerate(cands):
            o = iou(det.box, gb)
            if o > best:
                best, best_j = o, j
        if best >= thr and not used[det.image][best_j]:
            used[det.image][best_j] = True
            tp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    return average_precision(recall, precision), n_gt, recall, precision


def evaluate(dets: list[Det], gts: list[GT], class_names: list[str]) -> dict:
    thrs = np.round(np.arange(0.5, 0.96, 0.05), 2)
    per_class = {}
    for c, name in enumerate(class_names):
        aps = [ap_for_class(dets, gts, c, t)[0] for t in thrs]
        ap50, n_gt, rec, prec = ap_for_class(dets, gts, c, 0.5)
        if n_gt == 0:
            continue
        per_class[name] = {"ap50": round(ap50, 4), "ap50_95": round(float(np.mean(aps)), 4), "gt_boxes": n_gt,
                           "max_recall": round(float(rec[-1]) if len(rec) else 0.0, 4)}
    if not per_class:
        return {"map50": 0.0, "map50_95": 0.0, "per_class": {}}
    return {"map50": round(float(np.mean([v["ap50"] for v in per_class.values()])), 4),
            "map50_95": round(float(np.mean([v["ap50_95"] for v in per_class.values()])), 4),
            "per_class": per_class}
