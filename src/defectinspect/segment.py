"""Turn detection boxes into pixel masks, so defects can be measured (area, % of surface).

- SamRefiner: SAM2 prompted with each YOLO box (Ultralytics' SAM2 wrapper). Best quality, needs PyTorch.
- GrabCutRefiner: OpenCV GrabCut inside each box. No deep learning; runs anywhere as a fallback.
"""
from __future__ import annotations

import cv2
import numpy as np

from .onnx_detector import Detection


class SamRefiner:
    def __init__(self, model: str = "sam2.1_b.pt"):
        from ultralytics import SAM

        self.sam = SAM(model)

    def refine(self, img: np.ndarray, dets: list[Detection]) -> list[Detection]:
        if not dets:
            return dets
        res = self.sam(img, bboxes=[list(d.box) for d in dets], verbose=False)[0]
        masks = res.masks.data.cpu().numpy().astype(bool) if res.masks is not None else []
        for d, m in zip(dets, masks):
            x1, y1, x2, y2 = (int(round(v)) for v in d.box)
            clipped = np.zeros_like(m)
            clipped[y1:y2, x1:x2] = m[y1:y2, x1:x2]  # a defect mask never leaves its box
            d.mask = clipped
        return dets


class GrabCutRefiner:
    def __init__(self, iterations: int = 3):
        self.iterations = iterations

    def refine(self, img: np.ndarray, dets: list[Detection]) -> list[Detection]:
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = (int(round(v)) for v in d.box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                d.mask = None
                continue
            mask = np.zeros((h, w), np.uint8)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(bgr, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, self.iterations, cv2.GC_INIT_WITH_RECT)
                m = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
            except cv2.error:
                m = np.zeros((h, w), bool)
                m[y1:y2, x1:x2] = True
            d.mask = m
        return dets
