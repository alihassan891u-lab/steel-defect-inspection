"""Synthetic steel-strip images with known defects, written in NEU-DET's VOC layout (for tests only)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def texture(rng: np.random.Generator, size: int = 200) -> np.ndarray:
    rows = rng.normal(0, 6, (size, 1))
    img = 125 + rows + rng.normal(0, 5, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def add_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = img.shape
    if kind == "scratches":
        x = int(rng.integers(20, w - 20))
        y1, y2 = int(rng.integers(5, 60)), int(rng.integers(120, h - 5))
        cv2.line(img, (x, y1), (x + int(rng.integers(-10, 10)), y2), 40, 2)
        return max(0, x - 12), y1 - 2, min(w, x + 12), y2 + 2
    if kind == "patches":
        cx, cy = int(rng.integers(50, w - 50)), int(rng.integers(50, h - 50))
        ax, ay = int(rng.integers(18, 35)), int(rng.integers(14, 28))
        cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, 55, -1)
        return cx - ax, cy - ay, cx + ax, cy + ay
    # inclusion: cluster of small dark spots
    cx, cy = int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30))
    pts = [(cx + int(rng.integers(-12, 12)), cy + int(rng.integers(-12, 12))) for _ in range(5)]
    for p in pts:
        cv2.circle(img, p, 3, 50, -1)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4


def voc_xml(name: str, w: int, h: int, objs: list[tuple[str, tuple[int, int, int, int]]]) -> str:
    body = "".join(f"<object><name>{c}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax>"
                   f"<ymax>{b[3]}</ymax></bndbox></object>" for c, b in objs)
    return f"<annotation><filename>{name}.jpg</filename><size><width>{w}</width><height>{h}</height><depth>1</depth></size>{body}</annotation>"


def make_dataset(root: Path, n_per_class: int = 6, seed: int = 0) -> list[dict]:
    """Writes root/NEU-DET/ANNOTATIONS/*.xml and root/NEU-DET/IMAGES/*.jpg (NEU-DET layout)."""
    rng = np.random.default_rng(seed)
    (root / "NEU-DET" / "ANNOTATIONS").mkdir(parents=True, exist_ok=True)
    (root / "NEU-DET" / "IMAGES").mkdir(parents=True, exist_ok=True)
    meta = []
    for kind in ("scratches", "patches", "inclusion"):
        for i in range(n_per_class):
            img = texture(rng)
            objs = [(kind, add_defect(img, kind, rng))]
            if i % 3 == 0:  # some images with a second defect
                objs.append((kind, add_defect(img, kind, rng)))
            name = f"{kind}_{i + 1}"
            cv2.imwrite(str(root / "NEU-DET" / "IMAGES" / f"{name}.jpg"), img)
            (root / "NEU-DET" / "ANNOTATIONS" / f"{name}.xml").write_text(voc_xml(name, 200, 200, objs))
            meta.append({"name": name, "objs": objs})
    return meta
