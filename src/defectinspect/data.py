"""Dataset preparation: NEU-DET (Pascal VOC XML) -> YOLO format.

NEU-DET: 1,800 grayscale 200x200 hot-rolled steel images, 6 defect classes,
Pascal VOC boxes. Downloads come in several folder layouts (ANNOTATIONS/IMAGES,
train/validation, ...), so we search recursively and pair each XML with the
image of the same file stem.

Also works for any VOC-format defect dataset (e.g. PCB defects) via --classes.
"""
from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_ALIASES = {"rolled_in_scale": "rolled-in_scale", "pitted": "pitted_surface", "crack": "crazing"}


@dataclass
class Box:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Sample:
    stem: str
    image: Path
    width: int
    height: int
    boxes: list[Box]


def parse_voc(xml_path: Path) -> tuple[int, int, list[Box]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    boxes = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip().lower().replace(" ", "_")
        name = _ALIASES.get(name, name)
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            boxes.append(Box(name, x1, y1, x2, y2))
    return w, h, boxes


def collect(raw_dir: Path) -> list[Sample]:
    images = {p.stem: p for p in raw_dir.rglob("*") if p.suffix.lower() in IMG_EXT}
    samples = []
    for xml in sorted(raw_dir.rglob("*.xml")):
        img = images.get(xml.stem)
        if img is None:
            continue
        w, h, boxes = parse_voc(xml)
        samples.append(Sample(xml.stem, img, w, h, boxes))
    return samples


def to_yolo_lines(s: Sample, classes: list[str]) -> list[str]:
    out = []
    for b in s.boxes:
        if b.cls not in classes:
            continue
        cx, cy = (b.x1 + b.x2) / 2 / s.width, (b.y1 + b.y2) / 2 / s.height
        bw, bh = (b.x2 - b.x1) / s.width, (b.y2 - b.y1) / s.height
        out.append(f"{classes.index(b.cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


def stratified_split(samples: list[Sample], val: float = 0.15, test: float = 0.15, seed: int = 0) -> dict[str, list[Sample]]:
    """Split per dominant class so every class appears in every split in the same proportion."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        main = Counter(b.cls for b in s.boxes).most_common(1)
        groups[main[0][0] if main else "_none"].append(s)
    rng = random.Random(seed)
    split: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for _, items in sorted(groups.items()):
        items = sorted(items, key=lambda s: s.stem)
        rng.shuffle(items)
        n_test, n_val = round(len(items) * test), round(len(items) * val)
        split["test"] += items[:n_test]
        split["val"] += items[n_test:n_test + n_val]
        split["train"] += items[n_test + n_val:]
    return split


def write_yolo(samples: list[Sample], out_dir: Path, classes: list[str] | None = None, val: float = 0.15,
               test: float = 0.15, seed: int = 0) -> dict:
    classes = classes or NEU_CLASSES
    split = stratified_split(samples, val, test, seed)
    stats = {}
    for name, items in split.items():
        (out_dir / "images" / name).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / name).mkdir(parents=True, exist_ok=True)
        boxes = Counter()
        for s in items:
            shutil.copy2(s.image, out_dir / "images" / name / f"{s.stem}{s.image.suffix.lower()}")
            lines = to_yolo_lines(s, classes)
            (out_dir / "labels" / name / f"{s.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            boxes.update(b.cls for b in s.boxes)
        stats[name] = {"images": len(items), "boxes": dict(sorted(boxes.items()))}
    yaml = [f"path: {out_dir.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(classes)]
    (out_dir / "data.yaml").write_text("\n".join(yaml) + "\n")
    return stats


def read_yolo_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    """YOLO txt -> [(cls, x1, y1, x2, y2)] in pixels."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        c, cx, cy, w, h = line.split()[:5]
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        out.append((int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out
