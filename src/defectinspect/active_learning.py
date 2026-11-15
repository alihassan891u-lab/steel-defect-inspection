"""Active learning: label the images the model is least sure about, not random ones.

Labelling steel or PCB defects needs an expert, so the question a factory asks
is "how few images do we need to label?". Strategies:

- random:       baseline
- uncertainty:  images whose detections are least confident (or with no detection at all, since every
                NEU-DET image contains a defect, a blank prediction is a likely miss)
- hybrid:       top uncertain candidates, then k-center greedy on image features for diversity, so we
                don't label 50 near-identical scratches
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np


def image_uncertainty(scores: list[float]) -> float:
    """1 = model knows nothing; 0 = confident. Uses the least-confident detection, plus a penalty for none."""
    if not scores:
        return 1.0
    s = np.clip(np.array(scores, dtype=float), 1e-6, 1 - 1e-6)
    least = 1.0 - s.max()
    # mean binary entropy of all detections, normalised to [0,1]
    ent = float(np.mean(-(s * np.log2(s) + (1 - s) * np.log2(1 - s))))
    return float(0.5 * least + 0.5 * ent)


def features(img: np.ndarray, size: int = 32) -> np.ndarray:
    """Cheap appearance embedding: normalised thumbnail + gradient histogram."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ang = (np.arctan2(gy, gx) % np.pi).ravel()
    mag = np.hypot(gx, gy).ravel()
    hist, _ = np.histogram(ang, bins=8, range=(0, np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([thumb / np.sqrt(len(thumb)), hist * 4]).astype(np.float32)


def k_center_greedy(feats: np.ndarray, k: int, seed_idx: list[int] | None = None) -> list[int]:
    """Pick k points that cover feature space (each new pick is the farthest from those already picked)."""
    n = len(feats)
    if k >= n:
        return list(range(n))
    chosen = list(seed_idx or [])
    dist = np.full(n, np.inf)
    for c in chosen:
        dist = np.minimum(dist, np.linalg.norm(feats - feats[c], axis=1))
    if not chosen:
        first = int(np.argmax(np.linalg.norm(feats - feats.mean(0), axis=1)))
        chosen.append(first)
        dist = np.linalg.norm(feats - feats[first], axis=1)
    picks = []
    while len(picks) < k:
        i = int(np.argmax(dist))
        picks.append(i)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[i], axis=1))
    return picks


def select(pool: list[str], k: int, strategy: str, uncertainty: dict[str, float] | None = None,
           feats: dict[str, np.ndarray] | None = None, seed: int = 0, oversample: int = 3) -> list[str]:
    if k <= 0 or not pool:
        return []
    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(pool, min(k, len(pool)))
    if uncertainty is None:
        raise ValueError(f"strategy '{strategy}' needs uncertainty scores")
    ranked = sorted(pool, key=lambda p: (-uncertainty.get(p, 1.0), p))
    if strategy == "uncertainty":
        return ranked[:k]
    if strategy == "hybrid":
        cand = ranked[: min(len(ranked), k * oversample)]
        if feats is None:
            return cand[:k]
        f = np.stack([feats[c] for c in cand])
        return [cand[i] for i in k_center_greedy(f, k)]
    raise ValueError(f"unknown strategy '{strategy}'")


# ----------------------------------------------------------------------------- experiment (GPU)
def run_experiment(dataset_dir: str, data_yaml: str, strategies=("random", "hybrid"), start_frac: float = 0.2,
                   step_frac: float = 0.1, rounds: int = 3, epochs: int = 40, model: str = "yolo11n.pt",
                   imgsz: int = 640, seed: int = 0, out: str = "results/active_learning.json") -> dict:
    """Learning curves: mAP@50 on the fixed test split vs. number of labelled images, per strategy."""
    import shutil

    from .train import train, val

    root = Path(dataset_dir)
    all_train = sorted(p.stem for p in (root / "images" / "train").iterdir())
    rng = random.Random(seed)
    start = rng.sample(all_train, round(len(all_train) * start_frac))
    step = round(len(all_train) * step_frac)
    ext = {p.stem: p.suffix for p in (root / "images" / "train").iterdir()}
    curves: dict[str, list[dict]] = {}

    for strat in strategies:
        labelled = list(start)
        curves[strat] = []
        for rnd in range(rounds + 1):
            work = root.parent / f"_al_{strat}_{rnd}"
            if work.exists():
                shutil.rmtree(work)
            for sub in ("images", "labels"):
                (work / sub / "train").mkdir(parents=True)
                for split in ("val", "test"):
                    shutil.copytree(root / sub / split, work / sub / split)
            for s in labelled:
                shutil.copy2(root / "images" / "train" / f"{s}{ext[s]}", work / "images" / "train")
                shutil.copy2(root / "labels" / "train" / f"{s}.txt", work / "labels" / "train")
            yaml = Path(data_yaml).read_text().replace(f"path: {root.resolve()}", f"path: {work.resolve()}")
            (work / "data.yaml").write_text(yaml)
            best = train(str(work / "data.yaml"), model, epochs, imgsz, name=f"al_{strat}_{rnd}", seed=seed)
            m = val(best, str(work / "data.yaml"), "test", imgsz)
            curves[strat].append({"round": rnd, "labelled": len(labelled), "map50": m["map50"], "map50_95": m["map50_95"]})
            print(strat, curves[strat][-1])
            if rnd == rounds:
                break
            pool = [p for p in all_train if p not in set(labelled)]
            unc, fts = {}, {}
            if strat != "random":
                from ultralytics import YOLO

                yolo = YOLO(str(best))
                for p in pool:
                    img_path = root / "images" / "train" / f"{p}{ext[p]}"
                    r = yolo.predict(str(img_path), conf=0.01, verbose=False, imgsz=imgsz)[0]
                    unc[p] = image_uncertainty(r.boxes.conf.cpu().numpy().tolist())
                    fts[p] = features(cv2.imread(str(img_path)))
            labelled += select(pool, step, strat, unc, fts, seed=seed + rnd)
            shutil.rmtree(work, ignore_errors=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(curves, indent=2))
    return curves
