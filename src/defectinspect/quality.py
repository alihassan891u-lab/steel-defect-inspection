"""Turn detections into a quality decision a line operator can act on.

    PASS    no defect above the review threshold
    REVIEW  something uncertain: show it to a person (and queue it for labelling: active learning in production)
    FAIL    a defect breaks a rule (class + confidence + size)

Rules are data (configs/rules.yaml) so quality engineers change limits without code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .onnx_detector import Detection

DEFAULT_RULES = {
    "review_conf": 0.25,          # below this, ignore
    "fail_conf": 0.5,             # at/above this, a defect counts
    "default": {"max_area_pct": 0.0},   # any confident defect fails unless a class says otherwise
    "classes": {},                # e.g. {"patches": {"max_area_pct": 1.5}} tolerates small patches
    "mm_per_px": None,            # set from camera calibration to report mm²
}

COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 255), (255, 128, 0), (0, 255, 0), (255, 0, 0)]


@dataclass
class Finding:
    defect: Detection
    area_pct: float
    area_mm2: float | None
    verdict: str  # FAIL | REVIEW | OK
    reason: str


@dataclass
class Inspection:
    verdict: str
    findings: list[Finding] = field(default_factory=list)
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "latency_ms": self.latency_ms, "defects": [
            {"class": f.defect.name, "confidence": round(f.defect.score, 3), "box": [round(v, 1) for v in f.defect.box],
             "area_pct": round(f.area_pct, 3), "area_mm2": None if f.area_mm2 is None else round(f.area_mm2, 2),
             "verdict": f.verdict, "reason": f.reason} for f in self.findings]}


def load_rules(path: str | Path | None) -> dict:
    rules = {**DEFAULT_RULES}
    if path:
        rules.update(yaml.safe_load(Path(path).read_text()) or {})
    return rules


def judge(dets: list[Detection], image_shape: tuple[int, int], rules: dict | None = None) -> Inspection:
    r = rules or DEFAULT_RULES
    total = image_shape[0] * image_shape[1]
    findings = []
    for d in dets:
        if d.score < r["review_conf"]:
            continue
        pct = 100.0 * d.area_px / total
        mm2 = d.area_px * r["mm_per_px"] ** 2 if r.get("mm_per_px") else None
        limit = {**r["default"], **r["classes"].get(d.name, {})}.get("max_area_pct", 0.0)
        if d.score < r["fail_conf"]:
            findings.append(Finding(d, pct, mm2, "REVIEW", f"uncertain {d.name} ({d.score:.2f}), needs a human check"))
        elif pct > limit:
            findings.append(Finding(d, pct, mm2, "FAIL", f"{d.name} covers {pct:.2f}% of the surface (limit {limit}%)"))
        else:
            findings.append(Finding(d, pct, mm2, "OK", f"{d.name} within tolerance ({pct:.2f}% <= {limit}%)"))
    verdicts = {f.verdict for f in findings}
    overall = "FAIL" if "FAIL" in verdicts else "REVIEW" if "REVIEW" in verdicts else "PASS"
    return Inspection(overall, findings)


def draw(img: np.ndarray, insp: Inspection, names: list[str] | None = None) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = max(out.shape[:2]) / 640
    th = max(1, round(2 * scale))
    for f in insp.findings:
        d = f.defect
        color = COLORS[d.cls % len(COLORS)]
        if d.mask is not None:
            overlay = out.copy()
            overlay[d.mask] = color
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th, cv2.LINE_AA if f.verdict != "REVIEW" else cv2.LINE_4)
        label = f"{d.name} {d.score:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * max(scale, 0.6), color, th)
    banner = {"PASS": (40, 160, 40), "REVIEW": (0, 160, 230), "FAIL": (30, 30, 200)}[insp.verdict]
    cv2.rectangle(out, (0, 0), (out.shape[1], int(22 * max(scale, 0.8))), banner, -1)
    txt = insp.verdict + (f"  {insp.latency_ms:.0f} ms" if insp.latency_ms else "")
    cv2.putText(out, txt, (6, int(16 * max(scale, 0.8))), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(scale, 0.8), (255, 255, 255), th)
    return out
