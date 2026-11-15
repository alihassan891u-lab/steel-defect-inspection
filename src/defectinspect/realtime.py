"""Real-time inspection from a camera, RTSP stream or video file (ONNX Runtime, CPU).

    python -m defectinspect.realtime --model weights/best.onnx --source 0              # USB camera
    python -m defectinspect.realtime --model weights/best.onnx --source rtsp://... --headless --log results/line1.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2

from .data import NEU_CLASSES
from .onnx_detector import OnnxDetector
from .quality import draw, judge, load_rules


def run(model: str, source: str, names: list[str], rules_path: str | None = None, imgsz: int = 640, headless: bool = False,
        log: str | None = None, max_frames: int | None = None, threads: int | None = None) -> dict:
    det = OnnxDetector(model, names, imgsz=imgsz, threads=threads)
    rules = load_rules(rules_path)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {source}")
    times: deque[float] = deque(maxlen=60)
    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    logf = open(log, "a") if log else None
    n = 0
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            insp = judge(det.predict(frame), frame.shape[:2], rules)
            insp.latency_ms = (time.perf_counter() - t0) * 1000
            times.append(insp.latency_ms)
            counts[insp.verdict] += 1
            n += 1
            if logf and insp.verdict != "PASS":
                logf.write(json.dumps({"frame": n, "ts": time.time(), **insp.to_dict()}) + "\n")
            if not headless:
                vis = draw(frame, insp)
                fps = 1000 / (sum(times) / len(times))
                cv2.putText(vis, f"{fps:.1f} FPS", (vis.shape[1] - 110, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("defect inspection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if logf:
            logf.close()
        if not headless:
            cv2.destroyAllWindows()
    mean = sum(times) / len(times) if times else 0
    return {"frames": n, "mean_latency_ms": round(mean, 2), "fps": round(1000 / mean, 1) if mean else 0, **counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--names", default=",".join(NEU_CLASSES))
    ap.add_argument("--rules", default="configs/rules.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.model, a.source, a.names.split(","), a.rules if Path(a.rules).exists() else None, a.imgsz,
                         a.headless, a.log, threads=a.threads), indent=2))


if __name__ == "__main__":
    main()
