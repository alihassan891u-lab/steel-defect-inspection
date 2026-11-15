"""CPU benchmark + accuracy of the exported ONNX model, scored with our own evaluator.

    python scripts/benchmark_onnx.py --model runs/defects/train/weights/best.onnx --data data/neu_yolo --threads 4
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from defectinspect.data import NEU_CLASSES  # noqa: E402
from defectinspect.onnx_detector import OnnxDetector, benchmark  # noqa: E402
from defectinspect.train import evaluate_detector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--names", default=",".join(NEU_CLASSES))
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--runs", type=int, default=100)
a = ap.parse_args()
names = a.names.split(",")
det = OnnxDetector(a.model, names, imgsz=a.imgsz, threads=a.threads, conf=0.001)
acc = evaluate_detector(det, a.data, "test", names)
det.conf = 0.25
imgs = [cv2.imread(str(p)) for p in sorted((a.data / "images" / "test").iterdir())[:50]]
speed = benchmark(det, imgs, runs=a.runs)
res = {"model": a.model, "cpu": platform.processor() or platform.machine(), "threads": a.threads, "imgsz": a.imgsz,
       "accuracy_test": acc, "speed": speed}
Path("results").mkdir(exist_ok=True)
Path(f"results/benchmark_{Path(a.model).stem}_{a.imgsz}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
