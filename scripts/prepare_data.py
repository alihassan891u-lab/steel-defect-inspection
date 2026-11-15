"""Convert a VOC-format defect dataset (NEU-DET by default) to YOLO format with a stratified split.

    python scripts/prepare_data.py --raw data/raw/NEU-DET --out data/neu_yolo
    python scripts/prepare_data.py --raw data/raw/PCB --out data/pcb_yolo --classes missing_hole,mouse_bite,open_circuit,short,spur,spurious_copper
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from defectinspect.data import NEU_CLASSES, collect, write_yolo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--classes", default=",".join(NEU_CLASSES))
ap.add_argument("--val", type=float, default=0.15)
ap.add_argument("--test", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
samples = collect(a.raw)
if not samples:
    sys.exit(f"No image+XML pairs found under {a.raw}")
stats = write_yolo(samples, a.out, a.classes.split(","), a.val, a.test, a.seed)
print(f"{len(samples)} images -> {a.out}")
print(json.dumps(stats, indent=2))
