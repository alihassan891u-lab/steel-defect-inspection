"""Operator UI:  DEFECT_MODEL=weights/best.onnx streamlit run app/streamlit_app.py"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from defectinspect.data import NEU_CLASSES
from defectinspect.onnx_detector import OnnxDetector
from defectinspect.quality import draw, judge, load_rules

st.set_page_config(page_title="Steel Defect Inspection", page_icon="🔍", layout="wide")
st.sidebar.title("🔍 Defect Inspection")
model = st.sidebar.text_input("ONNX model", os.getenv("DEFECT_MODEL", "weights/best.onnx"))
fail_conf = st.sidebar.slider("Fail confidence", 0.1, 0.9, 0.5, 0.05)
review_conf = st.sidebar.slider("Review confidence", 0.05, 0.5, 0.25, 0.05)
use_masks = st.sidebar.checkbox("Pixel masks (GrabCut fallback; SAM2 in the notebook)", False)

if not Path(model).exists():
    st.warning(f"No model at `{model}`. Train with the Colab notebook, export ONNX, then point this box to it.")
    st.stop()


@st.cache_resource
def get_det(path: str) -> OnnxDetector:
    return OnnxDetector(path, NEU_CLASSES)


det = get_det(model)
rules = load_rules("configs/rules.yaml" if Path("configs/rules.yaml").exists() else None)
rules.update(fail_conf=fail_conf, review_conf=review_conf)
files = st.file_uploader("Images of the part / strip", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
rows = []
for f in files or []:
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    t0 = time.perf_counter()
    dets = det.predict(img)
    if use_masks and dets:
        from defectinspect.segment import GrabCutRefiner

        dets = GrabCutRefiner().refine(img, dets)
    insp = judge(dets, img.shape[:2], rules)
    insp.latency_ms = (time.perf_counter() - t0) * 1000
    c1, c2 = st.columns([1, 1])
    c1.image(cv2.cvtColor(draw(img, insp), cv2.COLOR_BGR2RGB), caption=f"{f.name}: {insp.verdict}", use_container_width=True)
    c2.json(insp.to_dict())
    rows.append({"image": f.name, "verdict": insp.verdict, "defects": len(insp.findings), "ms": round(insp.latency_ms, 1)})
if rows:
    df = pd.DataFrame(rows)
    st.subheader("Batch summary")
    st.dataframe(df, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "inspection.csv")
