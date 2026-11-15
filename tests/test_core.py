import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth import CLASSES, make_dataset  # noqa: E402

from defectinspect.active_learning import features, image_uncertainty, k_center_greedy, select
from defectinspect.data import collect, parse_voc, read_yolo_labels, write_yolo
from defectinspect.metrics import GT, Det, evaluate, iou_matrix
from defectinspect.onnx_detector import Detection, OnnxDetector, benchmark, letterbox, postprocess
from defectinspect.quality import draw, judge
from defectinspect.segment import GrabCutRefiner
from defectinspect.train import evaluate_detector


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    meta = make_dataset(root)
    out = root / "yolo"
    stats = write_yolo(collect(root), out, CLASSES, val=0.2, test=0.2)
    return root, out, meta, stats


# ----------------------------------------------------------------------- data
def test_voc_to_yolo_roundtrip(dataset):
    root, out, meta, stats = dataset
    assert sum(s["images"] for s in stats.values()) == 18
    for split in ("train", "val", "test"):  # stratified: every class in every split
        assert set(stats[split]["boxes"]) == {"scratches", "patches", "inclusion"}
    m = {x["name"]: x for x in meta}
    for lbl in (out / "labels" / "train").iterdir():
        boxes = read_yolo_labels(lbl, 200, 200)
        truth = m[lbl.stem]["objs"]
        assert len(boxes) == len(truth)
        for (c, *b), (name, tb) in zip(boxes, truth):
            assert CLASSES[c] == name
            assert np.allclose(b, np.clip(tb, 0, 200), atol=0.01)
    yaml = (out / "data.yaml").read_text()
    assert "5: scratches" in yaml and "test: images/test" in yaml


def test_voc_aliases_and_clipping(tmp_path):
    x = tmp_path / "a.xml"
    x.write_text("<annotation><size><width>200</width><height>200</height></size><object><name>Rolled_in_scale</name>"
                 "<bndbox><xmin>-5</xmin><ymin>10</ymin><xmax>250</xmax><ymax>40</ymax></bndbox></object></annotation>")
    w, h, boxes = parse_voc(x)
    assert boxes[0].cls == "rolled-in_scale" and (boxes[0].x1, boxes[0].x2) == (0.0, 200.0)


# -------------------------------------------------------------------- metrics
def test_metrics_reference_values():
    g = [GT("a", 0, (10, 10, 50, 50)), GT("a", 1, (60, 60, 90, 90)), GT("b", 0, (0, 0, 20, 20))]
    perfect = [Det("a", 0, .9, (10, 10, 50, 50)), Det("a", 1, .8, (60, 60, 90, 90)), Det("b", 0, .7, (0, 0, 20, 20))]
    assert evaluate(perfect, g, ["x", "y"])["map50"] == 1.0
    # a confident false positive first, then one hit out of two boxes: pycocotools gives 51/101 * 0.5
    r = evaluate([Det("a", 0, .95, (100, 100, 120, 120)), Det("a", 0, .9, (10, 10, 50, 50))], g, ["x", "y"])
    assert r["per_class"]["x"]["ap50"] == pytest.approx(0.2525, abs=1e-4)
    # duplicate detection of the same box is a false positive
    dup = perfect + [Det("a", 0, .85, (11, 11, 50, 50))]
    assert evaluate(dup, g, ["x", "y"])["per_class"]["x"]["ap50"] < 1.0
    # slightly misplaced boxes keep mAP50 but lose mAP50-95
    off = [Det(d.image, d.cls, d.score, tuple(v + 2 for v in d.box)) for d in perfect]  # IoU 0.68-0.82
    r = evaluate(off, g, ["x", "y"])
    assert r["map50"] == 1.0 and r["map50_95"] < 0.95
    assert iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(1 / 3)


# ------------------------------------------------------------- ONNX runtime path
NAMES = CLASSES


def test_postprocess_maps_back_through_letterbox():
    img = np.zeros((200, 400, 3), np.uint8)  # wide image -> vertical padding
    _, r, pad = letterbox(img, 640)
    assert r == 1.6 and pad == (0, 160)
    out = np.zeros((1, 4 + 6, 4), np.float32)
    out[0, :4, 0] = [320, 320, 160, 80]; out[0, 4 + 5, 0] = 0.9   # scratch
    out[0, :4, 1] = [322, 321, 160, 80]; out[0, 4 + 5, 1] = 0.8   # duplicate -> NMS removes
    out[0, :4, 2] = [320, 320, 160, 80]; out[0, 4 + 2, 2] = 0.7   # same box, other class -> kept
    out[0, :4, 3] = [100, 400, 20, 20]; out[0, 4 + 1, 3] = 0.1    # below conf
    dets = postprocess(out, r, pad, (200, 400), NAMES)
    assert [d.name for d in dets] == ["scratches", "patches"]
    assert dets[0].box == pytest.approx((150, 75, 250, 125))


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: 'detects' the darkest region of the letterboxed image."""

    class _In:
        name = "images"

    def get_inputs(self):
        return [self._In()]

    def run(self, _, feeds):
        x = feeds["images"][0, 0]
        ys, xs = np.where(x < 0.3)
        out = np.zeros((1, 10, 1), np.float32)
        if len(xs):
            out[0, :4, 0] = [(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, xs.max() - xs.min(), ys.max() - ys.min()]
            out[0, 4 + 2, 0] = 0.9
        return [out]


def test_onnx_detector_end_to_end_and_benchmark(dataset):
    _, out, _, _ = dataset
    det = OnnxDetector(None, NAMES, session=FakeSession())
    img = np.full((200, 200, 3), 200, np.uint8)
    cv2.rectangle(img, (40, 50), (90, 120), (10, 10, 10), -1)
    d = det.predict(img)[0]
    assert d.name == "patches" and d.box == pytest.approx((40, 50, 90, 120), abs=1.5)
    b = benchmark(det, [img], warmup=1, runs=5)
    assert b["runs"] == 5 and b["fps"] > 0


def test_evaluate_detector_oracle_vs_noisy(dataset):
    _, out, _, _ = dataset

    class Oracle:
        def __init__(self, jitter=0.0, drop=False):
            self.jitter, self.drop, self.i = jitter, drop, 0

        def predict(self, img):
            p = self.files[self.i]
            self.i += 1
            dets = [Detection(c, CLASSES[c], 0.9, (x1 + self.jitter, y1, x2 + self.jitter, y2))
                    for c, x1, y1, x2, y2 in read_yolo_labels(out / "labels" / "test" / f"{p.stem}.txt", 200, 200)]
            return [] if self.drop and self.i % 2 == 0 else dets  # "drop": misses every other image

    files = sorted((out / "images" / "test").iterdir())
    for det, check in ((Oracle(), lambda r: r["map50"] == 1.0), (Oracle(drop=True), lambda r: r["map50"] < 1.0),
                       (Oracle(jitter=6), lambda r: r["map50_95"] < 0.9)):
        det.files = files
        assert check(evaluate_detector(det, out, "test", CLASSES))


# -------------------------------------------------------------------- quality
def mk(name, cls, score, box):
    return Detection(cls, name, score, box)


def test_judge_rules():
    rules = {"review_conf": 0.25, "fail_conf": 0.5, "default": {"max_area_pct": 0.0},
             "classes": {"patches": {"max_area_pct": 1.5}}, "mm_per_px": 0.1}
    assert judge([], (200, 200), rules).verdict == "PASS"
    assert judge([mk("scratches", 5, 0.1, (0, 0, 10, 10))], (200, 200), rules).verdict == "PASS"      # noise
    assert judge([mk("scratches", 5, 0.4, (0, 0, 10, 10))], (200, 200), rules).verdict == "REVIEW"
    assert judge([mk("scratches", 5, 0.8, (0, 0, 10, 10))], (200, 200), rules).verdict == "FAIL"
    small = judge([mk("patches", 2, 0.9, (0, 0, 20, 20))], (200, 200), rules)                         # 1% of surface
    assert small.verdict == "PASS" and small.findings[0].verdict == "OK"
    assert small.findings[0].area_mm2 == pytest.approx(4.0)
    assert judge([mk("patches", 2, 0.9, (0, 0, 40, 40))], (200, 200), rules).verdict == "FAIL"        # 4%
    d = judge([mk("scratches", 5, 0.8, (0, 0, 10, 10)), mk("patches", 2, 0.3, (50, 50, 60, 60))], (200, 200), rules).to_dict()
    assert d["verdict"] == "FAIL" and [x["verdict"] for x in d["defects"]] == ["FAIL", "REVIEW"]


def test_draw_returns_annotated_image():
    img = np.full((200, 200), 128, np.uint8)
    insp = judge([mk("scratches", 5, 0.8, (10, 10, 60, 60))], img.shape)
    out = draw(img, insp)
    assert out.shape == (200, 200, 3) and not np.array_equal(out[0, 0], [128, 128, 128])  # banner drawn


# ------------------------------------------------------------- segmentation
def test_grabcut_mask_tracks_the_defect():
    rng = np.random.default_rng(1)
    img = np.clip(125 + rng.normal(0, 4, (200, 200)), 0, 255).astype(np.uint8)
    cv2.ellipse(img, (100, 100), (30, 18), 0, 0, 360, 40, -1)
    truth = np.zeros_like(img, bool)
    cv2.ellipse(truth.view(np.uint8), (100, 100), (30, 18), 0, 0, 360, 1, -1)
    d = GrabCutRefiner().refine(img, [mk("patches", 2, 0.9, (60, 72, 140, 128))])[0]
    inter, union = (d.mask & truth).sum(), (d.mask | truth).sum()
    assert inter / union > 0.6
    assert d.area_px < (140 - 60) * (128 - 72)  # mask area is tighter than the box


# ----------------------------------------------------------- active learning
def test_uncertainty_scoring():
    assert image_uncertainty([]) == 1.0                    # nothing found on a part that has a defect
    assert image_uncertainty([0.95]) < image_uncertainty([0.55])
    assert image_uncertainty([0.99, 0.98]) < image_uncertainty([0.6])


def test_selection_strategies(dataset):
    pool = [f"img{i}" for i in range(10)]
    unc = {p: i / 10 for i, p in enumerate(pool)}
    assert select(pool, 3, "random", seed=1) == select(pool, 3, "random", seed=1)
    assert select(pool, 3, "uncertainty", unc) == ["img9", "img8", "img7"]
    # hybrid: the 6 most uncertain images are near-duplicates in two groups -> picks span both groups
    feats = {p: np.array([0.0, 0.0]) + 0.01 * i for i, p in enumerate(pool)}
    for p in ("img9", "img8", "img7"):
        feats[p] = np.array([5.0, 5.0]) + 0.01 * int(p[-1])
    for p in ("img6", "img5", "img4"):
        feats[p] = np.array([-5.0, 0.0]) + 0.01 * int(p[-1])
    picks = select(pool, 2, "hybrid", unc, feats, oversample=3)
    groups = {"a" if feats[p][0] > 0 else "b" for p in picks}
    assert groups == {"a", "b"}
    with pytest.raises(ValueError):
        select(pool, 2, "uncertainty")


def test_features_and_k_center(dataset):
    root, out, _, _ = dataset
    imgs = sorted((out / "images" / "train").iterdir())
    f = np.stack([features(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in imgs])
    assert f.shape[0] == len(imgs) and np.isfinite(f).all()
    assert len(set(k_center_greedy(f, 4))) == 4
