import numpy as np

from locus.eval import grounding as g


def test_iou_basic():
    a = np.array([[0, 0, 1, 1], [0, 0, 0.5, 1], [0, 0, 1, 1], [np.nan] * 4, [0.2, 0.2, 0.2, 0.2]])
    b = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [0, 0, 1, 1], [0.2, 0.2, 0.2, 0.2]])
    np.testing.assert_allclose(g.iou_xyxy(a, b), [1, 0.5, 0, 0, 0])


def test_metrics_thresholds_and_invalid():
    iou = np.array([1.0, 0.8, 0.6, 0.5, 0.0])
    m = g.metrics(iou, invalid=1)
    assert m["n"] == 5 and m["invalid"] == 1
    assert m["acc50"] == 0.8 and m["acc75"] == 0.4 and m["acc90"] == 0.2
    assert abs(m["miou"] - 0.58) < 1e-9
    want = np.mean([(iou >= t).mean() for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]])
    assert abs(m["macc"] - want) < 1e-9 and "conf_ece" not in m


def test_auroc():
    assert g.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert g.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert g.auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    assert np.isnan(g.auroc([0.1, 0.2], [1, 1]))


def test_ece():
    assert g.ece([1.0, 1.0, 1.0, 1.0], [1, 1, 1, 1]) == 0.0
    assert abs(g.ece([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9  # one bin: |0.5 - 0.9|


def test_report_with_conf_and_leak(monkeypatch):
    monkeypatch.setattr(g, "train_ann_ids", lambda: frozenset({1, 2}))
    iou = np.array([0.9, 0.9, 0.1, 0.1])
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    r = g.report(iou, conf, ann_id=[1, 3, 2, 4])
    assert r["leaked_frac"] == 0.5
    assert r["all"]["acc50"] == 0.5 and r["leaked"]["acc50"] == 0.5 and r["clean"]["acc50"] == 0.5
    assert r["all"]["conf_auroc"] == 1.0 and r["clean"]["n"] == 2
