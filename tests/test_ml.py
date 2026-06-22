"""ML 분류기 + 시간분할 + 이진 지표 테스트."""

import numpy as np

from surveillance.detection.curves import binary_metrics, curve_summary
from surveillance.detection.ml import (
    FEATURE_COLS,
    build_labeled_frame,
    time_split,
    train_and_score,
)
from surveillance.detection.evaluate import build_truth
from surveillance.features.account_features import build_features
from surveillance.generator.dataset import DatasetConfig, generate_dataset


def _labeled(seed=42, duration=20_000):
    r = generate_dataset(DatasetConfig(seed=seed, duration=duration))
    f = build_features(r.events)
    t = build_truth(r.events, r.labels)
    return build_labeled_frame(f, t)


def test_time_split_has_no_window_overlap():
    """lookahead 누설 방지: train의 모든 윈도우가 test보다 앞서야 한다."""
    labeled = _labeled()
    train, test = time_split(labeled, train_fraction=0.7)
    assert len(train) > 0 and len(test) > 0
    assert train["window"].max() < test["window"].min()


def test_ml_features_exclude_identifiers_and_labels():
    """모델 입력 피처에 식별자/라벨이 섞이지 않아야 한다(누설 차단)."""
    for col in ("account_id", "window", "true_label", "is_manip", "episode_id"):
        assert col not in FEATURE_COLS


def test_train_and_score_produces_valid_probabilities():
    labeled = _labeled()
    train, test = time_split(labeled)
    res = train_and_score(train, test)
    assert "score" in res.test and "ml_pred" in res.test
    assert res.test["score"].between(0.0, 1.0).all()
    assert res.train_positives > 0 and res.test_positives > 0


def test_binary_metrics_hand_calc():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])  # TP=2, FN=1, TN=1, FP=1
    m = binary_metrics(y_true, y_pred)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 1)
    assert m.precision == 2 / 3 and m.recall == 2 / 3


def test_curve_summary_handles_single_class():
    s = curve_summary(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(s["roc_auc"])
