"""피처 → 룰 탐지 → 평가 파이프라인 테스트."""

import pandas as pd

from surveillance.detection.evaluate import build_truth, evaluate
from surveillance.detection.rules import RuleConfig, predict
from surveillance.features.account_features import build_features
from surveillance.features.windows import DEFAULT_WINDOW_MS
from surveillance.generator.dataset import DatasetConfig, generate_dataset


# ── 평가 지표 손계산 일치 ────────────────────────────────────────
def test_confusion_matrix_matches_hand_calc():
    pred = pd.DataFrame(
        {
            "account_id": ["A", "B", "C", "D"],
            "window": [0, 0, 0, 0],
            "predicted_label": ["SPOOFING", "NORMAL", "WASH_TRADING", "SPOOFING"],
            "is_alert": [True, False, True, True],
        }
    )
    truth = pd.DataFrame(
        {
            "account_id": ["A", "C", "B"],
            "window": [0, 0, 0],
            "true_label": ["SPOOFING", "WASH_TRADING", "SPOOFING"],
        }
    )
    # A: TP, B: 실제조작이나 정상예측 → FN, C: TP, D: 정상인데 알림 → FP
    rep = evaluate(pred, truth)
    assert (rep.tp, rep.fp, rep.fn, rep.tn) == (2, 1, 1, 0)
    assert rep.precision == 2 / 3
    assert rep.recall == 2 / 3


# ── 피처 정확성 ──────────────────────────────────────────────────
def test_cancel_ratio_feature():
    from surveillance.generator.events import EventType, OrderEvent, Side

    events = [
        OrderEvent(0, 0, EventType.NEW, 1, "A", "X", Side.BUY, 100, 5),
        OrderEvent(1, 0, EventType.NEW, 2, "A", "X", Side.BUY, 100, 5),
        OrderEvent(2, 10, EventType.CANCEL, 1, "A", "X", Side.BUY, 100, 5),
    ]
    feats = build_features(events, window_ms=DEFAULT_WINDOW_MS)
    row = feats[feats["account_id"] == "A"].iloc[0]
    assert row["num_new"] == 2 and row["num_cancel"] == 1
    assert row["cancel_ratio"] == 0.5


# ── 엔드투엔드: 주입한 패턴을 실제로 잡아내는가 ──────────────────────
def test_pipeline_detects_injected_patterns():
    result = generate_dataset(DatasetConfig(seed=42, duration=20_000))
    features = build_features(result.events)
    predictions = predict(features, RuleConfig())
    truth = build_truth(result.events, result.labels)
    rep = evaluate(predictions, truth)

    assert rep.n_positives > 0
    # 룰이 의미 있는 재현율/정밀도를 내야 한다(완벽일 필요는 없음)
    assert rep.recall >= 0.5
    assert rep.precision >= 0.5
    # 두 패턴 모두 어느 정도 잡아야 한다
    assert rep.per_class["SPOOFING"]["recall"] > 0
    assert rep.per_class["WASH_TRADING"]["recall"] > 0


def test_features_have_no_label_columns():
    """피처 테이블에 라벨/에피소드 정보가 새어들지 않아야 한다."""
    result = generate_dataset(DatasetConfig(seed=3, duration=4000))
    features = build_features(result.events)
    leaked = {"label", "episode_id", "true_label"} & set(features.columns)
    assert not leaked
