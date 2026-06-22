"""Phase 3 분석 테스트 — 에피소드 테이블 · 탐지 분해 · 지연 · 오탐."""

from surveillance.detection.analysis import (
    build_episode_table,
    episode_detection,
    false_positive_profile,
    recall_by_intensity,
    recall_by_pattern,
)
from surveillance.detection.evaluate import build_truth
from surveillance.detection.ml import FEATURE_COLS
from surveillance.detection.rules import RuleConfig, predict
from surveillance.features.account_features import build_features
from surveillance.generator.dataset import DatasetConfig, generate_dataset
from surveillance.generator.injectors import InjectionConfig


def _setup(seed=42, duration=20_000, n=10):
    inj = InjectionConfig(randomize_intensity=True, camouflage=False,
                          spoof_qty_range=(10, 90), layering_levels_range=(2, 8),
                          wash_trades_range=(1, 7))
    r = generate_dataset(DatasetConfig(
        seed=seed, duration=duration,
        num_spoofing_episodes=n, num_wash_episodes=n, num_layering_episodes=n,
        injection=inj))
    features = build_features(r.events)
    truth = build_truth(r.events, r.labels)
    episodes = build_episode_table(r.events, r.labels)
    return r, features, truth, episodes


def test_episode_table_covers_all_episodes():
    _, _, _, episodes = _setup(n=10)
    assert len(episodes) == 30  # 패턴 3종 × 10
    assert set(episodes["pattern"]) == {"SPOOFING", "WASH_TRADING", "LAYERING"}
    assert (episodes["start_ts"] >= 0).all()
    assert (episodes["intensity"] > 0).all()


def test_episode_detection_latency_is_positive_when_detected():
    _, features, _, episodes = _setup()
    det = episode_detection(episodes, predict(features, RuleConfig()))
    hit = det[det["detected"]]
    assert len(hit) > 0
    assert (hit["latency_ms"] > 0).all()
    # 미탐은 지연이 없음
    assert det[~det["detected"]]["latency_ms"].isna().all()


def test_stronger_episodes_have_higher_recall():
    """고정 임계값의 핵심 한계: 강한 조작이 약한 조작보다 더 잘 잡힌다."""
    _, features, _, episodes = _setup()
    det = episode_detection(episodes, predict(features, RuleConfig()))
    rbi = recall_by_intensity(det)
    for pattern, grp in rbi.groupby("pattern"):
        weak = grp[grp["intensity_bucket"].str.startswith("약함")]["recall"]
        strong = grp[grp["intensity_bucket"].str.startswith("강함")]["recall"]
        if len(weak) and len(strong):
            assert strong.iloc[0] >= weak.iloc[0]


def test_loose_thresholds_induce_false_positives():
    """임계값을 낮추면 오탐이 생기고(정밀도 비용), 기본 임계값에선 거의 없다."""
    _, features, truth, _ = _setup()
    tight_fp, _ = false_positive_profile(predict(features, RuleConfig()), truth, FEATURE_COLS)
    loose = RuleConfig(cancel_ratio_threshold=0.3, qty_zscore_threshold=1.0,
                       distinct_levels_threshold=2, order_to_trade_threshold=1.2,
                       self_trade_threshold=1)
    loose_fp, profile = false_positive_profile(predict(features, loose), truth, FEATURE_COLS)

    tight_n = int(tight_fp["count"].sum()) if len(tight_fp) else 0
    loose_n = int(loose_fp["count"].sum()) if len(loose_fp) else 0
    assert loose_n > tight_n
    assert not profile.empty


def test_recall_by_pattern_shape():
    _, features, _, episodes = _setup()
    det = episode_detection(episodes, predict(features, RuleConfig()))
    rbp = recall_by_pattern(det)
    assert set(rbp.index) == {"SPOOFING", "WASH_TRADING", "LAYERING"}
    assert (rbp["recall"].between(0, 1)).all()
