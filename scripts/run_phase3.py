"""Phase 3 — 패턴별 탐지 정확도 분해 · 탐지 지연 · 오탐 케이스 분석.

정답을 알고 있다는 강점을 끝까지 활용해 "어디서 얼마나 놓치고/늦고/헛짚는가"를 정량화한다.
실행:  python scripts/run_phase3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

from surveillance.detection.analysis import (  # noqa: E402
    build_episode_table,
    episode_detection,
    false_positive_profile,
    recall_by_intensity,
    recall_by_pattern,
)
from surveillance.detection.evaluate import build_truth  # noqa: E402
from surveillance.detection.ml import FEATURE_COLS  # noqa: E402
from surveillance.detection.rules import RuleConfig, predict  # noqa: E402
from surveillance.features.account_features import build_features  # noqa: E402
from surveillance.generator.dataset import DatasetConfig, generate_dataset  # noqa: E402
from surveillance.generator.injectors import InjectionConfig  # noqa: E402


def main() -> None:
    # 강도만 랜덤화(위장 OFF): 강도가 임계값을 가로질러 분포하므로 "약한 조작은 놓치고
    # 강한 조작은 잡는" 고정 임계값의 한계가 패턴 × 강도 분해로 깨끗이 드러난다.
    inj = InjectionConfig(
        randomize_intensity=True, camouflage=False,
        spoof_qty_range=(10, 90), layering_levels_range=(2, 8), wash_trades_range=(1, 7),
    )
    cfg = DatasetConfig(
        seed=42, duration=40_000,
        num_spoofing_episodes=25, num_wash_episodes=25, num_layering_episodes=25,
        injection=inj,
    )
    result = generate_dataset(cfg)
    features = build_features(result.events)
    truth = build_truth(result.events, result.labels)
    episodes = build_episode_table(result.events, result.labels)

    # 기본 룰로 탐지 → 에피소드별 탐지/지연
    preds = predict(features, RuleConfig())
    det = episode_detection(episodes, preds)

    print("=" * 60)
    print("Phase 3 — 탐지 정확도 분해 / 지연 / 오탐 분석 (위장 모드)")
    print("=" * 60)
    print(f"주입 에피소드 {len(episodes)}개  |  전체 탐지율(recall) "
          f"{det['detected'].mean():.3f}\n")

    print("[1] 패턴별 탐지 정확도")
    print(recall_by_pattern(det).to_string())
    print()

    print("[2] 강도(약/강)별 탐지율 — 약한 에피소드를 더 놓친다(고정 임계값의 한계)")
    print(recall_by_intensity(det).to_string(index=False))
    print()

    # [3] 탐지 지연
    hit = det[det["detected"]]
    print("[3] 탐지 지연(latency, ms) — 패턴 발생 → 최초 탐지")
    if len(hit):
        print(f"  중앙값 {hit['latency_ms'].median():.0f}  |  평균 "
              f"{hit['latency_ms'].mean():.0f}  |  최대 {hit['latency_ms'].max():.0f}")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for pat, g in hit.groupby("pattern"):
            ax.hist(g["latency_ms"], bins=15, alpha=0.6, label=pat)
        ax.set_xlabel("탐지 지연 (ms)"); ax.set_ylabel("에피소드 수")
        ax.set_title("패턴 발생 → 탐지까지 지연 분포")
        ax.legend()
        Path("artifacts").mkdir(exist_ok=True)
        fig.tight_layout(); fig.savefig("artifacts/detection_latency.png", dpi=120)
        plt.close(fig)
        print("  히스토그램: artifacts/detection_latency.png")
    print()

    # [4] 오탐 케이스 분석 — 임계값을 낮추면 무엇을 헛짚는가(정밀도 비용)
    print("[4] 오탐(FP) 케이스 분석 — 임계값을 공격적으로 낮춘 경우")
    loose = RuleConfig(cancel_ratio_threshold=0.3, qty_zscore_threshold=1.0,
                       distinct_levels_threshold=2, order_to_trade_threshold=1.2,
                       self_trade_threshold=1)
    loose_preds = predict(features, loose)
    by_rule, profile = false_positive_profile(loose_preds, truth, FEATURE_COLS)
    n_fp = int(by_rule["count"].sum()) if len(by_rule) else 0
    print(f"  유도된 오탐 {n_fp}건  (발화 룰별)")
    if n_fp:
        print(by_rule.to_string())
        print("  오탐 평균 피처 프로파일:")
        print(profile.round(3).to_string())
        print("  → 느슨한 임계값은 활발한 정상 계좌(취소·다레벨 주문)를 조작으로 오인한다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
