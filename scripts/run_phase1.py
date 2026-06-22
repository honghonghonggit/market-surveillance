"""Phase 1 인메모리 파이프라인 엔드투엔드 실행.

생성(패턴 주입 + ground truth) → (인메모리 스트림 순회) → 피처 → 룰 탐지 → 평가.
Kafka·Streamlit은 Phase 2. 동일 시드면 결과가 완전히 재현된다.

실행:  python scripts/run_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surveillance.detection.evaluate import build_truth, evaluate  # noqa: E402
from surveillance.detection.rules import RuleConfig, predict  # noqa: E402
from surveillance.features.account_features import build_features  # noqa: E402
from surveillance.features.windows import DEFAULT_WINDOW_MS  # noqa: E402
from surveillance.generator.dataset import DatasetConfig, generate_dataset  # noqa: E402
from surveillance.generator.events import EventType  # noqa: E402


def main() -> None:
    cfg = DatasetConfig(seed=42, duration=30_000)
    result = generate_dataset(cfg)

    n_new = sum(1 for e in result.events if e.event_type is EventType.NEW)
    n_trade = sum(1 for e in result.events if e.event_type is EventType.TRADE)
    n_cancel = sum(1 for e in result.events if e.event_type is EventType.CANCEL)
    print(
        f"생성 이벤트: {len(result.events):,}  "
        f"(NEW={n_new:,}, TRADE={n_trade:,}, CANCEL={n_cancel:,})  "
        f"| 조작 주문 라벨: {len(result.labels)}"
    )

    # 피처는 라벨이 없는 이벤트 스트림만 입력으로 받는다(누설 차단).
    features = build_features(result.events, window_ms=DEFAULT_WINDOW_MS)
    predictions = predict(features, RuleConfig())
    truth = build_truth(result.events, result.labels, window_ms=DEFAULT_WINDOW_MS)
    report = evaluate(predictions, truth)

    print()
    print(report.format())


if __name__ == "__main__":
    main()
