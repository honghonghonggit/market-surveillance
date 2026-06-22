"""룰 기반 탐지 — (계좌, 윈도우) 피처에 임계값 룰을 적용.

룰 기반을 먼저 두는 이유: 해석 가능하고, 각 임계값이 어떤 시세조종 메커니즘을
겨냥하는지 명확하며, ML 분류기(Phase 2)의 성능 비교 기준선이 되기 때문이다.
임계값은 인자화되어 있어 정밀도-재현율 트레이드오프를 관찰할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..generator.events import Label


@dataclass
class RuleConfig:
    # 스푸핑: 대량 주문(시장 대비 큰 z-score)을 여러 건 깔았다가 대부분 취소
    cancel_ratio_threshold: float = 0.7
    qty_zscore_threshold: float = 3.0
    min_new_for_spoof: int = 3        # 단발성 우연 취소를 배제
    # 워시트레이딩: 자기체결이 *반복적으로* 발생(드문 우연 자기체결과 구분)
    self_trade_threshold: int = 3


def predict(features: pd.DataFrame, config: RuleConfig | None = None) -> pd.DataFrame:
    """피처 테이블에 predicted_label / is_alert 컬럼을 추가해 반환."""
    cfg = config or RuleConfig()
    out = features.copy()

    is_wash = out["self_trade_count"] >= cfg.self_trade_threshold
    is_spoof = (
        (out["cancel_ratio"] >= cfg.cancel_ratio_threshold)
        & (out["qty_zscore"] >= cfg.qty_zscore_threshold)
        & (out["num_new"] >= cfg.min_new_for_spoof)
    )

    # 워시 신호를 우선(자기체결은 더 확정적인 증거)
    predicted = pd.Series(Label.NORMAL.value, index=out.index, dtype=object)
    predicted[is_spoof] = Label.SPOOFING.value
    predicted[is_wash] = Label.WASH_TRADING.value

    out["predicted_label"] = predicted
    out["is_alert"] = predicted != Label.NORMAL.value
    return out
