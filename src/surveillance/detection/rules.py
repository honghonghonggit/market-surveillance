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
    cancel_ratio_threshold: float = 0.7   # 스푸핑·레이어링 공통: 깔았다가 대부분 취소
    # 스푸핑: 소수 레벨에 *초대량* 주문 → 시장 대비 큰 주문량 z-score
    qty_zscore_threshold: float = 5.0
    min_new_for_spoof: int = 3            # 단발성 우연 취소 배제
    # 레이어링: *서로 다른 가격 레벨*에 다수 주문을 분산하되 거의 체결되지 않음
    distinct_levels_threshold: int = 5
    order_to_trade_threshold: float = 3.0  # 주문 대비 체결이 적음(허위 호가의 특징)
    # 워시트레이딩: 자기체결이 *반복적으로* 발생(드문 우연 자기체결과 구분)
    self_trade_threshold: int = 3


def predict(features: pd.DataFrame, config: RuleConfig | None = None) -> pd.DataFrame:
    """피처 테이블에 predicted_label / is_alert 컬럼을 추가해 반환.

    우선순위: 워시(자기체결, 가장 확정적) → 레이어링(다수 레벨) → 스푸핑(초대량).
    레이어링과 스푸핑은 둘 다 취소율이 높지만, '서로 다른 레벨 수'와 '주문량'으로
    구분한다(레이어링=넓게 분산, 스푸핑=소수 레벨에 집중).
    """
    cfg = config or RuleConfig()
    out = features.copy()
    high_cancel = out["cancel_ratio"] >= cfg.cancel_ratio_threshold

    is_wash = out["self_trade_count"] >= cfg.self_trade_threshold
    is_layering = (
        high_cancel
        & (out["distinct_price_levels"] >= cfg.distinct_levels_threshold)
        & (out["order_to_trade_ratio"] >= cfg.order_to_trade_threshold)
    )
    is_spoof = (
        high_cancel
        & (out["qty_zscore"] >= cfg.qty_zscore_threshold)
        & (out["num_new"] >= cfg.min_new_for_spoof)
    )

    predicted = pd.Series(Label.NORMAL.value, index=out.index, dtype=object)
    predicted[is_spoof] = Label.SPOOFING.value
    predicted[is_layering] = Label.LAYERING.value  # 레이어링이 스푸핑보다 우선
    predicted[is_wash] = Label.WASH_TRADING.value  # 워시가 최우선

    out["predicted_label"] = predicted
    out["is_alert"] = predicted != Label.NORMAL.value
    return out
