"""(계좌, 윈도우) 단위 행동 피처.

룰/ML 탐지의 입력이 되는 피처를 계산한다. 계좌 id는 *그룹 키*로만 쓰이며 피처가
아니다(어떤 계좌가 조작자인지 모델은 모른다 — 행동만으로 탐지한다).

핵심 피처
- cancel_ratio       : 윈도우 내 취소수/신규수 → 스푸핑(체결 직전 대량 취소) 신호
- order_to_trade_ratio: 신규수/체결수 → 주문은 많은데 체결은 적은 비정상 신호
- self_trade_count    : 자기체결 수 → 워시트레이딩의 직접 신호
- max_new_qty / qty_zscore: 시장 대비 비정상적으로 큰 주문 → 스푸핑 대량주문 신호
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..generator.events import EventType, OrderEvent
from .windows import DEFAULT_WINDOW_MS, window_index

Key = Tuple[str, int]  # (account_id, window)


def _blank() -> Dict[str, float]:
    return {
        "num_new": 0,
        "num_cancel": 0,
        "num_trade": 0,
        "self_trade_count": 0,
        "max_new_qty": 0,
    }


def build_features(
    events: Iterable[OrderEvent], window_ms: int = DEFAULT_WINDOW_MS
) -> pd.DataFrame:
    """OrderEvent 스트림을 (account_id, window) 피처 테이블로 집계."""
    acc: Dict[Key, Dict[str, float]] = defaultdict(_blank)
    window_new_qtys: Dict[int, List[int]] = defaultdict(list)

    for e in events:
        w = window_index(e.ts, window_ms)
        if e.event_type is EventType.NEW:
            row = acc[(e.account_id, w)]
            row["num_new"] += 1
            row["max_new_qty"] = max(row["max_new_qty"], e.quantity)
            window_new_qtys[w].append(e.quantity)
        elif e.event_type is EventType.CANCEL:
            acc[(e.account_id, w)]["num_cancel"] += 1
        elif e.event_type is EventType.TRADE:
            # 체결은 maker/taker 양쪽 계좌에 귀속(둘 다 그 윈도우에 '활동'했다)
            for participant in {e.maker_account, e.taker_account}:
                if participant is not None:
                    acc[(participant, w)]["num_trade"] += 1
            if e.is_self_trade and e.maker_account is not None:
                acc[(e.maker_account, w)]["self_trade_count"] += 1

    # 윈도우별 시장 통계(주문량 z-score 기준)
    win_mean: Dict[int, float] = {}
    win_std: Dict[int, float] = {}
    for w, qtys in window_new_qtys.items():
        arr = np.asarray(qtys, dtype=float)
        win_mean[w] = float(arr.mean())
        win_std[w] = float(arr.std())  # population std

    records = []
    for (account_id, w), row in acc.items():
        num_new = row["num_new"]
        num_trade = row["num_trade"]
        std = win_std.get(w, 0.0)
        qty_zscore = (row["max_new_qty"] - win_mean.get(w, 0.0)) / std if std > 0 else 0.0
        records.append(
            {
                "account_id": account_id,
                "window": w,
                "num_new": num_new,
                "num_cancel": row["num_cancel"],
                "num_trade": num_trade,
                "self_trade_count": row["self_trade_count"],
                "max_new_qty": row["max_new_qty"],
                "cancel_ratio": row["num_cancel"] / num_new if num_new > 0 else 0.0,
                "order_to_trade_ratio": num_new / num_trade if num_trade > 0 else float(num_new),
                "qty_zscore": qty_zscore,
            }
        )

    cols = [
        "account_id", "window", "num_new", "num_cancel", "num_trade",
        "self_trade_count", "max_new_qty", "cancel_ratio",
        "order_to_trade_ratio", "qty_zscore",
    ]
    return pd.DataFrame.from_records(records, columns=cols).sort_values(
        ["window", "account_id"]
    ).reset_index(drop=True)
