"""시뮬레이션 공유 컨텍스트.

정상 흐름 생성기와 패턴 주입기가 *동일한* 호가창/RNG/주문ID 발급기를 공유해
하나의 일관된 타임라인 위에서 동작하도록 한다. 덕분에 주입된 조작 주문도
주입 시점의 실제 호가창 상태를 기준으로 배치되어 미시구조가 일관된다.

라벨 부착도 여기서 일원화한다: 조작 주문을 제출할 때만 GroundTruthLabel을
order_id 기준으로 기록하고, 정상 주문은 라벨 테이블에 남기지 않는다(부재 = 정상).
이로써 라벨은 이벤트 스트림과 물리적으로 분리된 별도 테이블로만 존재한다.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .events import GroundTruthLabel, Label, OrderEvent, Side
from .orderbook import MatchingEngine


class StreamContext:
    def __init__(self, symbol: str, rng: np.random.Generator, tick_size: int = 1) -> None:
        self.symbol = symbol
        self.rng = rng
        self.tick_size = tick_size
        self.engine = MatchingEngine(symbol)
        self.events: List[OrderEvent] = []
        self.labels: List[GroundTruthLabel] = []
        self._next_oid = 0

    def new_order_id(self) -> int:
        oid = self._next_oid
        self._next_oid += 1
        return oid

    def submit(
        self,
        account_id: str,
        side: Side,
        price: int,
        quantity: int,
        ts: int,
        *,
        label: Label = Label.NORMAL,
        episode_id: Optional[str] = None,
        allow_self_trade: bool = True,
    ) -> int:
        """주문 제출. 발생 이벤트는 컨텍스트에 누적하고 order_id를 반환한다.

        label이 NORMAL이 아니면 해당 order_id를 조작으로 라벨링한다. 이 주문이
        이후 일으키는 모든 이벤트(NEW/CANCEL/체결)는 order_id로 조작에 귀속된다.
        allow_self_trade=False면 자기체결을 방지한다(정상 흐름용).
        """
        oid = self.new_order_id()
        self.events.extend(
            self.engine.submit(
                oid, account_id, side, price, quantity, ts,
                allow_self_trade=allow_self_trade,
            )
        )
        if label is not Label.NORMAL:
            self.labels.append(GroundTruthLabel(oid, label, episode_id))
        return oid

    def cancel(self, order_id: int, ts: int) -> None:
        """주문 취소. 라벨은 이미 order_id에 귀속돼 있으므로 별도 인자 불필요."""
        self.events.extend(self.engine.cancel(order_id, ts))
