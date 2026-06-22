"""경량 매칭엔진 — 가격-시간 우선순위(price-time priority) 호가창.

심볼 1개에 대해 실제 호가창 상태를 유지하고 교차 주문을 체결한다.
호가창이 진짜로 유지되므로 취소율·주문체결비율·가격충격·자기체결 같은 피처가
실제 시장 미시구조에서 도출된다(확률적 근사가 아님).

모든 상태 변화는 OrderEvent로 방출되며, 시퀀스 번호(seq)는 엔진이 소유해
전체 이벤트 스트림에서 단조 증가를 보장한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque, List, Optional

from .events import EventType, OrderEvent, Side


@dataclass(slots=True)
class _Resting:
    """호가창에 등록된 미체결 주문(가변)."""

    order_id: int
    account_id: str
    side: Side
    price: int
    quantity: int  # 잔량
    ts: int


class MatchingEngine:
    """단일 심볼 호가창 + 연속체결 매칭.

    가격-시간 우선순위: 같은 가격 레벨 안에서는 먼저 도착한 주문이 먼저 체결된다.
    체결 가격은 호가창에 먼저 있던 maker의 가격이다.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        # price(tick) -> 도착순 FIFO 큐
        self._bids: Dict[int, Deque[_Resting]] = {}
        self._asks: Dict[int, Deque[_Resting]] = {}
        self._by_id: Dict[int, _Resting] = {}
        self._seq: int = 0

    # ── 조회 헬퍼 ────────────────────────────────────────────────
    def best_bid(self) -> Optional[int]:
        return max(self._bids) if self._bids else None

    def best_ask(self) -> Optional[int]:
        return min(self._asks) if self._asks else None

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2

    def _next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    # ── 주문 제출 ────────────────────────────────────────────────
    def submit(
        self,
        order_id: int,
        account_id: str,
        side: Side,
        price: int,
        quantity: int,
        ts: int,
        allow_self_trade: bool = True,
    ) -> List[OrderEvent]:
        """주문 제출. 교차분은 체결(TRADE), 잔량은 호가창에 등록(NEW).

        allow_self_trade=False면 자기 자신의 미체결 주문은 건너뛰고 다음 maker와
        체결한다(거래소의 self-trade prevention). 정상 흐름은 이 옵션으로 우연한
        자기체결을 원천 차단하고, 워시트레이딩은 기본값(True)으로 자기체결을 만든다.
        """
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        events: List[OrderEvent] = []
        remaining = quantity
        book = self._asks if side is Side.BUY else self._bids

        def crosses(level_price: int) -> bool:
            return level_price <= price if side is Side.BUY else level_price >= price

        # 체결 가능한 가격 레벨을 best부터 순회한다. 단일 submit 동안 호가는 추가되지
        # 않고 제거만 되므로, 가격 레벨 스냅샷을 정렬해 순회해도 안전하다.
        levels = sorted(book) if side is Side.BUY else sorted(book, reverse=True)
        for lvl in levels:
            if remaining == 0 or not crosses(lvl):
                break
            queue = book[lvl]
            i = 0
            while remaining > 0 and i < len(queue):
                maker = queue[i]
                if not allow_self_trade and maker.account_id == account_id:
                    i += 1  # STP: 자기 주문은 건너뛰고(체결 안 함) 그대로 둔다
                    continue
                traded = min(remaining, maker.quantity)
                remaining -= traded
                maker.quantity -= traded
                events.append(
                    OrderEvent(
                        seq=self._next_seq(),
                        ts=ts,
                        event_type=EventType.TRADE,
                        order_id=order_id,
                        account_id=account_id,
                        symbol=self.symbol,
                        side=side,
                        price=lvl,            # maker 가격에 체결
                        quantity=traded,
                        maker_order_id=maker.order_id,
                        taker_order_id=order_id,
                        maker_account=maker.account_id,
                        taker_account=account_id,
                    )
                )
                if maker.quantity == 0:
                    del queue[i]  # 체결 완료된 maker 제거(뒤 주문이 i로 당겨짐)
                    self._by_id.pop(maker.order_id, None)
                # 부분체결이면 remaining==0이라 루프 종료
            if not queue:
                del book[lvl]

        # 잔량은 호가창에 등록.
        if remaining > 0:
            resting = _Resting(order_id, account_id, side, price, remaining, ts)
            own = self._bids if side is Side.BUY else self._asks
            own.setdefault(price, deque()).append(resting)
            self._by_id[order_id] = resting
            events.append(
                OrderEvent(
                    seq=self._next_seq(),
                    ts=ts,
                    event_type=EventType.NEW,
                    order_id=order_id,
                    account_id=account_id,
                    symbol=self.symbol,
                    side=side,
                    price=price,
                    quantity=remaining,
                )
            )
        return events

    # ── 주문 취소 ────────────────────────────────────────────────
    def cancel(self, order_id: int, ts: int) -> List[OrderEvent]:
        """미체결 주문 취소. 이미 사라진 주문이면 빈 리스트."""
        resting = self._by_id.pop(order_id, None)
        if resting is None:
            return []

        book = self._bids if resting.side is Side.BUY else self._asks
        queue = book.get(resting.price)
        if queue is not None:
            try:
                queue.remove(resting)
            except ValueError:
                pass
            if not queue:
                del book[resting.price]

        return [
            OrderEvent(
                seq=self._next_seq(),
                ts=ts,
                event_type=EventType.CANCEL,
                order_id=order_id,
                account_id=resting.account_id,
                symbol=self.symbol,
                side=resting.side,
                price=resting.price,
                quantity=resting.quantity,
            )
        ]

    def is_resting(self, order_id: int) -> bool:
        """주문이 아직 호가창에 미체결로 남아 있는지."""
        return order_id in self._by_id

    def best_resting_account(self, side: Side) -> Optional[str]:
        """해당 방향 best 레벨의 시간 우선(맨 앞) 주문 계좌. 비어 있으면 None.

        정상 흐름이 자기 자신과 체결(우연한 자기체결)하는 것을 피하도록(거래소의
        self-trade prevention에 대응) 사용한다. 워시트레이딩은 의도적으로 자기체결을
        만들므로 이 가드를 쓰지 않는다.
        """
        book = self._bids if side is Side.BUY else self._asks
        if not book:
            return None
        best = max(book) if side is Side.BUY else min(book)
        return book[best][0].account_id
