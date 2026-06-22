"""주문 생애주기 이벤트 데이터 모델.

탐지 대상 패턴(스푸핑·워시트레이딩)은 모두 *주문의 행동*으로 정의되므로,
시스템은 스냅샷이 아니라 주문 생애주기 이벤트 스트림(NEW/CANCEL/TRADE)을 다룬다.

설계상 가장 중요한 점: **ground truth 라벨을 OrderEvent에 넣지 않는다.**
라벨/에피소드 정보는 별도의 GroundTruthLabel 테이블로 물리적으로 분리되어,
피처 엔지니어링은 OrderEvent 스트림만 입력으로 받는다. 따라서 라벨 누설(leakage)이
타입 수준에서 구조적으로 불가능하다. (RA-Testbed의 lookahead 방지에 대응하는 장치)

가격은 정수 틱(tick) 단위로 표현해 부동소수 오차 없이 완전한 재현성을 보장한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """주문 생애주기 이벤트 종류."""

    NEW = "NEW"        # 신규 주문이 호가창에 등록됨(미체결 잔량)
    CANCEL = "CANCEL"  # 미체결 주문이 취소됨
    TRADE = "TRADE"    # 두 주문이 체결됨


class Side(str, Enum):
    """주문 방향."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class Label(str, Enum):
    """Ground truth 라벨. NORMAL을 제외한 값은 주입된 조작 패턴이다."""

    NORMAL = "NORMAL"
    SPOOFING = "SPOOFING"
    WASH_TRADING = "WASH_TRADING"
    LAYERING = "LAYERING"


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """주문 생애주기 이벤트 1건. (라벨 정보는 의도적으로 포함하지 않는다.)

    가격(`price`)은 정수 틱 단위. NEW/CANCEL은 단일 주문에 대한 이벤트이고,
    TRADE는 maker/taker 두 주문의 체결을 나타낸다.
    """

    seq: int                 # 단조 증가 시퀀스 번호(결정적 정렬)
    ts: int                  # 논리 타임스탬프(ms). 윈도우 집계의 기준
    event_type: EventType
    order_id: int
    account_id: str
    symbol: str
    side: Side
    price: int               # 정수 틱
    quantity: int

    # TRADE 이벤트에서만 채워진다. 자기체결(워시) 탐지에 maker/taker 계좌가 핵심.
    maker_order_id: Optional[int] = None
    taker_order_id: Optional[int] = None
    maker_account: Optional[str] = None
    taker_account: Optional[str] = None

    @property
    def is_self_trade(self) -> bool:
        """동일 계좌 간 체결(워시트레이딩의 기계적 신호)."""
        return (
            self.event_type is EventType.TRADE
            and self.maker_account is not None
            and self.maker_account == self.taker_account
        )


@dataclass(frozen=True, slots=True)
class GroundTruthLabel:
    """정답 라벨 1건. OrderEvent와 물리적으로 분리된 테이블로 보관한다.

    order_id 기준으로 어떤 주문이 어떤 조작 에피소드에 속하는지를 기록한다.
    NORMAL 주문은 라벨 테이블에 굳이 담지 않아도 되지만(부재 = 정상),
    평가 단계에서 명시적으로 다루기 위해 episode_id는 조작 주문에만 부여한다.
    """

    order_id: int
    label: Label
    episode_id: Optional[str] = None  # 조작 에피소드 식별자. NORMAL이면 None
