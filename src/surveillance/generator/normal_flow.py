"""정상 주문 흐름 생성기.

공정가치(fair value)가 랜덤워크를 따라 움직이고, 그 주변에 다수 계좌가
한정주문을 낸다. 일부는 호가를 가로질러(aggressive) 즉시 체결을 유발하고,
이미 호가창에 쌓인 주문 중 일부는 매 틱 낮은 확률로 취소되어 *기저 취소율*을
형성한다. 이 기저 취소율이 있어야 스푸핑의 비정상적으로 높은 취소율이 대비된다.

단일 RNG로 구동되므로 시드가 같으면 흐름이 완전히 재현된다(평가 엄밀성의 전제).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .events import Side
from .sim import StreamContext


@dataclass
class NormalFlowConfig:
    num_accounts: int = 12
    initial_price: int = 10_000      # 정수 틱
    volatility: float = 1.5          # 틱당 공정가치 랜덤워크 표준편차
    arrival_rate: float = 0.8        # 틱당 신규 주문 수(Poisson λ)
    aggressive_prob: float = 0.35    # 신규 주문이 호가를 가로지를(체결 유발) 확률
    max_spread: int = 5             # 패시브 주문이 미드에서 떨어지는 최대 틱
    min_qty: int = 1
    max_qty: int = 10
    cancel_prob: float = 0.02        # 호가창의 각 미체결 주문이 매 틱 취소될 확률


@dataclass
class NormalFlow:
    """매 틱 정상 주문 흐름을 한 스텝 진행시키는 stateful 스테퍼."""

    ctx: StreamContext
    config: NormalFlowConfig = field(default_factory=NormalFlowConfig)

    def __post_init__(self) -> None:
        self.fair_value: float = float(self.config.initial_price)
        self.accounts: List[str] = [f"ACC_{i:03d}" for i in range(self.config.num_accounts)]
        self._resting: List[int] = []  # 정상 흐름이 낸 미체결 주문 id 추적

    def step(self, ts: int) -> None:
        rng = self.ctx.rng
        cfg = self.config

        # 1) 공정가치 랜덤워크
        self.fair_value += rng.normal(0.0, cfg.volatility)
        mid = int(round(self.fair_value))

        # 2) 기저 취소: 미체결 주문 중 일부를 확률적으로 취소
        survivors: List[int] = []
        for oid in self._resting:
            if not self.ctx.engine.is_resting(oid):
                continue  # 이미 체결되어 사라짐
            if rng.random() < cfg.cancel_prob:
                self.ctx.cancel(oid, ts)
            else:
                survivors.append(oid)
        self._resting = survivors

        # 3) 신규 주문 도착(Poisson)
        for _ in range(rng.poisson(cfg.arrival_rate)):
            account = self.accounts[rng.integers(len(self.accounts))]
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            qty = int(rng.integers(cfg.min_qty, cfg.max_qty + 1))

            # self-trade prevention: 반대편 best가 자기 주문이면 공격적 주문을 피한다
            # (정상 참여자는 자기 자신과 체결하지 않는다 → 자기체결은 워시 신호로 깨끗이 남음)
            opp_account = self.ctx.engine.best_resting_account(side.opposite)
            aggressive = rng.random() < cfg.aggressive_prob and opp_account != account

            if aggressive:
                # 공격적: 반대편 best 호가만 가로질러 체결(상단만 소진).
                if side is Side.BUY:
                    ba = self.ctx.engine.best_ask()
                    price = ba if ba is not None else mid + 1
                else:
                    bb = self.ctx.engine.best_bid()
                    price = bb if bb is not None else mid - 1
            else:
                # 패시브: 미드 안쪽에 호가 제시
                offset = int(rng.integers(1, cfg.max_spread + 1))
                price = mid - offset if side is Side.BUY else mid + offset

            oid = self.ctx.submit(
                account, side, max(price, 1), qty, ts, allow_self_trade=False
            )
            if self.ctx.engine.is_resting(oid):
                self._resting.append(oid)
