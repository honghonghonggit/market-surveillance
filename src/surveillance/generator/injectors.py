"""조작 패턴 주입기 + ground truth 라벨.

각 에피소드는 제너레이터 코루틴으로 구현된다. `yield`에서 멈췄다가 드라이버가
`send(ts)`로 현재 틱을 넘겨주면 다음 행동을 수행한다. 이렇게 하면 정상 흐름과
같은 타임라인 위에서 한 틱씩 번갈아 진행되어, 주입 주문이 *주입 시점의 실제
호가창 상태*를 기준으로 배치된다(미시구조 일관성).

조작은 희소해야 한다(클래스 불균형 = 핵심 교훈). 주입 빈도/강도는 설정으로 조절한다.
모든 주입 주문은 order_id 기준으로 라벨링되므로, 그 주문이 일으키는 NEW/CANCEL/
체결 이벤트는 자동으로 해당 조작 에피소드에 귀속된다.

확장 포인트: 새 패턴(예: 레이어링)은 동일한 `(ctx, account, episode_id, cfg) ->
Generator` 시그니처의 에피소드 함수로 추가하면 된다(Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from .events import Label, Side
from .sim import StreamContext

# 에피소드 코루틴 타입: send(ts:int)로 구동, 값 반환 없음.
Episode = Generator[None, int, None]


@dataclass
class InjectionConfig:
    # 스푸핑: 대량 주문을 여러 개 깔았다가 체결 직전 전량 취소
    # 스푸핑: *소수 레벨*에 초대량 주문을 깔았다가 체결 직전 취소(큰 qty가 변별 신호)
    spoof_num_orders: int = 4
    spoof_qty: int = 80          # 정상 max_qty(~10) 대비 비정상적으로 큼
    spoof_hold_ticks: int = 4    # 깔고 나서 취소까지 유지하는 틱 수
    # 워시트레이딩: 동일 계좌 자기체결을 짧은 윈도우에 반복
    wash_num_trades: int = 6
    wash_qty: int = 15
    # 레이어링: *다수 레벨*에 중간 크기 주문을 층층이 분산(distinct level 수가 변별 신호)
    layering_num_levels: int = 7
    layering_qty: int = 18
    layering_hold_ticks: int = 4


def spoofing_episode(
    ctx: StreamContext, account: str, episode_id: str, cfg: InjectionConfig
) -> Episode:
    """초대량 매수 주문을 best bid의 *한두 레벨*에 집중해 매수 압력을 가장한 뒤,
    체결되기 전에 전량 취소한다. 큰 주문량(qty z-score)이 변별 신호다.
    → 해당 계좌 윈도우의 취소율과 주문량이 급등(소수 레벨에 집중)."""
    ts = yield  # priming 후 첫 send로 시작 ts 수신

    mid = ctx.engine.mid_price()
    anchor = ctx.engine.best_bid()
    if anchor is None:
        anchor = int(round(mid)) - 1 if mid is not None else 9_999

    oids = []
    for k in range(cfg.spoof_num_orders):
        price = max(anchor - (k % 2), 1)  # best bid·그 아래 한 틱(2레벨에 집중)
        oid = ctx.submit(
            account, Side.BUY, price, cfg.spoof_qty, ts,
            label=Label.SPOOFING, episode_id=episode_id,
        )
        oids.append(oid)

    for _ in range(cfg.spoof_hold_ticks):
        ts = yield

    for oid in oids:  # 체결 직전 전량 취소(스푸핑의 핵심)
        ctx.cancel(oid, ts)


def layering_episode(
    ctx: StreamContext, account: str, episode_id: str, cfg: InjectionConfig
) -> Episode:
    """여러 가격 레벨에 중간 크기 주문을 *층층이 분산* 배치해 호가창 깊이를 왜곡한 뒤
    전량 취소한다. 스푸핑과 달리 주문량은 그리 크지 않지만 *서로 다른 가격 레벨 수*가
    비정상적으로 많은 것이 변별 신호다. → distinct_price_levels + 취소율 급등."""
    ts = yield

    mid = ctx.engine.mid_price()
    anchor = ctx.engine.best_bid()
    if anchor is None:
        anchor = int(round(mid)) - 1 if mid is not None else 9_999

    oids = []
    for k in range(cfg.layering_num_levels):
        price = max(anchor - k, 1)  # 연속된 여러 레벨에 한 건씩 분산
        oid = ctx.submit(
            account, Side.BUY, price, cfg.layering_qty, ts,
            label=Label.LAYERING, episode_id=episode_id,
        )
        oids.append(oid)

    for _ in range(cfg.layering_hold_ticks):
        ts = yield

    for oid in oids:  # 목적 달성 후 전량 취소
        ctx.cancel(oid, ts)


def wash_trading_episode(
    ctx: StreamContext, account: str, episode_id: str, cfg: InjectionConfig
) -> Episode:
    """동일 계좌가 자기체결(self-trade)을 반복해 허위 거래량을 만든다.

    best ask보다 한 틱 낮은 가격(= 비어 있는 최저 매도 레벨)에 자기 매도를 깔고,
    같은 가격에 자기 매수를 넣어 *자기 자신과* 체결시킨다. 이렇게 하면 자기체결이
    결정적으로 보장된다(그 레벨엔 우리 매도만 존재). 스프레드가 1틱이라 한 틱 낮은
    가격이 매수호가를 침범하면 그 틱은 건너뛰고 다음 틱에 재시도한다."""
    ts = yield

    done = 0
    while done < cfg.wash_num_trades:
        ba = ctx.engine.best_ask()
        bb = ctx.engine.best_bid()
        mid = ctx.engine.mid_price()

        price = (ba - 1) if ba is not None else (int(round(mid)) if mid is not None else 10_000)
        if bb is not None and price <= bb:
            # 스프레드가 좁아 자기체결을 보장할 빈 레벨이 없음 → 다음 틱 재시도
            ts = yield
            continue

        # 비어 있는 최저 매도 레벨에 자기 매도 → 같은 가격 자기 매수로 자기체결
        ctx.submit(account, Side.SELL, price, cfg.wash_qty, ts,
                   label=Label.WASH_TRADING, episode_id=episode_id)
        ctx.submit(account, Side.BUY, price, cfg.wash_qty, ts,
                   label=Label.WASH_TRADING, episode_id=episode_id)
        done += 1
        ts = yield
