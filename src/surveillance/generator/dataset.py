"""데이터셋 오케스트레이션 — 정상 흐름 + 패턴 주입을 하나의 타임라인으로 병합.

매 틱 정상 흐름을 한 스텝 진행시키고, 예약된 조작 에피소드가 있으면 활성화해
같은 틱부터 코루틴으로 구동한다. 이벤트는 발생 순서(=seq 순서=(ts,seq) 순서)대로
누적되므로 별도 정렬이 필요 없다.

반환값은 (이벤트 스트림, ground truth 라벨 테이블)이며 두 가지는 물리적으로 분리된다.
시드가 같으면 전체 데이터셋이 완전히 재현된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List

import numpy as np

from .events import GroundTruthLabel, OrderEvent
from .injectors import (
    Episode,
    InjectionConfig,
    layering_episode,
    spoofing_episode,
    wash_trading_episode,
)
from .normal_flow import NormalFlow, NormalFlowConfig
from .sim import StreamContext


@dataclass
class DatasetConfig:
    symbol: str = "ACME"
    seed: int = 42
    duration: int = 20_000        # 틱 수(= ms)
    tick_size: int = 1
    warmup_ticks: int = 500       # 호가창이 형성될 시간(주입 전)
    tail_margin: int = 200        # 끝부분 여유(에피소드가 잘리지 않도록)
    num_spoofing_episodes: int = 8
    num_wash_episodes: int = 8
    num_layering_episodes: int = 8
    normal: NormalFlowConfig = field(default_factory=NormalFlowConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)


@dataclass
class DatasetResult:
    events: List[OrderEvent]
    labels: List[GroundTruthLabel]
    config: DatasetConfig


def generate_dataset(config: DatasetConfig | None = None) -> DatasetResult:
    cfg = config or DatasetConfig()
    rng = np.random.default_rng(cfg.seed)
    ctx = StreamContext(cfg.symbol, rng, cfg.tick_size)
    normal = NormalFlow(ctx, cfg.normal)

    # 에피소드 예약: 시작 틱 -> 코루틴 팩토리 목록
    scheduled: Dict[int, List[Callable[[], Episode]]] = {}
    lo, hi = cfg.warmup_ticks, cfg.duration - cfg.tail_margin

    def schedule(make_episode, account: str, episode_id: str) -> None:
        start = int(rng.integers(lo, hi))
        factory = partial(make_episode, ctx, account, episode_id, cfg.injection)
        scheduled.setdefault(start, []).append(factory)

    for i in range(cfg.num_spoofing_episodes):
        schedule(spoofing_episode, f"M_SPOOF_{i:02d}", f"spoof_{i:02d}")
    for i in range(cfg.num_wash_episodes):
        schedule(wash_trading_episode, f"M_WASH_{i:02d}", f"wash_{i:02d}")
    for i in range(cfg.num_layering_episodes):
        schedule(layering_episode, f"M_LAYER_{i:02d}", f"layer_{i:02d}")

    active: List[Episode] = []
    for ts in range(cfg.duration):
        normal.step(ts)

        for factory in scheduled.get(ts, []):
            gen = factory()
            next(gen)  # priming(첫 `ts = yield`까지 진행)
            active.append(gen)

        still: List[Episode] = []
        for gen in active:
            try:
                gen.send(ts)
            except StopIteration:
                pass
            else:
                still.append(gen)
        active = still

    return DatasetResult(events=ctx.events, labels=ctx.labels, config=cfg)
