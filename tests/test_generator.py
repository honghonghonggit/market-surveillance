"""합성 데이터 생성기 + 패턴 주입 테스트.

이 프로젝트의 핵심(ground truth 정합성·재현성·라벨 누설 분리)을 가장 공들여 검증한다.
"""

import dataclasses

import numpy as np
import pytest

from surveillance.generator.dataset import DatasetConfig, generate_dataset
from surveillance.generator.events import EventType, Label, OrderEvent
from surveillance.generator.normal_flow import NormalFlow, NormalFlowConfig
from surveillance.generator.sim import StreamContext


# ── 정상 흐름 ────────────────────────────────────────────────────
def _run_normal(seed: int, ticks: int = 2000):
    rng = np.random.default_rng(seed)
    ctx = StreamContext("ACME", rng)
    flow = NormalFlow(ctx, NormalFlowConfig())
    for ts in range(ticks):
        flow.step(ts)
    return ctx


def test_normal_flow_is_reproducible():
    a = _run_normal(7)
    b = _run_normal(7)
    assert [dataclasses.astuple(e) for e in a.events] == [
        dataclasses.astuple(e) for e in b.events
    ]


def test_normal_flow_produces_trades_and_baseline_cancels():
    ctx = _run_normal(7)
    types = [e.event_type for e in ctx.events]
    assert types.count(EventType.TRADE) > 0
    assert types.count(EventType.CANCEL) > 0
    # 기저 취소율은 존재하되 스푸핑(≈1.0)과 충분히 대비되어야 한다
    cancels = types.count(EventType.CANCEL)
    news = types.count(EventType.NEW)
    assert 0 < cancels / news < 0.6


def test_normal_flow_has_no_ground_truth_labels():
    ctx = _run_normal(7)
    assert ctx.labels == []  # 정상 주문은 라벨 테이블에 남지 않는다(부재 = 정상)


# ── 전체 데이터셋 / 주입 ──────────────────────────────────────────
@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(DatasetConfig(seed=42, duration=8000))


def test_dataset_is_reproducible():
    a = generate_dataset(DatasetConfig(seed=1, duration=4000))
    b = generate_dataset(DatasetConfig(seed=1, duration=4000))
    assert [dataclasses.astuple(e) for e in a.events] == [
        dataclasses.astuple(e) for e in b.events
    ]
    assert [dataclasses.astuple(l) for l in a.labels] == [
        dataclasses.astuple(l) for l in b.labels
    ]


def test_both_patterns_are_injected(dataset):
    injected = {l.label for l in dataset.labels}
    assert Label.SPOOFING in injected
    assert Label.WASH_TRADING in injected


def test_manipulation_is_rare(dataset):
    """조작 주문은 전체의 극소수여야 한다(불균형 = 핵심 교훈)."""
    manip_ids = {l.order_id for l in dataset.labels}
    new_order_ids = {e.order_id for e in dataset.events if e.event_type is EventType.NEW}
    ratio = len(manip_ids) / len(new_order_ids)
    assert ratio < 0.10


def test_ground_truth_labels_are_physically_separate(dataset):
    """라벨이 OrderEvent에 절대 섞이지 않음을 구조적으로 보장."""
    event_fields = {f.name for f in dataclasses.fields(OrderEvent)}
    assert "label" not in event_fields
    assert "episode_id" not in event_fields
    # 라벨은 별도 테이블에만 존재
    assert all(l.episode_id is not None for l in dataset.labels)


def test_spoofing_orders_are_large_and_cancelled(dataset):
    spoof_ids = {l.order_id for l in dataset.labels if l.label is Label.SPOOFING}
    spoof_events = [e for e in dataset.events if e.order_id in spoof_ids]
    # 스푸핑 주문은 비정상적으로 큼
    new_qtys = [e.quantity for e in spoof_events if e.event_type is EventType.NEW]
    assert new_qtys and min(new_qtys) >= 50
    # 깔린 스푸핑 주문은 취소된다
    cancelled = {e.order_id for e in spoof_events if e.event_type is EventType.CANCEL}
    assert cancelled, "스푸핑 주문이 취소 이벤트를 남겨야 한다"


def test_wash_trading_creates_self_trades(dataset):
    wash_ids = {l.order_id for l in dataset.labels if l.label is Label.WASH_TRADING}
    self_trades = [
        e for e in dataset.events
        if e.event_type is EventType.TRADE and e.is_self_trade
        and (e.maker_order_id in wash_ids or e.taker_order_id in wash_ids)
    ]
    assert len(self_trades) > 0, "워시트레이딩이 자기체결을 만들어야 한다"
