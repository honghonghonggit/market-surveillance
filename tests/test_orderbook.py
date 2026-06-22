"""경량 매칭엔진 단위 테스트."""

from surveillance.generator.events import EventType, Side
from surveillance.generator.orderbook import MatchingEngine


def test_resting_order_emits_new_and_sits_in_book():
    eng = MatchingEngine("ACME")
    events = eng.submit(1, "A", Side.BUY, price=100, quantity=10, ts=0)

    assert len(events) == 1
    assert events[0].event_type is EventType.NEW
    assert eng.best_bid() == 100
    assert eng.is_resting(1)


def test_crossing_order_fully_fills_maker():
    eng = MatchingEngine("ACME")
    eng.submit(1, "A", Side.SELL, price=100, quantity=10, ts=0)
    events = eng.submit(2, "B", Side.BUY, price=100, quantity=10, ts=1)

    trades = [e for e in events if e.event_type is EventType.TRADE]
    assert len(trades) == 1
    t = trades[0]
    assert t.price == 100 and t.quantity == 10
    assert t.maker_order_id == 1 and t.taker_order_id == 2
    assert t.maker_account == "A" and t.taker_account == "B"
    # 양쪽 모두 완전 체결 → 호가창 비어야 함, NEW 없음
    assert not eng.is_resting(1)
    assert eng.best_bid() is None and eng.best_ask() is None
    assert all(e.event_type is EventType.TRADE for e in events)


def test_partial_fill_rests_remainder():
    eng = MatchingEngine("ACME")
    eng.submit(1, "A", Side.SELL, price=100, quantity=4, ts=0)
    events = eng.submit(2, "B", Side.BUY, price=100, quantity=10, ts=1)

    trades = [e for e in events if e.event_type is EventType.TRADE]
    news = [e for e in events if e.event_type is EventType.NEW]
    assert sum(t.quantity for t in trades) == 4
    assert len(news) == 1 and news[0].quantity == 6  # 잔량 6 등록
    assert eng.best_bid() == 100


def test_price_time_priority_fifo():
    eng = MatchingEngine("ACME")
    eng.submit(1, "A", Side.BUY, price=100, quantity=5, ts=0)  # 먼저 도착
    eng.submit(2, "B", Side.BUY, price=100, quantity=5, ts=1)  # 나중 도착
    events = eng.submit(3, "C", Side.SELL, price=100, quantity=5, ts=2)

    trades = [e for e in events if e.event_type is EventType.TRADE]
    assert len(trades) == 1
    assert trades[0].maker_order_id == 1  # FIFO: 먼저 온 주문이 먼저 체결


def test_better_price_level_fills_first():
    eng = MatchingEngine("ACME")
    eng.submit(1, "A", Side.SELL, price=101, quantity=5, ts=0)
    eng.submit(2, "B", Side.SELL, price=100, quantity=5, ts=1)  # 더 좋은(낮은) 매도
    events = eng.submit(3, "C", Side.BUY, price=101, quantity=5, ts=2)

    trades = [e for e in events if e.event_type is EventType.TRADE]
    assert len(trades) == 1
    assert trades[0].maker_order_id == 2 and trades[0].price == 100


def test_cancel_removes_resting_order():
    eng = MatchingEngine("ACME")
    eng.submit(1, "A", Side.BUY, price=100, quantity=10, ts=0)
    events = eng.cancel(1, ts=5)

    assert len(events) == 1
    assert events[0].event_type is EventType.CANCEL
    assert events[0].quantity == 10
    assert not eng.is_resting(1)
    assert eng.best_bid() is None


def test_cancel_unknown_order_is_noop():
    eng = MatchingEngine("ACME")
    assert eng.cancel(999, ts=5) == []


def test_self_trade_flag():
    eng = MatchingEngine("ACME")
    eng.submit(1, "WASH", Side.SELL, price=100, quantity=5, ts=0)
    events = eng.submit(2, "WASH", Side.BUY, price=100, quantity=5, ts=1)
    trade = next(e for e in events if e.event_type is EventType.TRADE)
    assert trade.is_self_trade


def test_seq_is_monotonic():
    eng = MatchingEngine("ACME")
    seqs = []
    seqs += [e.seq for e in eng.submit(1, "A", Side.SELL, 100, 5, ts=0)]
    seqs += [e.seq for e in eng.submit(2, "B", Side.BUY, 100, 5, ts=1)]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
