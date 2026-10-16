import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from src.strategy_engine import StrategyEngine
from src.models import OptionLeg, Quote, ShortStraddle, ShortStrangle


class DummySettings:
    def __init__(self):
        self.profit_target = 1000.0
        self.stop_loss = 1000.0
        self.poll_interval_seconds = 0.001


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def execute_market_single_submission_with_fill_confirmation(self, product_id, side, size, reduce_only=False):
        self.calls.append((product_id, side, size, reduce_only))
        await asyncio.sleep(0)
        return float(size)


class ExchangeWithCancel:
    def __init__(self, legs=None, quotes=None, index_price=100.0, cancel_raises=False):
        self._legs = legs or []
        self._quotes = quotes or {}
        self._index = index_price
        self._cancel_raises = cancel_raises

    async def cancel_all_orders(self):
        if self._cancel_raises:
            raise RuntimeError("cancel failed")
        await asyncio.sleep(0)

    async def parse_option_positions(self):
        return self._legs

    async def get_best_quote(self, product_id):
        return self._quotes.get(product_id, Quote(best_bid=0.5, best_ask=0.6))

    async def get_index_price(self, symbol="BTCUSDT"):
        return self._index


def make_leg(pid, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=pid, symbol=f"OPT-{pid}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=entry_price, contract_value=1.0)


@pytest.mark.asyncio
async def test_cancel_pending_orders_noop_and_exception():
    settings = DummySettings()
    executor = FakeExecutor()
    ex = ExchangeWithCancel()
    engine = StrategyEngine(ex, None, executor, settings)

    # callable present and succeeds
    await engine._cancel_pending_orders()

    # callable present and raises
    ex2 = ExchangeWithCancel(cancel_raises=True)
    engine2 = StrategyEngine(ex2, None, executor, settings)
    await engine2._cancel_pending_orders()


@pytest.mark.asyncio
async def test_cancel_pending_orders_without_exchange_support():
    class ExchangeWithoutCancel:
        pass

    engine = StrategyEngine(ExchangeWithoutCancel(), None, FakeExecutor(), DummySettings())

    await engine._cancel_pending_orders()


@pytest.mark.asyncio
async def test_close_record_prefers_exchange_realized_pnl_and_fill_details():
    leg = make_leg(1, "put", 90, -2, entry_price=5.0)
    engine = StrategyEngine(ExchangeWithCancel(), None, FakeExecutor(), DummySettings())
    engine.executor.last_fill_details = {
        "product_id": 1,
        "filled_qty": 1.5,
        "avg_fill_price": 4.0,
        "exit_price": 3.9,
        "realized_pnl": 6.6,
    }

    await engine._record_realized_pnl_for_close(leg)

    assert engine._realized_pnl == pytest.approx(6.6)
    assert engine.closed_positions[0]["size"] == pytest.approx(1.5)
    assert engine.closed_positions[0]["closed_price"] == pytest.approx(3.9)


@pytest.mark.asyncio
async def test_close_record_calculates_pnl_from_fill_price_when_api_pnl_missing():
    leg = make_leg(1, "put", 90, -2, entry_price=5.0)
    engine = StrategyEngine(ExchangeWithCancel(), None, FakeExecutor(), DummySettings())
    engine.executor.last_fill_details = {
        "product_id": 1,
        "filled_qty": 1.5,
        "avg_fill_price": 4.0,
    }

    await engine._record_realized_pnl_for_close(leg)

    assert engine._realized_pnl == pytest.approx(1.5)
    assert engine.closed_positions[0]["size"] == pytest.approx(1.5)
    assert engine.closed_positions[0]["closed_price"] == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_close_record_falls_back_to_quote_and_skips_history_if_quote_fails():
    leg = make_leg(1, "put", 90, -1, entry_price=2.0)
    exchange = ExchangeWithCancel(quotes={1: Quote(best_bid=0.5, best_ask=1.25)})
    engine = StrategyEngine(exchange, None, FakeExecutor(), DummySettings())

    await engine._record_realized_pnl_for_close(leg)

    assert engine._realized_pnl == pytest.approx(0.75)
    assert engine.closed_positions[0]["closed_price"] == pytest.approx(1.25)

    class MissingQuoteExchange(ExchangeWithCancel):
        async def get_best_quote(self, product_id):
            raise RuntimeError("quote unavailable")

    failed_engine = StrategyEngine(MissingQuoteExchange(), None, FakeExecutor(), DummySettings())
    await failed_engine._record_realized_pnl_for_close(leg)

    assert failed_engine._realized_pnl == 0
    assert failed_engine.closed_positions == []


@pytest.mark.asyncio
async def test_find_active_short_pair_refreshes_stale_managed_positions():
    stale_put = make_leg(1, "put", 90, -1)
    active_put = make_leg(2, "put", 90, -1)
    active_call = make_leg(3, "call", 110, -1)
    exchange = ExchangeWithCancel(legs=[active_put, active_call])
    engine = StrategyEngine(exchange, None, FakeExecutor(), DummySettings())
    engine._managed_legs = {stale_put.product_id: stale_put}

    pair = await engine._find_active_short_pair()

    assert pair is not None
    assert pair.put_leg.product_id == active_put.product_id
    assert pair.call_leg.product_id == active_call.product_id
    assert engine._managed_legs == {2: active_put, 3: active_call}


@pytest.mark.asyncio
async def test_find_active_short_pair_reuses_valid_cached_pair():
    put = make_leg(1, "put", 90, -1, entry_price=2.0)
    call = make_leg(2, "call", 110, -1, entry_price=3.0)
    call.expiry = put.expiry
    exchange = ExchangeWithCancel(legs=[put, call])
    engine = StrategyEngine(exchange, None, FakeExecutor(), DummySettings())
    engine._managed_legs = {put.product_id: put, call.product_id: call}
    engine._active_put_leg = put
    engine._active_call_leg = call

    pair = await engine._find_active_short_pair()

    assert isinstance(pair, ShortStrangle)
    assert pair.put_leg is put
    assert pair.call_leg is call
    assert engine._reference_premiums == {"put": 2.0, "call": 3.0}


@pytest.mark.asyncio
async def test_close_all_continues_after_partial_exit_fill():
    put = make_leg(1, "put", 90, -1)
    call = make_leg(2, "call", 110, -1)

    class PartialExitExecutor(FakeExecutor):
        async def execute_market_single_submission_with_fill_confirmation(
            self, product_id, side, size, reduce_only=False
        ):
            self.calls.append((product_id, side, size, reduce_only))
            return size / 2 if product_id == 1 else size

    exchange = ExchangeWithCancel(
        legs=[put, call],
        quotes={1: Quote(0.5, 0.75), 2: Quote(0.5, 0.75)},
    )
    executor = PartialExitExecutor()
    engine = StrategyEngine(exchange, None, executor, DummySettings())
    engine._managed_legs = {put.product_id: put, call.product_id: call}

    with pytest.raises(RuntimeError, match="Failed to close all option positions"):
        await engine._close_all_open_option_positions()

    assert executor.calls == [(1, "buy", 1, True), (2, "buy", 1, True)]
    assert call.product_id not in engine._managed_legs


@pytest.mark.asyncio
async def test_ensure_iron_fly_trims_excess_existing_wing_quantity():
    put = make_leg(1, "put", 100, -1, entry_price=2.0)
    call = make_leg(2, "call", 100, -1, entry_price=2.0)
    put_wing = make_leg(3, "put", 90, 2, entry_price=1.0)
    call_wing = make_leg(4, "call", 110, 1, entry_price=1.0)
    exchange = ExchangeWithCancel(
        legs=[put, call, put_wing, call_wing],
        quotes={3: Quote(best_bid=0.5, best_ask=0.75)},
    )
    executor = FakeExecutor()
    engine = StrategyEngine(exchange, None, executor, DummySettings())
    engine._managed_legs = {
        leg.product_id: leg for leg in [put, call, put_wing, call_wing]
    }

    await engine._ensure_iron_fly(ShortStraddle(put, call))

    assert executor.calls == [(3, "sell", 1, True)]
    assert engine._managed_legs[put_wing.product_id].size == 1
    assert engine._iron_fly_mode is True


def test_leg_market_value_and_realized_pnl_respect_side_quantity_and_contract_value():
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    short = OptionLeg(1, "P", "put", 90, expiry, -2, 3, 1.5)
    long = OptionLeg(2, "C", "call", 110, expiry, 2, 3, 1.5)
    quote = Quote(best_bid=4, best_ask=5)

    assert StrategyEngine._calculate_realized_pnl_from_close(short, quote) == pytest.approx(-6)
    assert StrategyEngine._calculate_realized_pnl_from_close(long, quote) == pytest.approx(3)
    assert StrategyEngine._calculate_realized_pnl_from_fill_price(short, 5) == pytest.approx(-6)
    assert StrategyEngine._calculate_realized_pnl_from_fill_price(long, 5, filled_qty=1) == pytest.approx(3)
