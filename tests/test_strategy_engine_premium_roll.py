import asyncio
from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.models import OptionLeg, Quote, ShortStraddle
from src.strategy_engine import StrategyEngine


EXPIRY = datetime(2026, 12, 31, tzinfo=timezone.utc)


def make_leg(product_id, option_type, strike, size, entry_price):
    return OptionLeg(product_id, f"OPT-{product_id}", option_type, strike, EXPIRY, size, entry_price, 1.0)


class FakeExchange:
    def __init__(self, legs, quotes=None, index_price=100.0):
        self.legs = legs
        self.quotes = quotes or {}
        self.index_price = index_price
        self.position_fetches = 0

    async def parse_option_positions(self):
        self.position_fetches += 1
        return list(self.legs)

    async def get_best_quote(self, product_id):
        return self.quotes[product_id]

    async def get_index_price(self, symbol="BTCUSDT"):
        return self.index_price


class FakePositions:
    def __init__(self, replacement=None, wings=None):
        self.replacement = replacement
        self.wings = wings or {}
        self.replacement_requests = []

    async def get_option_product_closest_to_premium(self, **kwargs):
        self.replacement_requests.append(kwargs)
        if isinstance(self.replacement, list):
            return self.replacement.pop(0)
        return self.replacement

    async def get_option_product_for_strike(self, option_type, strike, expiry, **kwargs):
        return self.wings[option_type]


class FakeExecutor:
    def __init__(self, exchange, product_legs=None):
        self.exchange = exchange
        self.product_legs = product_legs or {}
        self.calls = []
        self.last_fill_details = None
        self.order_event = asyncio.Event()

    async def execute_market_single_submission_with_fill_confirmation(self, product_id, side, size, reduce_only):
        self.calls.append((product_id, side, size, reduce_only))
        if reduce_only:
            self.exchange.legs = [leg for leg in self.exchange.legs if leg.product_id != product_id]
        else:
            leg = self.product_legs.get(product_id)
            if leg is not None:
                self.exchange.legs.append(leg)
            self.order_event.set()
        self.last_fill_details = None
        return size


@pytest.fixture
def settings():
    return Settings("key", "secret", "test", 0.001, 1e9, 1e9, 2.0, "INFO", False)


def make_engine(exchange, positions, executor, settings):
    return StrategyEngine(exchange, positions, executor, settings)


@pytest.mark.asyncio
async def test_waits_for_preexisting_matching_short_pair(settings):
    exchange = FakeExchange([])
    engine = make_engine(exchange, FakePositions(), FakeExecutor(exchange), settings)

    assert await engine._find_active_short_pair() is None
    assert exchange.position_fetches == 1
    exchange.legs = [make_leg(1, "put", 90, -1, 2), make_leg(2, "call", 110, -1, 2)]
    pair = await engine._find_active_short_pair()

    assert pair is not None
    assert exchange.position_fetches == 2
    assert pair.put_leg.product_id == 1
    assert pair.call_leg.product_id == 2


@pytest.mark.asyncio
async def test_roll_closes_triggered_leg_before_opening_replacement(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 1.5)
    replacement_leg = make_leg(3, "put", 100, -1, 2.8)
    exchange = FakeExchange(
        [put, call],
        {1: Quote(3.0, 4.0), 2: Quote(2.8, 3.0), 3: Quote(2.7, 2.9)},
    )
    positions = FakePositions({
        "id": 3,
        "symbol": "P-100",
        "strike_price": 100,
        "settlement_time": EXPIRY.isoformat(),
        "contract_value": 1,
    })
    executor = FakeExecutor(exchange, {3: replacement_leg})
    engine = make_engine(exchange, positions, executor, settings)

    await engine._roll_short_leg(call, put)

    assert executor.calls == [(1, "buy", 1, True), (3, "sell", 1, False)]
    assert [leg.product_id for leg in exchange.legs] == [2, 3]
    assert positions.replacement_requests[0]["target_premium"] == 3.0
    assert engine._reference_premiums["call"] == 3.0
    assert engine._managed_legs[3].entry_price == 2.8


@pytest.mark.asyncio
async def test_roll_does_nothing_below_configured_trigger(settings):
    settings.adjustment_multiplier = 2.5
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], {1: Quote(4.8, 4.9), 2: Quote(2.0, 2.1)})
    executor = FakeExecutor(exchange)
    engine = make_engine(exchange, FakePositions({"id": 3}), executor, settings)
    engine._reference_premiums["call"] = 2.0

    await engine._roll_short_leg(call, put)

    assert executor.calls == []
    assert engine._reference_premiums["call"] == 2.0


@pytest.mark.asyncio
async def test_roll_keeps_both_legs_when_no_replacement_is_available(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], {1: Quote(3.0, 4.0), 2: Quote(3.0, 4.0)})
    positions = FakePositions(None)
    executor = FakeExecutor(exchange)
    engine = make_engine(exchange, positions, executor, settings)

    await engine._roll_short_leg(call, put)

    assert executor.calls == []
    assert exchange.legs == [put, call]
    assert engine._reference_premiums["call"] == call.entry_price


@pytest.mark.asyncio
async def test_roll_rejects_wrong_expiry_before_closing_any_leg(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], {1: Quote(3.0, 4.0), 2: Quote(3.0, 4.0), 3: Quote(2.8, 3.0)})
    replacement = {
        "id": 3,
        "symbol": "P-100",
        "strike_price": 100,
        "settlement_time": datetime(2027, 1, 31, tzinfo=timezone.utc).isoformat(),
    }
    executor = FakeExecutor(exchange)
    engine = make_engine(exchange, FakePositions(replacement), executor, settings)

    with pytest.raises(RuntimeError, match="active expiry"):
        await engine._roll_short_leg(call, put)

    assert executor.calls == []
    assert exchange.legs == [put, call]


@pytest.mark.asyncio
async def test_partial_close_reconciles_position_and_does_not_open_replacement(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], {1: Quote(3.0, 4.0), 2: Quote(3.0, 4.0), 3: Quote(2.8, 3.0)})
    product = {
        "id": 3,
        "symbol": "P-100",
        "strike_price": 100,
        "settlement_time": EXPIRY.isoformat(),
    }

    class PartialCloseExecutor(FakeExecutor):
        async def execute_market_single_submission_with_fill_confirmation(self, product_id, side, size, reduce_only):
            self.calls.append((product_id, side, size, reduce_only))
            return size / 2 if reduce_only else size

    executor = PartialCloseExecutor(exchange)
    engine = make_engine(exchange, FakePositions(product), executor, settings)
    engine._managed_legs = {put.product_id: put, call.product_id: call}

    with pytest.raises(RuntimeError, match="Close was not fully filled"):
        await engine._roll_short_leg(call, put)

    assert executor.calls == [(1, "buy", 1, True)]
    assert engine._managed_legs[put.product_id] is put


@pytest.mark.asyncio
@pytest.mark.parametrize(("trigger_type", "multiplier"), [("put", 1.5), ("call", 2.5)])
async def test_monitor_rolls_only_the_leg_that_reaches_configured_trigger(settings, trigger_type, multiplier):
    settings.adjustment_multiplier = multiplier
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    trigger_id = 1 if trigger_type == "put" else 2
    replaced_id = 2 if trigger_type == "put" else 1
    replacement_type = "call" if trigger_type == "put" else "put"
    replacement_strike = 105 if replacement_type == "call" else 95
    trigger_premium = multiplier * 2
    replacement = make_leg(3, replacement_type, replacement_strike, -1, 2.8)
    exchange = FakeExchange(
        [put, call],
        {
            1: Quote(trigger_premium - 0.1, trigger_premium) if trigger_type == "put" else Quote(1.9, 2.0),
            2: Quote(trigger_premium - 0.1, trigger_premium) if trigger_type == "call" else Quote(1.9, 2.0),
            3: Quote(2.7, 2.9),
        },
    )
    product = {
        "id": 3,
        "symbol": f"{replacement_type}-{replacement_strike}",
        "strike_price": replacement_strike,
        "settlement_time": EXPIRY.isoformat(),
        "contract_value": 1,
    }
    positions = FakePositions(product, {"put": {"id": 4}, "call": {"id": 5}})
    executor = FakeExecutor(exchange, {3: replacement})
    engine = make_engine(exchange, positions, executor, settings)
    task = asyncio.create_task(engine._monitor_premium_roll_iron_fly())

    await asyncio.wait_for(executor.order_event.wait(), timeout=0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert executor.calls[:2] == [(replaced_id, "buy", 1, True), (3, "sell", 1, False)]
    assert sorted(leg.product_id for leg in exchange.legs) == sorted([trigger_id, 3])
    assert engine._reference_premiums[trigger_type] == trigger_premium


@pytest.mark.asyncio
@pytest.mark.parametrize(("multiplier", "first_trigger", "second_trigger"), [(1.5, 3.0, 4.5), (2.5, 5.0, 12.5)])
async def test_repeated_call_adjustment_uses_configured_latest_reference(settings, multiplier, first_trigger, second_trigger):
    settings.adjustment_multiplier = multiplier
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    first_replacement = make_leg(3, "put", 95, -1, 2.8)
    second_replacement = make_leg(4, "put", 100, -1, 5.8)
    first_product = {
        "id": 3, "symbol": "P-95", "strike_price": 95,
        "settlement_time": EXPIRY.isoformat(), "contract_value": 1,
    }
    second_product = {
        "id": 4, "symbol": "P-100", "strike_price": 100,
        "settlement_time": EXPIRY.isoformat(), "contract_value": 1,
    }
    exchange = FakeExchange(
        [put, call],
        {
            1: Quote(1.9, 2.0), 2: Quote(first_trigger - 0.1, first_trigger),
            3: Quote(2.7, 2.9), 4: Quote(5.5, 5.9),
        },
    )
    positions = FakePositions([first_product, second_product])
    executor = FakeExecutor(exchange, {3: first_replacement, 4: second_replacement})
    engine = make_engine(exchange, positions, executor, settings)

    await engine._roll_short_leg(call, put)
    exchange.quotes[2] = Quote(second_trigger - 0.1, second_trigger)
    await engine._roll_short_leg(call, exchange.legs[1])

    assert executor.calls == [
        (1, "buy", 1, True), (3, "sell", 1, False),
        (3, "buy", 1, True), (4, "sell", 1, False),
    ]
    assert [request["target_premium"] for request in positions.replacement_requests] == [first_trigger, second_trigger]
    assert engine._reference_premiums["call"] == second_trigger
    assert engine._managed_legs[4].entry_price == 5.8


def test_iron_fly_breakevens_include_hedge_debit_and_realized_pnl(settings):
    exchange = FakeExchange([])
    engine = make_engine(exchange, FakePositions(), FakeExecutor(exchange), settings)
    legs = [
        make_leg(1, "put", 100, -1, 5),
        make_leg(2, "call", 100, -1, 5),
        make_leg(3, "put", 90, 1, 2),
        make_leg(4, "call", 110, 1, 2),
    ]
    engine._realized_pnl = 1

    assert engine._find_iron_fly_breakevens(legs) == pytest.approx((93, 107))


@pytest.mark.asyncio
async def test_strategy_exit_submits_closes_only_once(settings):
    exchange = FakeExchange([make_leg(1, "put", 90, -1, 2)])
    executor = FakeExecutor(exchange)
    engine = make_engine(exchange, FakePositions(), executor, settings)

    await engine._exit_strategy("profit")
    await engine._exit_strategy("profit")

    assert executor.calls == [(1, "buy", 1, True)]
    assert engine.state.status == "Completed - Profit Target Reached"


@pytest.mark.asyncio
async def test_strategy_pnl_marks_missing_quote_incomplete(settings):
    leg = make_leg(1, "put", 90, -1, 2)

    class MissingQuoteExchange(FakeExchange):
        async def get_best_quote(self, product_id):
            raise RuntimeError("quote unavailable")

    exchange = MissingQuoteExchange([leg])
    engine = make_engine(exchange, FakePositions(), FakeExecutor(exchange), settings)
    engine._realized_pnl = 12.0

    pnl = await engine._calculate_strategy_pnl()

    assert pnl == (12.0, 0.0, 12.0)
    assert engine._pnl_complete is False


@pytest.mark.asyncio
async def test_close_all_skips_zero_size_and_sells_long_option(settings):
    zero_leg = make_leg(1, "put", 90, 0, 2)
    long_leg = make_leg(2, "call", 110, 1, 3)
    exchange = FakeExchange(
        [zero_leg, long_leg],
        {2: Quote(best_bid=4.0, best_ask=5.0)},
    )
    executor = FakeExecutor(exchange)
    engine = make_engine(exchange, FakePositions(), executor, settings)

    await engine._close_all_open_option_positions()

    assert executor.calls == [(2, "sell", 1, True)]
    assert engine._managed_legs == {zero_leg.product_id: zero_leg}


@pytest.mark.asyncio
async def test_exit_attempts_remaining_legs_after_one_close_fails(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], {1: Quote(1, 2), 2: Quote(1, 2)})

    class OneCloseFailsExecutor(FakeExecutor):
        async def execute_market_single_submission_with_fill_confirmation(self, product_id, side, size, reduce_only):
            self.calls.append((product_id, side, size, reduce_only))
            if product_id == 1:
                raise RuntimeError("exchange rejected close")
            self.exchange.legs = [leg for leg in self.exchange.legs if leg.product_id != product_id]
            return size

    executor = OneCloseFailsExecutor(exchange)
    engine = make_engine(exchange, FakePositions(), executor, settings)
    engine._managed_legs = {put.product_id: put, call.product_id: call}

    with pytest.raises(RuntimeError, match="Failed to close all option positions"):
        await engine._exit_strategy("stop_loss")
    await engine._exit_strategy("stop_loss")

    assert executor.calls == [(1, "buy", 1, True), (2, "buy", 1, True)]
    assert engine.state.status == "exit failed"


@pytest.mark.asyncio
async def test_short_straddle_opens_matching_iron_fly_wings_once(settings):
    put = make_leg(1, "put", 100, -2, 5)
    call = make_leg(2, "call", 100, -2, 5)
    exchange = FakeExchange(
        [put, call],
        {
            3: Quote(1.0, 1.2),
            4: Quote(1.1, 1.3),
        },
    )
    positions = FakePositions(
        wings={
            "put": {"id": 3, "symbol": "P-90", "strike_price": 90, "settlement_time": EXPIRY.isoformat(), "contract_value": 1},
            "call": {"id": 4, "symbol": "C-110", "strike_price": 110, "settlement_time": EXPIRY.isoformat(), "contract_value": 1},
        }
    )
    executor = FakeExecutor(
        exchange,
        {
            3: make_leg(3, "put", 90, 2, 1.2),
            4: make_leg(4, "call", 110, 2, 1.3),
        },
    )
    engine = make_engine(exchange, positions, executor, settings)
    engine._managed_legs = {leg.product_id: leg for leg in exchange.legs}

    straddle = ShortStraddle(put, call)
    await engine._ensure_iron_fly(straddle)
    await engine._ensure_iron_fly(straddle)

    assert executor.calls == [(3, "buy", 2, False), (4, "buy", 2, False)]
    assert engine.state.current_structure == "iron_fly"
