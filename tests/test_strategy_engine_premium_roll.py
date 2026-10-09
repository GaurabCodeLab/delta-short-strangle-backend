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

    async def parse_option_positions(self):
        return list(self.legs)

    async def get_best_quote(self, product_id):
        return self.quotes[product_id]

    async def get_index_price(self, symbol="BTCUSDT"):
        return self.index_price


class FakePositions:
    def __init__(self, replacement=None, wings=None):
        self.replacement = replacement
        self.wings = wings or {}

    async def get_option_product_closest_to_premium(self, **kwargs):
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
    return Settings("key", "secret", "test", 0.001, 10.0, 1e9, 1e9, "INFO", False)


def make_engine(exchange, positions, executor, settings):
    return StrategyEngine(exchange, positions, executor, settings)


@pytest.mark.asyncio
async def test_waits_for_preexisting_matching_short_pair(settings):
    exchange = FakeExchange([])
    engine = make_engine(exchange, FakePositions(), FakeExecutor(exchange), settings)

    assert await engine._find_active_short_pair() is None
    exchange.legs = [make_leg(1, "put", 90, -1, 2), make_leg(2, "call", 110, -1, 2)]
    pair = await engine._find_active_short_pair()

    assert pair is not None
    assert pair.put_leg.product_id == 1
    assert pair.call_leg.product_id == 2


@pytest.mark.asyncio
async def test_roll_closes_triggered_leg_before_opening_replacement(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 3)
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

    await engine._roll_short_leg(put, call, exchange.quotes[2])

    assert executor.calls == [(1, "buy", 1, True), (3, "sell", 1, False)]
    assert [leg.product_id for leg in exchange.legs] == [2, 3]


@pytest.mark.asyncio
async def test_monitor_rolls_only_the_leg_that_reaches_double_premium(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 3)
    replacement = make_leg(3, "put", 95, -1, 2.8)
    exchange = FakeExchange(
        [put, call],
        {
            1: Quote(3.0, 4.0),
            2: Quote(2.8, 3.0),
            3: Quote(2.7, 2.9),
        },
    )
    product = {
        "id": 3,
        "symbol": "P-95",
        "strike_price": 95,
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

    assert executor.calls[:2] == [(1, "buy", 1, True), (3, "sell", 1, False)]
    assert [leg.product_id for leg in exchange.legs] == [2, 3]


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
    assert engine.state.status == "Completed - Profit"


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
