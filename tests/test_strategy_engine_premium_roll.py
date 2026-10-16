import asyncio
from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.models import OptionLeg, Quote, ShortStraddle, ShortStrangle
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
    def __init__(self, replacement=None, wings=None, products_by_strike=None):
        self.replacement = replacement
        self.wings = wings or {}
        self.products_by_strike = products_by_strike or {}
        self.replacement_requests = []

    async def get_option_product_closest_to_premium(self, **kwargs):
        self.replacement_requests.append(kwargs)
        if isinstance(self.replacement, list):
            return self.replacement.pop(0)
        return self.replacement

    async def get_option_product_for_strike(self, option_type, strike, expiry, **kwargs):
        if (option_type, strike) in self.products_by_strike:
            return self.products_by_strike[(option_type, strike)]
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
async def test_roll_tracks_fill_when_exchange_position_endpoint_is_stale(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange(
        [put, call],
        {
            1: Quote(1.0, 1.0),
            2: Quote(3.9, 4.0),
            3: Quote(2.8, 2.9),
        },
    )
    replacement = {
        "id": 3,
        "symbol": "P-100",
        "strike_price": 100,
        "settlement_time": EXPIRY.isoformat(),
        "contract_value": 1,
    }

    class StalePositionExecutor(FakeExecutor):
        async def execute_market_single_submission_with_fill_confirmation(
            self, product_id, side, size, reduce_only
        ):
            self.calls.append((product_id, side, size, reduce_only))
            if reduce_only:
                self.exchange.legs = [
                    leg for leg in self.exchange.legs if leg.product_id != product_id
                ]
            elif product_id == 3:
                self.last_fill_details = {
                    "product_id": 3,
                    "filled_qty": size,
                    "avg_fill_price": 2.85,
                }
            return size

    executor = StalePositionExecutor(exchange)
    engine = make_engine(exchange, FakePositions(replacement), executor, settings)

    await engine._roll_short_leg(call, put)

    tracked = engine._managed_legs[3]
    assert tracked.size == -1
    assert tracked.entry_price == pytest.approx(2.85)
    assert tracked.strike == 100
    assert engine._active_put_leg is tracked
    assert [call[0] for call in executor.calls] == [1, 3]


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
@pytest.mark.parametrize(
    ("invalid_pair", "error"),
    [
        ("same_type", "opposite option type"),
        ("expiry", "same expiry"),
        ("quantity", "matching quantities"),
    ],
)
async def test_roll_rejects_invalid_active_pair_before_market_data_or_orders(
    settings, invalid_pair, error
):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    call.expiry = put.expiry
    exchange = FakeExchange([put, call])
    positions = FakePositions({"id": 3})
    executor = FakeExecutor(exchange)
    engine = make_engine(exchange, positions, executor, settings)

    if invalid_pair == "same_type":
        trigger_leg = put
        leg_to_replace = make_leg(3, "put", 100, -1, 2)
        leg_to_replace.expiry = put.expiry
    else:
        trigger_leg = call
        leg_to_replace = put
        if invalid_pair == "expiry":
            trigger_leg.expiry = datetime(2027, 1, 31, tzinfo=timezone.utc)
        else:
            trigger_leg.size = -2

    with pytest.raises(ValueError, match=error):
        await engine._roll_short_leg(trigger_leg, leg_to_replace)

    assert executor.calls == []
    assert positions.replacement_requests == []


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
@pytest.mark.parametrize(
    ("touch_side", "index_price", "target_strike", "replacement_type", "closed_product_id", "wing_strikes"),
    [
        ("call", 110, 110, "put", 1, (100, 120)),
        ("put", 90, 90, "call", 2, (80, 100)),
    ],
)
async def test_index_touch_converts_strangle_to_iron_fly_before_premium_roll(
    settings,
    touch_side,
    index_price,
    target_strike,
    replacement_type,
    closed_product_id,
    wing_strikes,
):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    replacement_product_id = 3
    put_wing_id = 4
    call_wing_id = 5
    lower_breakeven = target_strike - 4
    upper_breakeven = target_strike + 4
    products = {
        (replacement_type, target_strike): {
            "id": replacement_product_id,
            "symbol": f"{replacement_type}-{target_strike}",
            "strike_price": target_strike,
            "settlement_time": EXPIRY.isoformat(),
            "contract_value": 1,
        },
        ("put", lower_breakeven): {
            "id": put_wing_id,
            "symbol": f"put-{wing_strikes[0]}",
            "strike_price": wing_strikes[0],
            "settlement_time": EXPIRY.isoformat(),
            "contract_value": 1,
        },
        ("call", upper_breakeven): {
            "id": call_wing_id,
            "symbol": f"call-{wing_strikes[1]}",
            "strike_price": wing_strikes[1],
            "settlement_time": EXPIRY.isoformat(),
            "contract_value": 1,
        },
    }
    exchange = FakeExchange(
        [put, call],
        {product_id: Quote(1.0, 1.0) for product_id in range(1, 6)},
        index_price=index_price,
    )
    positions = FakePositions(products_by_strike=products)
    completed = asyncio.Event()

    class ConversionExecutor(FakeExecutor):
        async def execute_market_single_submission_with_fill_confirmation(
            self, product_id, side, size, reduce_only
        ):
            filled = await super().execute_market_single_submission_with_fill_confirmation(
                product_id, side, size, reduce_only
            )
            if len(self.calls) == 4:
                completed.set()
            return filled

    executor = ConversionExecutor(
        exchange,
        {
            replacement_product_id: make_leg(
                replacement_product_id, replacement_type, target_strike, -1, 2
            ),
            put_wing_id: make_leg(put_wing_id, "put", wing_strikes[0], 1, 1),
            call_wing_id: make_leg(call_wing_id, "call", wing_strikes[1], 1, 1),
        },
    )
    engine = make_engine(exchange, positions, executor, settings)
    task = asyncio.create_task(engine._monitor_premium_roll_iron_fly())

    await asyncio.wait_for(completed.wait(), timeout=0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.state.current_structure == "iron_fly"
    assert engine._iron_fly_mode is True
    assert positions.replacement_requests == []
    assert executor.calls[0] == (closed_product_id, "buy", 1, True)
    assert closed_product_id == (1 if touch_side == "call" else 2)
    assert sorted(
        (leg.option_type, leg.strike, leg.size)
        for leg in exchange.legs
    ) == sorted(
        [
            ("call", target_strike, -1),
            ("put", target_strike, -1),
            ("put", wing_strikes[0], 1),
            ("call", wing_strikes[1], 1),
        ]
    )


@pytest.mark.asyncio
async def test_index_conversion_rechecks_touch_after_adjustment_lock_is_released(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], index_price=110)
    executor = FakeExecutor(exchange)
    engine = make_engine(exchange, FakePositions(), executor, settings)
    waiting_for_lock = asyncio.Event()
    release_lock = asyncio.Event()

    class DelayedLock:
        def locked(self):
            return True

        async def __aenter__(self):
            waiting_for_lock.set()
            await release_lock.wait()
            return self

        async def __aexit__(self, *_):
            return False

    engine._adjustment_lock = DelayedLock()

    conversion = asyncio.create_task(
        engine._convert_strangle_to_iron_fly(ShortStrangle(put, call), 110)
    )
    await asyncio.wait_for(waiting_for_lock.wait(), timeout=0.2)
    assert engine._conversion_started is True
    exchange.index_price = 100
    release_lock.set()

    assert await conversion is None
    assert engine._conversion_started is False
    assert executor.calls == []
    assert exchange.legs == [put, call]


@pytest.mark.asyncio
async def test_index_conversion_retains_target_when_touch_remains_after_lock_release(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], index_price=110)
    waiting_for_lock = asyncio.Event()
    release_lock = asyncio.Event()

    class DelayedLock:
        def locked(self):
            return True

        async def __aenter__(self):
            waiting_for_lock.set()
            await release_lock.wait()
            return self

        async def __aexit__(self, *_):
            return False

    class NoConversionProduct(FakePositions):
        async def get_option_product_for_strike(self, option_type, strike, expiry, **kwargs):
            return None

    engine = make_engine(exchange, NoConversionProduct(), FakeExecutor(exchange), settings)
    engine._adjustment_lock = DelayedLock()
    conversion = asyncio.create_task(
        engine._convert_strangle_to_iron_fly(ShortStrangle(put, call), 110)
    )

    await asyncio.wait_for(waiting_for_lock.wait(), timeout=0.2)
    exchange.index_price = 120
    release_lock.set()

    with pytest.raises(RuntimeError, match="No put product"):
        await conversion

    assert engine._conversion_started is True
    assert engine._conversion_target_strike == 110
    assert engine._conversion_trigger_side == "call"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("product", "quote", "error"),
    [
        (None, Quote(1, 1), "No put product"),
        (
            {
                "id": 3,
                "contract_type": "call_option",
                "strike_price": 110,
                "settlement_time": EXPIRY.isoformat(),
            },
            Quote(1, 1),
            "wrong option type",
        ),
        (
            {
                "id": 3,
                "strike_price": 110,
                "settlement_time": datetime(2027, 1, 31, tzinfo=timezone.utc).isoformat(),
            },
            Quote(1, 1),
            "active expiry",
        ),
        (
            {
                "id": 3,
                "strike_price": 105,
                "settlement_time": EXPIRY.isoformat(),
            },
            Quote(1, 1),
            "exact opposite-side option",
        ),
        (
            {
                "id": 3,
                "strike_price": 110,
                "settlement_time": EXPIRY.isoformat(),
            },
            Quote(0, 0),
            "valid bid quote",
        ),
    ],
)
async def test_index_conversion_rejects_invalid_common_strike_products(
    settings, product, quote, error
):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)

    class ProductPositions(FakePositions):
        async def get_option_product_for_strike(self, option_type, strike, expiry, **kwargs):
            return product

    exchange = FakeExchange(
        [put, call],
        {1: Quote(1, 1), 2: Quote(1, 1), 3: quote},
        index_price=110,
    )
    engine = make_engine(exchange, ProductPositions(), FakeExecutor(exchange), settings)

    with pytest.raises(RuntimeError, match=error):
        await engine._convert_strangle_to_iron_fly(ShortStrangle(put, call), 110)

    assert engine._conversion_started is True
    assert exchange.legs == [put, call]


@pytest.mark.asyncio
async def test_index_conversion_recovers_when_opposite_short_was_already_closed(settings):
    original_put = make_leg(1, "put", 90, -1, 2)
    trigger_call = make_leg(2, "call", 110, -1, 2)
    replacement = make_leg(3, "put", 110, -1, 2)
    put_wing = make_leg(4, "put", 106, 1, 1)
    call_wing = make_leg(5, "call", 114, 1, 1)
    replacement_product = {
        "id": 3,
        "symbol": "put-110",
        "strike_price": 110,
        "settlement_time": EXPIRY.isoformat(),
        "contract_value": 1,
    }
    products = {
        ("put", 110): replacement_product,
        ("put", 106): {
            "id": 4,
            "symbol": "put-106",
            "strike_price": 106,
            "settlement_time": EXPIRY.isoformat(),
            "contract_value": 1,
        },
        ("call", 114): {
            "id": 5,
            "symbol": "call-114",
            "strike_price": 114,
            "settlement_time": EXPIRY.isoformat(),
            "contract_value": 1,
        },
    }
    exchange = FakeExchange(
        [trigger_call],
        {product_id: Quote(1, 1) for product_id in range(3, 6)},
        index_price=110,
    )
    positions = FakePositions(products_by_strike=products)
    executor = FakeExecutor(
        exchange,
        {3: replacement, 4: put_wing, 5: call_wing},
    )
    engine = make_engine(exchange, positions, executor, settings)

    result = await engine._convert_strangle_to_iron_fly(
        ShortStrangle(original_put, trigger_call), 110
    )

    assert isinstance(result, ShortStraddle)
    assert result.put_leg.product_id == 3
    assert result.call_leg.product_id == 2
    assert engine._iron_fly_mode is True
    assert engine._conversion_started is False
    assert sorted(leg.product_id for leg in exchange.legs) == [2, 3, 4, 5]


@pytest.mark.asyncio
async def test_index_conversion_is_noop_without_a_touch_or_index_price(settings):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], index_price=100)
    engine = make_engine(exchange, FakePositions(), FakeExecutor(exchange), settings)
    pair = ShortStrangle(put, call)

    assert await engine._convert_strangle_to_iron_fly(pair, 100) is None
    assert await engine._convert_strangle_to_iron_fly(pair, None) is None
    assert engine._conversion_started is False
    assert exchange.position_fetches == 0


@pytest.mark.asyncio
async def test_index_conversion_clears_request_if_iron_fly_is_already_established(settings):
    exchange = FakeExchange([])
    engine = make_engine(exchange, FakePositions(), FakeExecutor(exchange), settings)
    engine._conversion_started = True
    engine._iron_fly_mode = True

    assert await engine._convert_strangle_to_iron_fly(None, None) is None
    assert engine._conversion_started is False
    assert engine._conversion_source_pair is None


@pytest.mark.asyncio
async def test_index_conversion_resumes_from_existing_short_straddle(settings):
    source_put = make_leg(1, "put", 90, -1, 2)
    source_call = make_leg(2, "call", 110, -1, 2)
    straddle_put = make_leg(3, "put", 110, -1, 2)
    straddle_call = make_leg(4, "call", 110, -1, 2)
    put_wing = make_leg(5, "put", 106, 1, 1)
    call_wing = make_leg(6, "call", 114, 1, 1)
    products = {
        ("put", 106): {
            "id": 5,
            "symbol": "put-106",
            "strike_price": 106,
            "settlement_time": EXPIRY.isoformat(),
            "contract_value": 1,
        },
        ("call", 114): {
            "id": 6,
            "symbol": "call-114",
            "strike_price": 114,
            "settlement_time": EXPIRY.isoformat(),
            "contract_value": 1,
        },
    }
    exchange = FakeExchange(
        [straddle_put, straddle_call],
        {product_id: Quote(1, 1) for product_id in (5, 6)},
        index_price=110,
    )
    executor = FakeExecutor(exchange, {5: put_wing, 6: call_wing})
    engine = make_engine(exchange, FakePositions(products_by_strike=products), executor, settings)

    result = await engine._convert_strangle_to_iron_fly(
        ShortStrangle(source_put, source_call), 110
    )

    assert isinstance(result, ShortStraddle)
    assert result.put_leg.product_id == 3
    assert result.call_leg.product_id == 4
    assert engine._iron_fly_mode is True
    assert engine._conversion_started is False
    assert executor.calls == [(5, "buy", 1, False), (6, "buy", 1, False)]


@pytest.mark.asyncio
async def test_pending_conversion_fails_safely_when_trigger_leg_cannot_be_reconciled(settings):
    source_put = make_leg(1, "put", 90, -1, 2)
    source_call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([])
    engine = make_engine(exchange, FakePositions(), FakeExecutor(exchange), settings)
    engine._conversion_started = True
    engine._conversion_source_pair = ShortStrangle(source_put, source_call)
    engine._conversion_target_strike = 110
    engine._conversion_trigger_side = "call"

    with pytest.raises(RuntimeError, match="triggering short leg"):
        await engine._convert_strangle_to_iron_fly(None, None)

    assert engine._conversion_started is True


@pytest.mark.asyncio
@pytest.mark.parametrize("conversion_state", ["pending", "established"])
async def test_premium_roll_is_blocked_during_or_after_iron_fly_conversion(settings, conversion_state):
    put = make_leg(1, "put", 90, -1, 2)
    call = make_leg(2, "call", 110, -1, 2)
    exchange = FakeExchange([put, call], {1: Quote(4, 5), 2: Quote(4, 5)})
    executor = FakeExecutor(exchange)
    engine = make_engine(exchange, FakePositions({"id": 3}), executor, settings)
    engine._conversion_started = conversion_state == "pending"
    engine._iron_fly_mode = conversion_state == "established"

    await engine._roll_short_leg(call, put)

    assert executor.calls == []
    assert exchange.legs == [put, call]


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
