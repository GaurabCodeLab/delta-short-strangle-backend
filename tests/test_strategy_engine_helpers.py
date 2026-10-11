import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from src.strategy_engine import StrategyEngine
from src.models import OptionLeg, Quote


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


def test_leg_market_value_and_realized_pnl_respect_side_quantity_and_contract_value():
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    short = OptionLeg(1, "P", "put", 90, expiry, -2, 3, 1.5)
    long = OptionLeg(2, "C", "call", 110, expiry, 2, 3, 1.5)
    quote = Quote(best_bid=4, best_ask=5)

    assert StrategyEngine._calculate_realized_pnl_from_close(short, quote) == pytest.approx(-6)
    assert StrategyEngine._calculate_realized_pnl_from_close(long, quote) == pytest.approx(3)
    assert StrategyEngine._calculate_realized_pnl_from_fill_price(short, 5) == pytest.approx(-6)
    assert StrategyEngine._calculate_realized_pnl_from_fill_price(long, 5, filled_qty=1) == pytest.approx(3)
