import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from src.strategy_engine import StrategyEngine
from src.models import OptionLeg, Quote


class SimpleSettings:
    def __init__(self, profit_target=50.0, stop_loss=100.0, poll_interval_seconds=0.01):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.poll_interval_seconds = poll_interval_seconds


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def execute_market_single_submission_with_fill_confirmation(self, product_id, side, size, reduce_only=False):
        self.calls.append((product_id, side, size, reduce_only))
        # Simulate immediate full fill
        await asyncio.sleep(0)
        return float(size)


class FakeExchange:
    def __init__(self, quote_bid=1.0, quote_ask=2.0, legs=None):
        self._quote_bid = quote_bid
        self._quote_ask = quote_ask
        self._legs = legs or []

    async def get_best_quote(self, product_id):
        return Quote(best_bid=self._quote_bid, best_ask=self._quote_ask)

    async def parse_option_positions(self):
        return self._legs

    async def get_index_price(self, symbol="BTCUSDT"):
        return 1000.0


def make_leg(product_id, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=product_id, symbol=f"OPT-{product_id}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=float(entry_price), contract_value=1.0)


@pytest.mark.asyncio
async def test_calculate_strategy_pnl_combines_realized_and_unrealized():
    settings = SimpleSettings()
    # create legs and set exchange to return quotes
    leg1 = make_leg(1, "call", 100, -1, entry_price=2.0)
    leg2 = make_leg(2, "put", 100, -1, entry_price=3.0)

    exchange = FakeExchange(quote_bid=5.0, quote_ask=6.0, legs=[leg1, leg2])
    executor = FakeExecutor()
    engine = StrategyEngine(exchange, None, executor, settings)
    # simulate some realized pnl
    engine._realized_pnl = 10.0

    realized, unrealized, combined = await engine._calculate_strategy_pnl()
    assert realized == pytest.approx(10.0)
    # unrealized is computed using _calculate_realized_pnl_from_close with best bid/ask
    assert unrealized != 0.0
    assert combined == pytest.approx(realized + unrealized)
