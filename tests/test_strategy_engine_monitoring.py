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
    def __init__(self, exchange):
        self.exchange = exchange
        self.calls = []
        self.last_fill_details = None

    async def execute_market_single_submission_with_fill_confirmation(self, product_id, side, size, reduce_only):
        self.calls.append((product_id, side, size, reduce_only))
        if reduce_only:
            self.exchange._legs = [leg for leg in self.exchange._legs if leg.product_id != product_id]
        return float(size)


class FakeExchange:
    def __init__(self, legs=None, bid=1.0, ask=2.0, index_price=1000.0):
        self._legs = legs or []
        self._bid = bid
        self._ask = ask
        self._index = index_price

    async def get_best_quote(self, product_id):
        return Quote(best_bid=self._bid, best_ask=self._ask)

    async def parse_option_positions(self):
        return self._legs

    async def get_index_price(self, symbol="BTCUSDT"):
        return self._index


def make_leg(pid, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=pid, symbol=f"OPT-{pid}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=entry_price, contract_value=1.0)


@pytest.mark.asyncio
async def test_check_strategy_exit_conditions_triggers_profit_and_stop():
    # test profit exit
    settings = SimpleSettings(profit_target=5.0, stop_loss=10.0)
    # two legs to compute unrealized pnl
    leg1 = make_leg(1, "call", 100.0, -1, entry_price=1.0)
    leg2 = make_leg(2, "put", 100.0, -1, entry_price=1.0)
    # set exchange quotes so unrealized is large
    exchange = FakeExchange(legs=[leg1, leg2], bid=100.0, ask=101.0)
    positions = None
    executor = FakeExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    # no realized pnl, but unrealized will be high due to quotes
    triggered = await engine._check_strategy_exit_conditions()
    assert triggered is True
    assert engine.state.exit_reason in {"profit", "stop_loss"}

    # test stop loss negative (simulate large adverse move where shorts lose)
    settings2 = SimpleSettings(profit_target=10000.0, stop_loss=1.0)
    # set quotes very high so short positions incur losses (entry_price - ask becomes negative)
    exchange2 = FakeExchange(legs=[leg1, leg2], bid=100.0, ask=101.0)
    engine2 = StrategyEngine(exchange2, positions, FakeExecutor(exchange2), settings2)
    triggered2 = await engine2._check_strategy_exit_conditions()
    assert triggered2 is True
    assert engine2.state.exit_reason == "stop_loss"
