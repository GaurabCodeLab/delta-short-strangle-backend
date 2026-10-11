# Delta BTC Options Strategy Bot

Python async trading bot for Delta Exchange BTC options. It waits for an existing short strangle or short straddle; it does not create the initial position.

## Key Features

- Modular architecture:
  - `exchange_client.py` (Delta API wrapper + retries)
  - `position_manager.py` (position discovery and live state)
  - `strategy_engine.py` (premium-roll and Iron Fly state machine)
  - `order_executor.py` (market order placement + fill confirmation)
- Rolls the opposite short strangle leg when either active leg's current ask reaches its latest reference premium multiplied by `ADJUSTMENT_MULTIPLIER`
- Selects a same-expiry replacement whose bid premium is closest to the triggering leg's live ask and whose strike narrows the gap
- Converts a resulting short straddle to an Iron Fly with quantity-matched long call and put wings near its breakevens
- Monitors strategy PnL and Iron Fly breakevens, closing all managed option legs on an exit condition
- Detailed structured logging

## Strategy Behavior

### Strategy Lifecycle

1. The bot waits for exactly one short call and one short put with the same expiry and quantity. A pre-existing short straddle is accepted and proceeds directly to Iron Fly setup.
2. While the position is a strangle, each leg is monitored independently using its best ask as the estimated close premium. When that ask reaches its side's latest reference premium times `ADJUSTMENT_MULTIPLIER`, the triggering leg stays open and the opposite leg is closed and replaced with a same-expiry option of the same type and quantity. The replacement bid is selected to be closest to the triggering leg's live ask, with strikes restricted to the current gap.
3. When the two short legs share a strike, the bot buys a put wing below and a call wing above that strike, nearest to the short straddle's premium-based breakevens. Wing quantity matches the short quantity.
4. The bot exits all managed option legs at the configured total-PnL profit target, loss limit, or when BTC reaches/passes either calculated Iron Fly breakeven.

Order fills are reconciled into local strategy state to avoid duplicating adjustments while the exchange position endpoint catches up. Test on testnet before using live credentials; market orders and partial fills can still leave risk if the exchange rejects or cannot fill an order.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy env template and set credentials:

```bash
copy .env.example .env
```

3. Run bot:

```bash
python -m src.main
```

4. Run tests:

```bash
python -m pytest tests/test_strategy_workflow.py
```

## Testing

Run the focused strategy workflow test file:

```bash
python -m pytest tests/test_strategy_workflow.py
```

Run the full test suite:

```bash
python -m pytest
```

Run tests with coverage reporting for the `src/` package:

```bash
python -m pytest --cov=src --cov-report=term-missing
```

Generate both terminal and HTML coverage reports:

```bash
python -m pytest --cov=src --cov-report=term-missing --cov-report=html
```

The HTML report is written to `htmlcov/index.html`.

Run a single test case by name:

```bash
python -m pytest tests/test_strategy_workflow.py -k test_short_straddle_monitor_exits_on_profit_target
```

## Environment Variables

- `DELTA_API_KEY`
- `DELTA_API_SECRET`
- `DELTA_BASE_URL` (default: `https://cdn-ind.testnet.deltaex.org`)
- `DELTA_SSL_VERIFY` (default: `false`)
- `POLL_INTERVAL_SECONDS` (default: `0.2`)
- `PROFIT_TARGET` (default: `50`)
- `STOP_LOSS` (default: `100`)
- `ADJUSTMENT_MULTIPLIER` (required, positive finite number; example: `2`)
- `LOG_LEVEL` (default: `INFO`)

Heartbeat logs run on the same interval as `POLL_INTERVAL_SECONDS`.

## Notes

- The code uses `delta-rest-client` and wraps sync methods with `asyncio.to_thread` for compatibility.
- Delta REST field names can vary by account/product type. Mapping logic is centralized in `src/models.py` and `src/exchange_client.py`.
- Test first on a non-production Delta environment.
