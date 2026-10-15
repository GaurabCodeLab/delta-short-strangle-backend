from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.config import Settings
from src.exchange_client import DeltaExchangeClient
from src.models import OptionLeg, ShortStraddle, ShortStrangle
from src.order_executor import OrderExecutor
from src.position_manager import PositionManager

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyState:
    triggered: bool = False
    last_index_price: Optional[float] = None
    status_message: str = "starting"
    action: str = "waiting for short strangle"
    status: str = "initializing"
    trigger_price: Optional[str] = None
    trigger_pnl: Optional[float] = None
    previous_index_price: Optional[float] = None
    reference_strike: Optional[float] = None
    current_structure: str = "unknown"
    threshold: Optional[float] = None
    last_transition: Optional[str] = None
    combined_pnl: Optional[float] = None
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    exit_reason: Optional[str] = None
    iron_fly_breakevens: Optional[tuple[float, float]] = None
    reference_premiums: Optional[dict[str, float]] = None
    adjustment_thresholds: Optional[dict[str, float]] = None
    live_best_asks: Optional[dict[str, float]] = None
    adjustment_candidate: Optional[dict[str, float | str | None]] = None


class StrategyEngine:
    def __init__(
        self,
        exchange: DeltaExchangeClient,
        positions: PositionManager,
        executor: OrderExecutor,
        settings: Settings,
    ) -> None:
        self.exchange = exchange
        self.positions = positions
        self.executor = executor
        self.settings = settings
        self.state = StrategyState()
        self._adjustment_lock = asyncio.Lock()
        self._exit_lock = asyncio.Lock()
        self._current_structure: str = "unknown"
        self._realized_pnl = 0.0
        self.closed_positions: list[dict] = []
        self._active_call_leg = None
        self._active_put_leg = None
        self._iron_fly_mode = False
        self._iron_fly_breakevens: tuple[float, float] | None = None
        self._exit_started = False
        self._managed_legs: dict[int, OptionLeg] | None = None
        self._pnl_complete = True
        self._strategy_active = False
        self._reference_premiums: dict[str, float] = {}

    def _set_strategy_state(
        self,
        action: str,
        status: str,
        status_message: str | None = None,
        trigger_price: str | None = None,
        trigger_pnl: float | None = None,
    ) -> None:
        self.state.action = action
        self.state.status = status
        if status_message is not None:
            self.state.status_message = status_message
        self.state.trigger_price = trigger_price
        self.state.trigger_pnl = trigger_pnl

    @staticmethod
    def _calculate_realized_pnl_from_close(leg, quote) -> float:
        if leg.size > 0:
            return abs(leg.size) * leg.contract_value * (quote.best_bid - leg.entry_price)
        return abs(leg.size) * leg.contract_value * (leg.entry_price - quote.best_ask)

    @staticmethod
    def _calculate_realized_pnl_from_fill_price(leg, fill_price: float, filled_qty: float | None = None) -> float:
        qty = abs(leg.size) if filled_qty is None else float(filled_qty)
        if leg.size > 0:
            return qty * leg.contract_value * (fill_price - leg.entry_price)
        return qty * leg.contract_value * (leg.entry_price - fill_price)

    async def _cancel_pending_orders(self) -> None:
        cancel_orders = getattr(self.exchange, "cancel_all_orders", None)
        if callable(cancel_orders):
            try:
                await cancel_orders()
            except Exception as exc:
                LOGGER.warning("Unable to cancel pending orders: %s", exc)
            return
        LOGGER.info("No pending-order cancellation support available in exchange client")

    async def _close_all_open_option_positions(self) -> None:
        if self._managed_legs is None:
            legs = await self.exchange.parse_option_positions()
            self._managed_legs = {leg.product_id: leg for leg in legs}
        else:
            legs = list(self._managed_legs.values())
        if not legs:
            LOGGER.info("No open option positions found to close")
            return

        failures = []
        for leg in legs:
            size = abs(leg.size)
            if size <= 0:
                continue
            side = "buy" if leg.size < 0 else "sell"
            try:
                filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                    product_id=leg.product_id,
                    side=side,
                    size=size,
                    reduce_only=True,
                )
                if filled + 1e-8 < size:
                    raise RuntimeError(f"Exit close was not fully filled: {filled}/{size}")
                await self._record_realized_pnl_for_close(leg)
                self._managed_legs.pop(leg.product_id, None)
            except Exception as exc:
                failures.append(f"product {leg.product_id}: {exc}")
        if failures:
            raise RuntimeError("Failed to close all option positions: " + "; ".join(failures))

    async def _record_realized_pnl_for_close(self, leg) -> None:
        realized_pnl = 0.0
        closed_price = None
        filled_qty = abs(leg.size)

        last_fill = getattr(self.executor, "last_fill_details", None)
        if isinstance(last_fill, dict) and last_fill.get("product_id") == leg.product_id:
            api_realized_pnl = last_fill.get("realized_pnl")
            fill_price = last_fill.get("avg_fill_price")
            if api_realized_pnl is not None:
                realized_pnl = float(api_realized_pnl)
                closed_price = float(last_fill.get("exit_price") if last_fill.get("exit_price") is not None else fill_price)
                filled_qty = last_fill.get("filled_qty", abs(leg.size))
            elif fill_price is not None:
                filled_qty = last_fill.get("filled_qty", abs(leg.size))
                realized_pnl = self._calculate_realized_pnl_from_fill_price(leg, float(fill_price), float(filled_qty))
                closed_price = float(fill_price)

        if realized_pnl == 0.0 and closed_price is None:
            try:
                quote = await self.exchange.get_best_quote(leg.product_id)
            except Exception:
                LOGGER.debug("Unable to compute realized P&L for closing leg %s", leg.product_id)
                return
            realized_pnl = self._calculate_realized_pnl_from_close(leg, quote)
            closed_price = float(quote.best_ask if leg.size < 0 else quote.best_bid)

        self._realized_pnl += realized_pnl
        self.closed_positions.append(
            {
                "product_id": leg.product_id,
                "symbol": leg.symbol,
                "option_type": leg.option_type,
                "strike": leg.strike,
                "expiry": leg.expiry.isoformat() if leg.expiry is not None else None,
                "side": "short" if leg.size < 0 else "long",
                "size": float(filled_qty),
                "entry_price": float(leg.entry_price),
                "closed_price": closed_price,
                "closed_time": datetime.utcnow().isoformat() + "Z",
                "realized_pnl": float(realized_pnl),
            }
        )
    async def _find_active_short_pair(self) -> ShortStrangle | ShortStraddle | None:
        fetched_positions = self._managed_legs is None
        if self._managed_legs is None:
            fetched = await self.exchange.parse_option_positions()
            self._managed_legs = {leg.product_id: leg for leg in fetched}
        legs = list(self._managed_legs.values())
        shorts = [leg for leg in legs if leg.size < 0]
        puts = [leg for leg in shorts if leg.option_type.lower() == "put"]
        calls = [leg for leg in shorts if leg.option_type.lower() == "call"]

        if self._active_put_leg is not None and self._active_call_leg is not None:
            put = next((leg for leg in puts if leg.product_id == self._active_put_leg.product_id), None)
            call = next((leg for leg in calls if leg.product_id == self._active_call_leg.product_id), None)
            if (
                put is not None
                and call is not None
                and put.expiry.date() == call.expiry.date()
                and math.isclose(abs(put.size), abs(call.size), rel_tol=0, abs_tol=5e-4)
                and put.strike <= call.strike
            ):
                self._initialize_reference_premiums(put, call)
                return ShortStrangle(put, call) if not math.isclose(put.strike, call.strike, abs_tol=1e-8) else ShortStraddle(put, call)

        pair = self._build_short_pair(legs)
        if pair is None and not fetched_positions:
            fetched = await self.exchange.parse_option_positions()
            self._managed_legs = {leg.product_id: leg for leg in fetched}
            pair = self._build_short_pair(fetched)
        if pair is None:
            return None

        put, call = pair
        self._active_put_leg, self._active_call_leg = put, call
        self._initialize_reference_premiums(put, call)
        return ShortStraddle(put, call) if math.isclose(put.strike, call.strike, rel_tol=0, abs_tol=1e-8) else ShortStrangle(put, call)

    @staticmethod
    def _build_short_pair(legs) -> tuple[OptionLeg, OptionLeg] | None:
        shorts = [leg for leg in legs if leg.size < 0]
        puts = [leg for leg in shorts if leg.option_type.lower() == "put"]
        calls = [leg for leg in shorts if leg.option_type.lower() == "call"]
        if len(shorts) != 2 or len(puts) != 1 or len(calls) != 1:
            return None

        put, call = puts[0], calls[0]
        same_expiry = put.expiry.date() == call.expiry.date()
        same_quantity = math.isclose(abs(put.size), abs(call.size), rel_tol=0, abs_tol=5e-4)
        if not same_expiry or not same_quantity or put.strike > call.strike:
            return None
        return put, call

    def _initialize_reference_premiums(self, put_leg, call_leg) -> None:
        self._reference_premiums.setdefault("put", float(put_leg.entry_price))
        self._reference_premiums.setdefault("call", float(call_leg.entry_price))

    async def _reconcile_managed_position(self, product_id: int) -> OptionLeg | None:
        positions = await self.exchange.parse_option_positions()
        actual = next((leg for leg in positions if leg.product_id == product_id and abs(leg.size) > 0), None)
        if self._managed_legs is None:
            self._managed_legs = {}
        if actual is None:
            self._managed_legs.pop(product_id, None)
        else:
            self._managed_legs[product_id] = actual
        return actual

    async def _submit_short_close(self, leg) -> None:
        size = abs(leg.size)
        filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
            product_id=leg.product_id,
            side="buy",
            size=size,
            reduce_only=True,
        )
        if filled + 1e-8 < size:
            await self._reconcile_managed_position(leg.product_id)
            raise RuntimeError(f"Close was not fully filled for product {leg.product_id}: {filled}/{size}")
        await self._record_realized_pnl_for_close(leg)
        if self._managed_legs is not None:
            self._managed_legs.pop(leg.product_id, None)

    async def _roll_short_leg(self, trigger_leg, leg_to_replace) -> None:
        async with self._adjustment_lock:
            if trigger_leg.option_type.lower() == leg_to_replace.option_type.lower():
                raise ValueError("A premium adjustment must replace the opposite option type")
            if trigger_leg.expiry.date() != leg_to_replace.expiry.date():
                raise ValueError("Active option legs must have the same expiry before adjustment")
            if not math.isclose(abs(trigger_leg.size), abs(leg_to_replace.size), rel_tol=0, abs_tol=5e-4):
                raise ValueError("Active option legs must have matching quantities before adjustment")

            option_type = trigger_leg.option_type.lower()
            reference_premium = self._reference_premiums.setdefault(option_type, float(trigger_leg.entry_price))
            current = await self.exchange.get_best_quote(trigger_leg.product_id)
            trigger_premium = reference_premium * self.settings.adjustment_multiplier
            if current.best_ask < trigger_premium:
                return
            replacement_quote_target = float(current.best_ask)
            replacement = await self.positions.get_option_product_closest_to_premium(
                active_leg=leg_to_replace,
                remaining_leg=trigger_leg,
                target_premium=replacement_quote_target,
            )
            if replacement is None:
                LOGGER.warning("No narrowing replacement found for %s at premium %.8f", leg_to_replace.symbol, replacement_quote_target)
                return

            replacement_type = PositionManager._normalize_option_type(replacement.get("contract_type"))
            replacement_expiry = PositionManager._to_dt_safe(replacement.get("settlement_time"))
            replacement_strike = float(replacement.get("strike_price") or 0)
            if replacement_type and replacement_type != leg_to_replace.option_type.lower():
                raise RuntimeError("Selected replacement has the wrong option type")
            if replacement_expiry is None or replacement_expiry.date() != leg_to_replace.expiry.date():
                raise RuntimeError("Selected replacement does not have the active expiry")
            if not min(trigger_leg.strike, leg_to_replace.strike) <= replacement_strike <= max(trigger_leg.strike, leg_to_replace.strike):
                raise RuntimeError("Selected replacement would not narrow the active strike gap")

            replacement_quote = await self.exchange.get_best_quote(int(replacement["id"]))
            if replacement_quote.best_bid is None or replacement_quote.best_bid <= 0:
                raise RuntimeError("Selected replacement does not have a valid bid quote")
            await self._submit_short_close(leg_to_replace)
            filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(replacement["id"]),
                side="sell",
                size=abs(leg_to_replace.size),
                reduce_only=False,
            )
            if filled + 1e-8 < abs(leg_to_replace.size):
                await self._reconcile_managed_position(int(replacement["id"]))
                raise RuntimeError(f"Replacement short was not fully filled for product {replacement['id']}: {filled}/{abs(leg_to_replace.size)}")
            last_fill = getattr(self.executor, "last_fill_details", None)
            fill_price = last_fill.get("avg_fill_price") if isinstance(last_fill, dict) else None
            replacement_leg = await self._reconcile_managed_position(int(replacement["id"]))
            if replacement_leg is None and fill_price is not None:
                replacement_leg = OptionLeg(
                    product_id=int(replacement["id"]),
                    symbol=str(replacement.get("symbol") or replacement["id"]),
                    option_type=leg_to_replace.option_type,
                    strike=replacement_strike,
                    expiry=replacement_expiry,
                    size=-float(filled),
                    entry_price=float(fill_price),
                    contract_value=float(replacement.get("contract_value") or leg_to_replace.contract_value),
                )
                self._managed_legs[replacement_leg.product_id] = replacement_leg
            if replacement_leg is None:
                raise RuntimeError(f"Replacement position {replacement['id']} could not be reconciled after execution")
            if not math.isclose(abs(replacement_leg.size), abs(trigger_leg.size), rel_tol=0, abs_tol=5e-4):
                raise RuntimeError(f"Replacement position {replacement['id']} quantity does not match the active leg")
            if leg_to_replace.option_type.lower() == "put":
                self._active_put_leg = replacement_leg
            else:
                self._active_call_leg = replacement_leg
            self._reference_premiums[option_type] = replacement_quote_target
            LOGGER.info(
                "Premium adjustment trigger=%s kept_product=%s closed_product=%s replacement_product=%s reference_premium=%.8f replacement_entry=%.8f",
                trigger_leg.option_type,
                trigger_leg.product_id,
                leg_to_replace.product_id,
                replacement["id"],
                replacement_quote_target,
                replacement_leg.entry_price,
            )

    @staticmethod
    def _expiry_pnl_at_price(legs, price: float) -> float:
        total = 0.0
        for leg in legs:
            intrinsic = max(price - leg.strike, 0.0) if leg.option_type.lower() == "call" else max(leg.strike - price, 0.0)
            direction = 1.0 if leg.size > 0 else -1.0
            total += direction * abs(leg.size) * leg.contract_value * (intrinsic - leg.entry_price)
        return total

    def _find_iron_fly_breakevens(self, legs) -> tuple[float, float] | None:
        strikes = sorted({float(leg.strike) for leg in legs})
        if len(strikes) < 2:
            return None
        points = [0.0, *strikes, max(strikes[-1] * 2, strikes[-1] + 1.0)]
        roots: list[float] = []
        for left, right in zip(points, points[1:]):
            pnl_left = self._expiry_pnl_at_price(legs, left) + self._realized_pnl
            pnl_right = self._expiry_pnl_at_price(legs, right) + self._realized_pnl
            if pnl_left == 0:
                roots.append(left)
            if pnl_left * pnl_right < 0:
                roots.append(left + (right - left) * (-pnl_left) / (pnl_right - pnl_left))
        if len(roots) < 2:
            return None
        return min(roots), max(roots)

    async def _ensure_iron_fly(self, straddle: ShortStraddle) -> None:
        strike = straddle.call_leg.strike
        total_credit = (
            abs(straddle.call_leg.size) * straddle.call_leg.contract_value * straddle.call_leg.entry_price
            + abs(straddle.put_leg.size) * straddle.put_leg.contract_value * straddle.put_leg.entry_price
        )
        lower_exposure = abs(straddle.put_leg.size) * straddle.put_leg.contract_value
        upper_exposure = abs(straddle.call_leg.size) * straddle.call_leg.contract_value
        lower_breakeven = strike - total_credit / lower_exposure
        upper_breakeven = strike + total_credit / upper_exposure

        if self._managed_legs is None:
            fetched = await self.exchange.parse_option_positions()
            self._managed_legs = {leg.product_id: leg for leg in fetched}
        open_legs = list(self._managed_legs.values())
        long_legs = [leg for leg in open_legs if leg.size > 0 and leg.expiry.date() == straddle.call_leg.expiry.date()]
        quantity = min(abs(straddle.put_leg.size), abs(straddle.call_leg.size))
        hedges = (
            ("put", lower_breakeven, straddle.put_leg, strike - 1e-8,
             next((leg for leg in long_legs if leg.option_type.lower() == "put" and leg.strike < strike), None)),
            ("call", upper_breakeven, straddle.call_leg, strike + 1e-8,
             next((leg for leg in long_legs if leg.option_type.lower() == "call" and leg.strike > strike), None)),
        )
        for option_type, target, reference_leg, boundary, hedge in hedges:
            existing_quantity = abs(hedge.size) if hedge is not None else 0.0
            if math.isclose(existing_quantity, quantity, rel_tol=0, abs_tol=5e-4):
                continue
            if hedge is not None and existing_quantity > quantity:
                excess = existing_quantity - quantity
                filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                    product_id=hedge.product_id, side="sell", size=excess, reduce_only=True
                )
                if filled + 1e-8 < excess:
                    raise RuntimeError(f"{option_type.title()} hedge excess was not fully closed")
                await self._record_realized_pnl_for_close(hedge)
                hedge.size = quantity
                continue

            product = None
            if hedge is not None:
                product = {
                    "id": hedge.product_id,
                    "symbol": hedge.symbol,
                    "strike_price": hedge.strike,
                    "settlement_time": hedge.expiry.isoformat(),
                    "contract_value": hedge.contract_value,
                }
            if product is None:
                bound = {"max_strike": boundary} if option_type == "put" else {"min_strike": boundary}
                product = await self.positions.get_option_product_for_strike(
                    option_type, target, reference_leg.expiry, **bound
                )
            if product is None:
                raise RuntimeError(f"No live {option_type} hedge strike on the defined-risk side was found")

            order_quantity = quantity - existing_quantity
            hedge_quote = await self.exchange.get_best_quote(int(product["id"]))
            filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(product["id"]), side="buy", size=order_quantity, reduce_only=False
            )
            if filled + 1e-8 < order_quantity:
                raise RuntimeError(f"{option_type.title()} hedge order was not fully filled")
            self._track_hedge(product, option_type, filled, hedge_quote.best_ask, reference_leg)

        self._iron_fly_mode = True
        self._current_structure = "iron_fly"
        self.state.current_structure = "iron_fly"
        self.state.last_transition = "straddle->iron_fly"
        self._set_strategy_state(
            action="opening iron fly",
            status="iron fly formed",
            status_message="short straddle hedged with long call and put wings",
        )

    def _track_hedge(self, product, option_type: str, quantity: float, entry_price: float, reference_leg) -> None:
        if self._managed_legs is None:
            self._managed_legs = {}
        last_fill = getattr(self.executor, "last_fill_details", None)
        fill_price = last_fill.get("avg_fill_price") if isinstance(last_fill, dict) else None
        leg = OptionLeg(
            product_id=int(product["id"]),
            symbol=str(product.get("symbol") or product["id"]),
            option_type=option_type,
            strike=float(product["strike_price"]),
            expiry=PositionManager._to_dt_safe(product.get("settlement_time")) or reference_leg.expiry,
            size=float(quantity),
            entry_price=float(fill_price if fill_price is not None else entry_price),
            contract_value=float(product.get("contract_value") or reference_leg.contract_value),
        )
        existing = self._managed_legs.get(leg.product_id)
        if existing is not None:
            total_size = existing.size + leg.size
            leg.entry_price = (existing.entry_price * existing.size + leg.entry_price * leg.size) / total_size
            leg.size = total_size
        self._managed_legs[leg.product_id] = leg

    async def _calculate_strategy_pnl(self) -> tuple[float, float, float]:
        realized_pnl = self._realized_pnl
        if self._managed_legs is None:
            try:
                fetched = await self.exchange.parse_option_positions()
                self._managed_legs = {leg.product_id: leg for leg in fetched}
            except Exception:
                self._managed_legs = {}

        unrealized_pnl = 0.0
        self._pnl_complete = True
        for leg in self._managed_legs.values():
            try:
                quote = await self.exchange.get_best_quote(leg.product_id)
            except Exception:
                self._pnl_complete = False
                continue
            unrealized_pnl += self._calculate_realized_pnl_from_close(leg, quote)

        return realized_pnl, unrealized_pnl, realized_pnl + unrealized_pnl

    async def _exit_strategy(self, reason: str) -> None:
        async with self._exit_lock:
            if self._exit_started:
                return
            self._exit_started = True
            self._set_strategy_state(
                "exiting", "exit in progress", f"Closing all positions: {reason}"
            )
            try:
                await self._close_all_open_option_positions()
                await self._cancel_pending_orders()
            except Exception:
                self._set_strategy_state(
                    "exit failed", "exit failed",
                    "An error occurred while closing positions; no duplicate exit submission will be made",
                )
                raise
            self.state.triggered = True
            self.state.exit_reason = reason
            status = {
                "profit": "Completed - Profit Target Reached",
                "stop_loss": "Completed - Stop Loss Hit",
            }.get(reason, f"Completed - {reason.replace('_', ' ').title()}")
            self._set_strategy_state(
                action="completed",
                status=status,
                status_message=f"Strategy exit condition reached ({reason}); all open option positions were closed",
                trigger_price=None,
                trigger_pnl=self.state.trigger_pnl,
            )

    async def refresh_live_state(self) -> None:
        _, _, combined_pnl = await self._calculate_strategy_pnl()
        self.state.trigger_pnl = combined_pnl if self._pnl_complete else None
        self.state.combined_pnl = combined_pnl if self._pnl_complete else None
        self.state.profit_target = self.settings.profit_target
        self.state.stop_loss = self.settings.stop_loss

    async def _check_strategy_exit_conditions(self) -> bool:
        if self._exit_started:
            return True
        if not self._strategy_active:
            if await self._find_active_short_pair() is None:
                return False
            self._strategy_active = True
        await self.refresh_live_state()
        combined_pnl = self.state.combined_pnl
        if combined_pnl is None or not self._pnl_complete:
            return False
        if combined_pnl >= self.settings.profit_target:
            await self._exit_strategy("profit")
            return True
        if combined_pnl <= -self.settings.stop_loss:
            await self._exit_strategy("stop_loss")
            return True
        return False

    async def _monitor_premium_roll_iron_fly(self) -> None:
        while not self._exit_started:
            if await self._check_strategy_exit_conditions():
                return

            try:
                index_price = await self.exchange.get_index_price("BTCUSDT")
                self.state.previous_index_price = self.state.last_index_price
                self.state.last_index_price = index_price
            except Exception as exc:
                index_price = None
                LOGGER.warning("Unable to fetch BTC index price: %s", exc)

            pair = await self._find_active_short_pair()
            if pair is None:
                self.state.current_structure = "unknown"
                self._set_strategy_state(
                    action="waiting for short strangle",
                    status="initial position missing",
                    status_message="waiting for one short call and one short put with matching expiry and quantity",
                )
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue
            self._strategy_active = True

            if isinstance(pair, ShortStraddle):
                self.state.current_structure = "straddle"
                self.state.reference_premiums = None
                self.state.adjustment_thresholds = None
                self.state.live_best_asks = None
                self.state.adjustment_candidate = None
                await self._ensure_iron_fly(pair)
            else:
                self.state.current_structure = "strangle"
                put_quote = await self.exchange.get_best_quote(pair.put_leg.product_id)
                call_quote = await self.exchange.get_best_quote(pair.call_leg.product_id)
                self._set_strategy_state(
                    action="monitoring short strangle",
                    status="waiting for premium adjustment trigger",
                    status_message=(
                        "monitoring each short leg for a premium trigger at "
                        f"{self.settings.adjustment_multiplier:g}x its current reference"
                    ),
                )
                put_reference = self._reference_premiums.get("put", pair.put_leg.entry_price)
                call_reference = self._reference_premiums.get("call", pair.call_leg.entry_price)
                multiplier = self.settings.adjustment_multiplier
                put_threshold = multiplier * put_reference
                call_threshold = multiplier * call_reference
                self.state.reference_premiums = {
                    "put": float(put_reference),
                    "call": float(call_reference),
                }
                self.state.adjustment_thresholds = {
                    "put": float(put_threshold),
                    "call": float(call_threshold),
                }
                self.state.live_best_asks = {
                    "put": float(put_quote.best_ask),
                    "call": float(call_quote.best_ask),
                }
                self.state.adjustment_candidate = None
                if put_reference > 0 and put_quote.best_ask >= put_threshold:
                    self.state.adjustment_candidate = {
                        "side": "put",
                        "triggering_side": "put",
                        "replaced_side": "call",
                        "reference_premium": float(put_reference),
                        "threshold_premium": float(put_threshold),
                        "current_best_ask": float(put_quote.best_ask),
                    }
                    await self._roll_short_leg(pair.put_leg, pair.call_leg)
                elif call_reference > 0 and call_quote.best_ask >= call_threshold:
                    self.state.adjustment_candidate = {
                        "side": "call",
                        "triggering_side": "call",
                        "replaced_side": "put",
                        "reference_premium": float(call_reference),
                        "threshold_premium": float(call_threshold),
                        "current_best_ask": float(call_quote.best_ask),
                    }
                    await self._roll_short_leg(pair.call_leg, pair.put_leg)

            if self._iron_fly_mode:
                legs = list(self._managed_legs.values()) if self._managed_legs is not None else []
                straddle_legs = [pair.put_leg, pair.call_leg]
                hedges = [
                    leg for leg in legs
                    if leg.size > 0 and leg.expiry.date() == pair.put_leg.expiry.date()
                ]
                active_legs = [*straddle_legs, *hedges]
                self._iron_fly_breakevens = self._find_iron_fly_breakevens(active_legs)
                self.state.iron_fly_breakevens = self._iron_fly_breakevens
                if self._iron_fly_breakevens is not None and index_price is not None:
                    lower, upper = self._iron_fly_breakevens
                    if lower <= index_price <= upper and (math.isclose(index_price, lower, abs_tol=1e-8) or math.isclose(index_price, upper, abs_tol=1e-8)):
                        await self._exit_strategy("iron_fly_breakeven")
                        return
                    if index_price <= lower or index_price >= upper:
                        await self._exit_strategy("iron_fly_breakeven")
                        return
                self.state.current_structure = "iron_fly"
                self.state.status_message = f"monitoring Iron Fly breakevens={self._iron_fly_breakevens}"

            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def run(self) -> None:
        try:
            await self._monitor_premium_roll_iron_fly()
        finally:
            LOGGER.info("Premium-roll and Iron Fly strategy loop stopped")
