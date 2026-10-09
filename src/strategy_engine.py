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
        self._atm_reference_strike: Optional[float] = None
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

    def _set_status(self, message: str) -> None:
        self.state.status_message = message

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
    def _is_valid_short_straddle(straddle: ShortStraddle | None) -> bool:
        if straddle is None:
            return False
        if straddle.put_leg is None or straddle.call_leg is None:
            return False
        if straddle.put_leg.option_type.lower() != "put" or straddle.call_leg.option_type.lower() != "call":
            return False
        if straddle.put_leg.expiry.date() != straddle.call_leg.expiry.date():
            return False
        if straddle.put_leg.size >= 0 or straddle.call_leg.size >= 0:
            return False
        if not math.isclose(abs(straddle.put_leg.size), abs(straddle.call_leg.size), rel_tol=0, abs_tol=5e-4):
            return False
        return math.isclose(straddle.put_leg.strike, straddle.call_leg.strike, rel_tol=0, abs_tol=1e-8)

    @staticmethod
    def _is_valid_short_strangle(strangle: ShortStrangle | None) -> bool:
        if strangle is None:
            return False
        if strangle.put_leg is None or strangle.call_leg is None:
            return False
        if strangle.put_leg.option_type.lower() != "put" or strangle.call_leg.option_type.lower() != "call":
            return False
        if strangle.put_leg.expiry.date() != strangle.call_leg.expiry.date():
            return False
        if strangle.put_leg.size >= 0 or strangle.call_leg.size >= 0:
            return False
        if math.isclose(strangle.put_leg.strike, strangle.call_leg.strike, rel_tol=0, abs_tol=1e-8):
            return False
        if not math.isclose(abs(strangle.put_leg.size), abs(strangle.call_leg.size), rel_tol=0, abs_tol=5e-4):
            return False
        return True

    @staticmethod
    def _compute_leg_market_value(leg, quote) -> float:
        return abs(leg.size) * leg.contract_value * quote.best_ask

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
    async def _get_active_short_option_legs(self) -> list:
        legs = await self.exchange.parse_option_positions()
        return [leg for leg in legs if leg.size < 0]

    async def _find_active_short_pair(self) -> ShortStrangle | ShortStraddle | None:
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
                return ShortStrangle(put, call) if not math.isclose(put.strike, call.strike, abs_tol=1e-8) else ShortStraddle(put, call)

        if len(shorts) == 2 and len(puts) == 1 and len(calls) == 1:
            put, call = puts[0], calls[0]
            same_expiry = put.expiry.date() == call.expiry.date()
            same_size = math.isclose(abs(put.size), abs(call.size), rel_tol=0, abs_tol=5e-4)
            if same_expiry and same_size and put.strike <= call.strike:
                self._active_put_leg, self._active_call_leg = put, call
                if math.isclose(put.strike, call.strike, rel_tol=0, abs_tol=1e-8):
                    return ShortStraddle(put, call)
                return ShortStrangle(put, call)

        fetched = await self.exchange.parse_option_positions()
        self._managed_legs = {leg.product_id: leg for leg in fetched}
        shorts = [leg for leg in fetched if leg.size < 0]
        puts = [leg for leg in shorts if leg.option_type.lower() == "put"]
        calls = [leg for leg in shorts if leg.option_type.lower() == "call"]
        if len(shorts) != 2 or len(puts) != 1 or len(calls) != 1:
            return None
        put, call = puts[0], calls[0]
        if put.expiry.date() != call.expiry.date() or not math.isclose(abs(put.size), abs(call.size), rel_tol=0, abs_tol=5e-4) or put.strike > call.strike:
            return None
        self._active_put_leg, self._active_call_leg = put, call
        return ShortStraddle(put, call) if math.isclose(put.strike, call.strike, rel_tol=0, abs_tol=1e-8) else ShortStrangle(put, call)

    async def _submit_short_close(self, leg) -> None:
        size = abs(leg.size)
        filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
            product_id=leg.product_id,
            side="buy",
            size=size,
            reduce_only=True,
        )
        if filled + 1e-8 < size:
            raise RuntimeError(f"Close was not fully filled for product {leg.product_id}: {filled}/{size}")
        await self._record_realized_pnl_for_close(leg)
        if self._managed_legs is not None:
            self._managed_legs.pop(leg.product_id, None)

    async def _roll_short_leg(self, leg, remaining_leg, remaining_quote) -> None:
        async with self._adjustment_lock:
            current = await self.exchange.get_best_quote(leg.product_id)
            if current.best_ask < 2 * leg.entry_price:
                return
            replacement = await self.positions.get_option_product_closest_to_premium(
                active_leg=leg,
                remaining_leg=remaining_leg,
                target_premium=float(remaining_quote.best_bid),
            )
            if replacement is None:
                LOGGER.warning("No narrowing replacement found for %s at premium %.8f", leg.symbol, remaining_quote.best_bid)
                return

            replacement_quote = await self.exchange.get_best_quote(int(replacement["id"]))
            await self._submit_short_close(leg)
            filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(replacement["id"]),
                side="sell",
                size=abs(leg.size),
                reduce_only=False,
            )
            if filled + 1e-8 < abs(leg.size):
                raise RuntimeError(f"Replacement short was not fully filled for product {replacement['id']}")
            last_fill = getattr(self.executor, "last_fill_details", None)
            fill_price = last_fill.get("avg_fill_price") if isinstance(last_fill, dict) else None
            replacement_leg = OptionLeg(
                product_id=int(replacement["id"]),
                symbol=str(replacement.get("symbol") or replacement["id"]),
                option_type=leg.option_type,
                strike=float(replacement["strike_price"]),
                expiry=PositionManager._to_dt_safe(replacement.get("settlement_time")) or leg.expiry,
                size=-float(filled),
                entry_price=float(fill_price if fill_price is not None else replacement_quote.best_bid),
                contract_value=float(replacement.get("contract_value") or leg.contract_value),
            )
            if self._managed_legs is not None:
                self._managed_legs[replacement_leg.product_id] = replacement_leg
            if leg.option_type.lower() == "put":
                self._active_put_leg = replacement_leg
            else:
                self._active_call_leg = replacement_leg
            LOGGER.info(
                "Rolled short %s product=%s to product=%s toward premium=%.8f",
                leg.option_type,
                leg.product_id,
                replacement["id"],
                remaining_quote.best_bid,
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

    async def _close_short_straddle(self, straddle: ShortStraddle) -> None:
        async with self._exit_lock:
            submitted_orders = []
            for leg in (straddle.put_leg, straddle.call_leg):
                current_leg = await self.positions.find_open_option_leg(
                    option_type=leg.option_type,
                    strike=leg.strike,
                    expiry=leg.expiry,
                    side="short",
                )
                if current_leg is None:
                    continue

                size = abs(current_leg.size)
                filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                    product_id=current_leg.product_id,
                    side="buy",
                    size=size,
                    reduce_only=True,
                )
                await self._record_realized_pnl_for_close(current_leg)
                submitted_orders.append(
                    {
                        "product_id": current_leg.product_id,
                        "filled_size": filled,
                        "requested_size": size,
                    }
                )

            await asyncio.sleep(0.1)
            LOGGER.info("Short straddle exit summary | submitted_orders=%s", submitted_orders)

    async def _monitor_short_straddle(self, straddle: ShortStraddle) -> bool:
        profit_threshold = self.settings.profit_target
        loss_threshold = -self.settings.stop_loss

        while True:
            current_put = await self.positions.find_open_option_leg(
                option_type="put",
                strike=straddle.put_leg.strike,
                expiry=straddle.put_leg.expiry,
                side="short",
            )
            current_call = await self.positions.find_open_option_leg(
                option_type="call",
                strike=straddle.call_leg.strike,
                expiry=straddle.call_leg.expiry,
                side="short",
            )
            if current_put is None and current_call is None:
                return False

            if current_put is None or current_call is None:
                # Do not perform a partial (single-leg) close here. Instead,
                # perform a full strategy exit to ensure we only ever close
                # all positions on overall strategy exit conditions.
                await self._exit_strategy("stop_loss")
                return True

            total_pnl = 0.0
            missing_prices = False
            for leg in (current_put, current_call):
                try:
                    quote = await self.exchange.get_best_quote(leg.product_id)
                except Exception:
                    missing_prices = True
                    continue

                pnl = abs(leg.size) * leg.contract_value * (leg.entry_price - quote.best_ask)
                total_pnl += pnl

            if missing_prices:
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue

            self._set_strategy_state(
                action="monitoring short straddle",
                status="waiting for short straddle exit",
                status_message=(
                    f"short straddle monitoring pnl={total_pnl:.4f} "
                    f"profit_threshold={profit_threshold:.4f} loss_threshold={loss_threshold:.4f}"
                ),
                trigger_price=None,
                trigger_pnl=total_pnl,
            )

            if total_pnl >= profit_threshold or total_pnl <= loss_threshold:
                await self._close_short_straddle(straddle)
                return True

            await asyncio.sleep(self.settings.poll_interval_seconds)

    @staticmethod
    def _crossed_or_touched(
        prev_price: Optional[float],
        curr_price: float,
        strike: float,
        option_type: str = "",
    ) -> bool:
        if math.isclose(curr_price, strike, rel_tol=0, abs_tol=1e-8):
            return True

        if prev_price is None:
            # Only treat a touch/crossing as a trigger when the market has
            # actually moved and crossed the strike from the opposite side.
            return False

        opt = option_type.lower()
        if opt == "put":
            return prev_price > strike and curr_price <= strike
        if opt == "call":
            return prev_price < strike and curr_price >= strike
        return False

    async def _convert_straddle_to_strangle(self, straddle: ShortStraddle, current_index_price: float) -> float:
        if self._atm_reference_strike is None:
            self._atm_reference_strike = float(straddle.put_leg.strike)

        threshold = self.settings.strike_adjustment_threshold
        if current_index_price >= self._atm_reference_strike + threshold:
            LOGGER.info(
                "Straddle->strangle adjustment: index %.2f crossed upper threshold %.2f from reference %.2f",
                current_index_price,
                threshold,
                self._atm_reference_strike,
            )
            qty = abs(straddle.call_leg.size)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=straddle.call_leg.product_id,
                side="buy",
                size=qty,
                reduce_only=True,
            )
            await self._record_realized_pnl_for_close(straddle.call_leg)
            new_strike = self._atm_reference_strike + 2 * threshold
            new_product = await self.positions.get_option_product_for_strike(
                option_type="call",
                strike=new_strike,
                expiry=straddle.call_leg.expiry,
            )
            if new_product is None:
                raise RuntimeError(f"Could not find call product for straddle->strangle conversion at strike {new_strike}")
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(new_product["id"]),
                side="sell",
                size=qty,
                reduce_only=False,
            )
            self._current_structure = "strangle"
            return new_strike

        if current_index_price <= self._atm_reference_strike - threshold:
            LOGGER.info(
                "Straddle->strangle adjustment: index %.2f crossed lower threshold %.2f from reference %.2f",
                current_index_price,
                threshold,
                self._atm_reference_strike,
            )
            qty = abs(straddle.put_leg.size)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=straddle.put_leg.product_id,
                side="buy",
                size=qty,
                reduce_only=True,
            )
            await self._record_realized_pnl_for_close(straddle.put_leg)
            new_strike = self._atm_reference_strike - 2 * threshold
            new_product = await self.positions.get_option_product_for_strike(
                option_type="put",
                strike=new_strike,
                expiry=straddle.put_leg.expiry,
            )
            if new_product is None:
                raise RuntimeError(f"Could not find put product for straddle->strangle conversion at strike {new_strike}")
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(new_product["id"]),
                side="sell",
                size=qty,
                reduce_only=False,
            )
            self._current_structure = "strangle"
            return new_strike

        raise RuntimeError("No straddle-to-strangle adjustment required")

    async def _convert_strangle_to_straddle(self, strangle: ShortStrangle, current_index_price: float) -> float:
        if current_index_price >= strangle.call_leg.strike:
            qty = abs(strangle.put_leg.size)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=strangle.put_leg.product_id,
                side="buy",
                size=qty,
                reduce_only=True,
            )
            await self._record_realized_pnl_for_close(strangle.put_leg)
            new_strike = float(strangle.call_leg.strike)
            new_product = await self.positions.get_option_product_for_strike(
                option_type="put",
                strike=new_strike,
                expiry=strangle.put_leg.expiry,
            )
            if new_product is None:
                raise RuntimeError(f"Could not find put product for strangle->straddle conversion at strike {new_strike}")
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(new_product["id"]),
                side="sell",
                size=qty,
                reduce_only=False,
            )
            self._atm_reference_strike = new_strike
            self._current_structure = "straddle"
            return new_strike

        if current_index_price <= strangle.put_leg.strike:
            qty = abs(strangle.call_leg.size)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=strangle.call_leg.product_id,
                side="buy",
                size=qty,
                reduce_only=True,
            )
            await self._record_realized_pnl_for_close(strangle.call_leg)
            new_strike = float(strangle.put_leg.strike)
            new_product = await self.positions.get_option_product_for_strike(
                option_type="call",
                strike=new_strike,
                expiry=strangle.call_leg.expiry,
            )
            if new_product is None:
                raise RuntimeError(f"Could not find call product for strangle->straddle conversion at strike {new_strike}")
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(new_product["id"]),
                side="sell",
                size=qty,
                reduce_only=False,
            )
            self._atm_reference_strike = new_strike
            self._current_structure = "straddle"
            return new_strike

        raise RuntimeError("No strangle-to-straddle adjustment required")

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
            self._set_strategy_state(
                action="completed",
                status=f"Completed - {reason.replace('_', ' ').title()}",
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
            return False
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
                await self._ensure_iron_fly(pair)
            else:
                self.state.current_structure = "strangle"
                put_quote = await self.exchange.get_best_quote(pair.put_leg.product_id)
                call_quote = await self.exchange.get_best_quote(pair.call_leg.product_id)
                self._set_strategy_state(
                    action="monitoring short strangle",
                    status="waiting for premium doubling",
                    status_message="monitoring each short leg for a 2x premium trigger",
                )
                if pair.put_leg.entry_price > 0 and put_quote.best_ask >= 2 * pair.put_leg.entry_price:
                    await self._roll_short_leg(pair.put_leg, pair.call_leg, call_quote)
                elif pair.call_leg.entry_price > 0 and call_quote.best_ask >= 2 * pair.call_leg.entry_price:
                    await self._roll_short_leg(pair.call_leg, pair.put_leg, put_quote)

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

    async def _monitor_dynamic_straddle_strangle(self) -> bool:
        self._current_structure = "unknown"
        while True:
            if await self._check_strategy_exit_conditions():
                return True

            try:
                index_price = await self.exchange.get_index_price("BTCUSDT")
            except Exception as exc:
                LOGGER.warning("Unable to fetch BTC index price during dynamic adjustment monitoring: %s", exc)
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue
            prev_index_price = self.state.last_index_price
            self.state.last_index_price = index_price
            LOGGER.info(
                "Dynamic adjustment loop: index_price=%.2f previous_index_price=%s reference_strike=%s current_structure=%s",
                index_price,
                prev_index_price,
                self._atm_reference_strike,
                self._current_structure,
            )

            try:
                straddle = await self.positions.detect_short_straddle()
            except Exception:
                straddle = None

            active_short_legs = await self._get_active_short_option_legs()
            if len(active_short_legs) != 2:
                LOGGER.warning(
                    "Initial position reconciliation failed: expected 2 active short legs but found %d",
                    len(active_short_legs),
                )

            if self._is_valid_short_straddle(straddle):
                if self._atm_reference_strike is None:
                    self._atm_reference_strike = float(straddle.put_leg.strike)
                self.state.previous_index_price = prev_index_price
                self.state.reference_strike = self._atm_reference_strike
                self.state.current_structure = "straddle"
                self.state.threshold = self.settings.strike_adjustment_threshold
                LOGGER.info(
                    "Detected short straddle with reference strike %.2f and threshold %.2f",
                    self._atm_reference_strike,
                    self.settings.strike_adjustment_threshold,
                )
                self._current_structure = "straddle"
                if index_price >= self._atm_reference_strike + self.settings.strike_adjustment_threshold:
                    LOGGER.info(
                        "Dynamic straddle->strangle adjustment triggered at index %.2f (previous %.2f) using reference %.2f",
                        index_price,
                        prev_index_price,
                        self._atm_reference_strike,
                    )
                    await self._convert_straddle_to_strangle(straddle, index_price)
                    self._set_strategy_state(
                        action="adjusting short straddle",
                        status="strangle formed",
                        status_message="converted short straddle to short strangle",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue

                if index_price <= self._atm_reference_strike - self.settings.strike_adjustment_threshold:
                    LOGGER.info(
                        "Dynamic straddle->strangle adjustment triggered at index %.2f (previous %.2f) using reference %.2f",
                        index_price,
                        prev_index_price,
                        self._atm_reference_strike,
                    )
                    await self._convert_straddle_to_strangle(straddle, index_price)
                    self.state.previous_index_price = prev_index_price
                    self.state.reference_strike = self._atm_reference_strike
                    self.state.current_structure = self._current_structure
                    self.state.threshold = self.settings.strike_adjustment_threshold
                    self.state.last_transition = "straddle->strangle"
                    self._set_strategy_state(
                        action="adjusting short straddle",
                        status="strangle formed",
                        status_message="converted short straddle to short strangle",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue

                LOGGER.info(
                    "Monitoring short straddle: waiting for threshold crossing at index %.2f from reference %.2f",
                    index_price,
                    self._atm_reference_strike,
                )
                self._set_strategy_state(
                    action="monitoring short straddle",
                    status="waiting for reference drift",
                    status_message="monitoring short straddle for dynamic adjustment",
                    trigger_price=None,
                    trigger_pnl=None,
                )
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue

            try:
                strangle = await self.positions.detect_short_strangle(require_otm=False)
            except Exception:
                strangle = None
            if self._is_valid_short_strangle(strangle):
                self.state.previous_index_price = prev_index_price
                self.state.reference_strike = None
                self.state.current_structure = "strangle"
                self.state.threshold = self.settings.strike_adjustment_threshold
                if len(active_short_legs) != 2:
                    LOGGER.warning(
                        "Strangle reconciliation failed: expected 2 active short legs but found %d",
                        len(active_short_legs),
                    )
                    self._set_strategy_state(
                        action="monitoring dynamic adjustment",
                        status="position mismatch",
                        status_message="expected exactly two active short legs for the strategy",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue
                self._current_structure = "strangle"

                put_exists = await self.positions.find_open_option_leg(
                    option_type=strangle.put_leg.option_type,
                    strike=strangle.put_leg.strike,
                    expiry=strangle.put_leg.expiry,
                    side="short",
                )
                call_exists = await self.positions.find_open_option_leg(
                    option_type=strangle.call_leg.option_type,
                    strike=strangle.call_leg.strike,
                    expiry=strangle.call_leg.expiry,
                    side="short",
                )
                if put_exists is None and call_exists is None:
                    self._set_strategy_state(
                        action="monitoring dynamic adjustment",
                        status="no short strangle found",
                        status_message="no short strangle found",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    return True

                triggered_put = self._crossed_or_touched(
                    prev_index_price,
                    index_price,
                    strangle.put_leg.strike,
                    "put",
                )
                triggered_call = self._crossed_or_touched(
                    prev_index_price,
                    index_price,
                    strangle.call_leg.strike,
                    "call",
                )
                if triggered_put or triggered_call:
                    LOGGER.info(
                        "Dynamic strangle->straddle adjustment triggered at index %.2f (previous %.2f) put_strike=%.2f call_strike=%.2f",
                        index_price,
                        prev_index_price,
                        strangle.put_leg.strike,
                        strangle.call_leg.strike,
                    )
                    await self._convert_strangle_to_straddle(strangle, index_price)
                    self.state.previous_index_price = prev_index_price
                    self.state.reference_strike = self._atm_reference_strike
                    self.state.current_structure = self._current_structure
                    self.state.threshold = self.settings.strike_adjustment_threshold
                    self.state.last_transition = "strangle->straddle"
                    self._set_strategy_state(
                        action="adjusting short strangle",
                        status="straddle formed",
                        status_message="converted short strangle to short straddle",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue

                LOGGER.info(
                    "Monitoring short strangle: waiting for strike touch at index %.2f put_strike=%.2f call_strike=%.2f",
                    index_price,
                    strangle.put_leg.strike,
                    strangle.call_leg.strike,
                )
                self._set_strategy_state(
                    action="monitoring short strangle",
                    status="waiting for strike touch",
                    status_message="monitoring short strangle for dynamic adjustment",
                    trigger_price=None,
                    trigger_pnl=None,
                )
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue

            LOGGER.warning(
                "Required initial short straddle position is missing; waiting for a valid short straddle or short strangle"
            )
            self._set_strategy_state(
                action="waiting for short straddle",
                status="initial position missing",
                status_message=(
                    "required initial short straddle position is missing; "
                    "expected one short call and one short put with same quantity and expiry"
                ),
                trigger_price=None,
                trigger_pnl=None,
            )
            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def run(self) -> None:
        try:
            await self._monitor_premium_roll_iron_fly()
        finally:
            LOGGER.info("Premium-roll and Iron Fly strategy loop stopped")
