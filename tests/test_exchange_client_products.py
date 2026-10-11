import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
import pytest

from src.exchange_client import DeltaExchangeClient, ExchangeClientError
from src.models import OptionLeg, Quote
from src.position_manager import PositionManager


@pytest.mark.asyncio
async def test_get_products_raw_pagination_and_dedup():
    # prepare page payloads
    page1 = {"success": True, "result": [{"id": 1}, {"id": 2}]}
    page2 = {"success": True, "result": [{"id": 2}, {"id": 3}]}
    page3 = {"success": True, "result": []}

    calls = [page1, page2, page3]

    async def _call(method_name, **kwargs):
        # simulate a response object with json method
        payload = calls.pop(0)

        def json():
            return payload

        return SimpleNamespace(json=json)

    client = SimpleNamespace()
    client._call = _call
    # bind method
    client.get_products_raw = DeltaExchangeClient.get_products_raw.__get__(client, client.__class__)

    products = await client.get_products_raw()
    assert isinstance(products, list)
    # dedup ensures ids 1,2,3 only once
    ids = {int(p.get("id")) for p in products}
    assert ids == {1, 2, 3}


@pytest.mark.asyncio
async def test_get_products_raw_stops_on_duplicate_and_failed_pages():
    calls = [
        {"success": True, "result": [{"id": 1}]},
        {"success": True, "result": [{"id": 1}]},
        {"success": False, "result": []},
    ]

    async def _call(method_name, **kwargs):
        payload = calls.pop(0)

        def json():
            return payload

        return SimpleNamespace(json=json)

    client = SimpleNamespace()
    client._call = _call
    client.get_products_raw = DeltaExchangeClient.get_products_raw.__get__(client, client.__class__)

    products = await client.get_products_raw()
    assert [p["id"] for p in products] == [1]


@pytest.mark.asyncio
async def test_place_market_order_and_get_order_success_and_failure():
    # place_market_order expects self._order_type_market and _tif_ioc exist
    async def _call_place(method_name, **kwargs):
        if method_name == "place_order":
            return {"order_id": "abc"}
        if method_name == "order_history":
            return {"result": [{"id": "abc", "status": "filled"}]}
        raise Exception("unexpected")

    client = SimpleNamespace()
    client._call = _call_place
    client._order_type_market = "MARKET"
    client._tif_ioc = "IOC"
    # bind methods
    client.place_market_order = DeltaExchangeClient.place_market_order.__get__(client, client.__class__)
    client.get_order = DeltaExchangeClient.get_order.__get__(client, client.__class__)

    resp = await client.place_market_order(product_id=1, side="buy", size=1.0, reduce_only=True)
    assert isinstance(resp, dict)

    order = await client.get_order(order_id="abc", product_id=1)
    assert isinstance(order, dict)
    assert order.get("status") == "filled"

    # test unexpected response type for place_market_order raises
    async def _call_bad(method_name, **kwargs):
        if method_name == "place_order":
            return "not-a-dict"
        return {"result": []}

    client2 = SimpleNamespace()
    client2._call = _call_bad
    client2._order_type_market = "MARKET"
    client2._tif_ioc = "IOC"
    client2.place_market_order = DeltaExchangeClient.place_market_order.__get__(client2, client2.__class__)

    with pytest.raises(ExchangeClientError):
        await client2.place_market_order(product_id=1, side="buy", size=1.0, reduce_only=False)


@pytest.mark.asyncio
async def test_premium_replacement_lookup_chooses_closest_narrowing_live_strike():
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)

    class Exchange:
        async def get_products_raw(self):
            return [
                {"id": 1, "contract_type": "call_options", "strike_price": "120", "settlement_time": expiry.isoformat(), "state": "live"},
                {"id": 2, "contract_type": "call_options", "strike_price": "100", "settlement_time": expiry.isoformat(), "state": "live"},
                {"id": 3, "contract_type": "call_options", "strike_price": "90", "settlement_time": expiry.isoformat(), "state": "live"},
                {"id": 4, "contract_type": "call_options", "strike_price": "80", "settlement_time": expiry.isoformat(), "state": "live"},
                {"id": 5, "contract_type": "call_options", "strike_price": "95", "settlement_time": datetime(2027, 1, 31, tzinfo=timezone.utc).isoformat(), "state": "live"},
                {"id": 6, "contract_type": "call_options", "strike_price": "110", "settlement_time": expiry.isoformat(), "state": "expired"},
            ]

        async def get_best_quote(self, product_id):
            bids = {2: 4.5, 3: 2.9, 4: 2.0, 6: 3.0}
            return Quote(best_bid=bids[product_id], best_ask=bids[product_id] + 0.1)

    active = OptionLeg(1, "C-120", "call", 120, expiry, -1, 2.0)
    remaining = OptionLeg(10, "P-80", "put", 80, expiry, -1, 3.0)
    manager = PositionManager(Exchange())

    replacement = await manager.get_option_product_closest_to_premium(active, remaining, 3.0)

    assert replacement["id"] == 3


@pytest.mark.asyncio
async def test_strike_lookup_respects_iron_fly_wing_bounds():
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)

    class Exchange:
        async def get_products_raw(self):
            return [
                {"id": 1, "contract_type": "call_options", "strike_price": "100", "settlement_time": expiry.isoformat()},
                {"id": 2, "contract_type": "call_options", "strike_price": "110", "settlement_time": expiry.isoformat()},
            ]

    manager = PositionManager(Exchange())

    product = await manager.get_option_product_for_strike("call", 99, expiry, min_strike=100)

    assert product["id"] == 1


@pytest.mark.asyncio
async def test_option_cache_skips_bad_metadata_and_is_reused():
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)

    class Exchange:
        def __init__(self):
            self.product_fetches = 0

        async def get_products_raw(self):
            self.product_fetches += 1
            return [
                {"id": 1, "contract_type": "perpetual_futures", "strike_price": "100", "settlement_time": expiry.isoformat()},
                {"id": 2, "contract_type": "put_options", "strike_price": "bad", "settlement_time": expiry.isoformat()},
                {"id": 3, "contract_type": "call_options", "strike_price": "100", "settlement_time": "not-a-date"},
                {"id": 4, "contract_type": "put_options", "strike_price": "100", "settlement_time": expiry.isoformat()},
            ]

    exchange = Exchange()
    manager = PositionManager(exchange)

    result = await manager.get_option_product_for_strike("put", 100, expiry)
    repeated = await manager.get_option_product_for_strike("put", 100, expiry)

    assert result["id"] == repeated["id"] == 4
    assert exchange.product_fetches == 1


@pytest.mark.asyncio
async def test_premium_lookup_returns_none_when_candidates_have_no_usable_quotes():
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)

    class Exchange:
        async def get_products_raw(self):
            return [
                {"id": 1, "contract_type": "put_options", "strike_price": "90", "settlement_time": expiry.isoformat(), "state": "live"},
                {"id": 2, "contract_type": "put_options", "strike_price": "95", "settlement_time": expiry.isoformat(), "state": "expired"},
            ]

        async def get_best_quote(self, product_id):
            raise RuntimeError("quote unavailable")

    active = OptionLeg(10, "P-90", "put", 90, expiry, -1, 1)
    remaining = OptionLeg(11, "C-110", "call", 110, expiry, -1, 1)
    manager = PositionManager(Exchange())

    assert await manager.get_option_product_closest_to_premium(active, remaining, 1) is None


@pytest.mark.asyncio
async def test_position_detection_handles_index_failure_and_invalid_structures():
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)
    put = OptionLeg(1, "P-90", "put", 90, expiry, -1, 2)
    call = OptionLeg(2, "C-110", "call", 110, expiry, -1, 2)

    class Exchange:
        async def parse_option_positions(self):
            return [put, call]

        async def get_index_price(self, symbol):
            raise RuntimeError("index unavailable")

    manager = PositionManager(Exchange())

    strangle = await manager.detect_short_strangle(require_otm=True)
    assert strangle.put_leg is put
    assert strangle.call_leg is call
    with pytest.raises(RuntimeError, match="No short straddle"):
        await manager.detect_short_straddle()


@pytest.mark.asyncio
async def test_position_detection_rejects_strangle_when_index_is_outside_strikes():
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)
    put = OptionLeg(1, "P-90", "put", 90, expiry, -1, 2)
    call = OptionLeg(2, "C-110", "call", 110, expiry, -1, 2)

    class Exchange:
        async def parse_option_positions(self):
            return [put, call]

        async def get_index_price(self, symbol):
            return 80

    with pytest.raises(RuntimeError, match="No short strangle"):
        await PositionManager(Exchange()).detect_short_strangle(require_otm=True)
