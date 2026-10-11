from __future__ import annotations

import os
import math
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    api_key: str
    api_secret: str
    base_url: str
    poll_interval_seconds: float
    strike_adjustment_threshold: float
    profit_target: float
    stop_loss: float
    adjustment_multiplier: float
    log_level: str
    ssl_verify: bool


def load_settings() -> Settings:
    api_key = os.getenv("DELTA_API_KEY", "")
    api_secret = os.getenv("DELTA_API_SECRET", "")
    if not api_key or not api_secret:
        raise ValueError("Missing DELTA_API_KEY or DELTA_API_SECRET in environment.")

    poll_interval = float(os.getenv("POLL_INTERVAL_SECONDS", "0.2"))
    if not (poll_interval > 0) or poll_interval != poll_interval:
        raise ValueError("POLL_INTERVAL_SECONDS must be a positive finite number.")

    multiplier_value = os.getenv("ADJUSTMENT_MULTIPLIER")
    if multiplier_value is None or not multiplier_value.strip():
        raise ValueError("Missing ADJUSTMENT_MULTIPLIER in environment.")
    try:
        adjustment_multiplier = float(multiplier_value)
    except ValueError as exc:
        raise ValueError("ADJUSTMENT_MULTIPLIER must be a positive finite number.") from exc
    if not math.isfinite(adjustment_multiplier) or adjustment_multiplier <= 0:
        raise ValueError("ADJUSTMENT_MULTIPLIER must be a positive finite number.")

    return Settings(
        api_key=api_key,
        api_secret=api_secret,
        base_url=os.getenv("DELTA_BASE_URL", "https://cdn-ind.testnet.deltaex.org"),
        poll_interval_seconds=poll_interval,
        strike_adjustment_threshold=float(os.getenv("STRIKE_ADJUSTMENT_THRESHOLD", "1000")),
        profit_target=float(os.getenv("PROFIT_TARGET", "50")),
        stop_loss=float(os.getenv("STOP_LOSS", "100")),
        adjustment_multiplier=adjustment_multiplier,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        ssl_verify=os.getenv("DELTA_SSL_VERIFY", "false").strip().lower() in {"1", "true", "yes", "y"},
    )
