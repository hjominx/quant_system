"""
ETH/Cash micro live runner.

This script runs guarded decision cycles for KRW-ETH:
1. Pull recent 4h or daily candles from Bithumb.
2. Evaluate the selected ETH/Cash strategy profile.
3. Check balances, open orders, spread, account risk, and micro-live caps.
4. Place at most one guarded market or short-lived limit order.

Default mode is dry-run. Real orders require all three:
    - .env TRADING_MODE=LIVE
    - --live
    - --yes-i-understand-risk

Example dry-run:
    python scripts/run_eth_cash_micro_live.py --loop --max-order-krw 10000 --max-position-krw 30000

Example real micro-live:
    python scripts/run_eth_cash_micro_live.py --loop --live --yes-i-understand-risk --max-order-krw 10000 --max-position-krw 30000
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN
from pathlib import Path
from typing import Iterable, Optional
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.bithumb_client import BithumbClient
from execution.models import Balances, Candle, Order, Orderbook, OrderSide
from notifications.discord import (
    COLOR_DANGER,
    COLOR_INFO,
    COLOR_SUCCESS,
    COLOR_WARNING,
    DiscordEmbed,
    DiscordField,
    DiscordNotifier,
)
from notifications.discord_status import (
    AccountSnapshot,
    build_snapshot_embeds,
    fetch_krw_prices_for_balances,
)
from risk.live_guard import (
    RiskLimits,
    RiskGateError,
    assert_risk_allows_action,
    evaluate_risk,
    format_pct,
    load_risk_state,
    save_risk_state,
)
from strategies.base import Signal, StrategyContext, TradeDecision
from strategies.eth_adaptive_alpha import ETHAdaptiveAlphaStrategy
from strategies.eth_cash_regime import ETHCashRegimeStrategy
from strategies.eth_micro_scalper import ETHMicroScalperStrategy
from strategies.eth_pulse_scalper import PulseScalperStrategy
from utils.config import Config, get_config
from utils.logger import setup_logging


MARKET = "KRW-ETH"
DEFAULT_TIMEFRAME = "4h"
DEFAULT_PROFILE = "intraday4h"
DEFAULT_CANDLE_COUNT = 200
KRW_QUANT = Decimal("1")
VOLUME_QUANT = Decimal("0.00000001")
ADAPTIVE_ALPHA_PROFILES = {"stable_quality"}
SCALP_PROFILES = {"scalp_active", "scalp_1m"}
PULSE_PROFILES = {"pulse_10s"}
REBATE_MICRO_PROFILES = {"rebate_micro"}
NIGHT_MICRO_PROFILES = {"night_micro"}
MICRO_PROFILES = REBATE_MICRO_PROFILES | NIGHT_MICRO_PROFILES
SELECTABLE_MARKET_PROFILES = PULSE_PROFILES | MICRO_PROFILES
START_STOP_PROFILES = MICRO_PROFILES
RUNTIME_PROFILE_CHOICES = {
    "stable_quality",
    "scalp_active",
    "scalp_1m",
    "pulse_10s",
    "rebate_micro",
    "night_micro",
}
PROFILE_DEFAULT_TIMEFRAMES = {
    "daily": "1d",
    "intraday4h": "4h",
    "aggressive4h": "4h",
    "stable_quality": "4h",
    "scalp_active": "5m",
    "scalp_1m": "1m",
    "pulse_10s": "1m",
    "rebate_micro": "1m",
    "night_micro": "1m",
}
PROFILE_DEFAULT_CANDLE_COUNTS = {
    "scalp_active": 200,
    "scalp_1m": 200,
    "pulse_10s": 10,
    "rebate_micro": 10,
    "night_micro": 10,
}


@dataclass(frozen=True)
class PilotPlan:
    """Order plan produced after strategy and safety checks."""
    action: str
    reason: str
    krw_amount: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    order_type: str = "market"
    limit_price: Decimal = Decimal("0")
    limit_wait_seconds: int = 0
    dry_run: bool = True


@dataclass(frozen=True)
class PilotState:
    """Persistent micro-live state used for cooldown and daily order caps."""
    date: str
    orders_today: int = 0
    last_order_at: Optional[str] = None
    last_pulse_buy_at: Optional[str] = None
    last_action: Optional[str] = None
    last_status_alert_at: Optional[str] = None
    averaging_count: int = 0


@dataclass(frozen=True)
class RuntimeProfile:
    """Runtime strategy mode resolved for one live cycle."""
    profile: str
    market: str
    timeframe: str
    candle_count: int
    source: str
    market_source: str


@dataclass(frozen=True)
class RebateTurnoverProgress:
    """Persistent turnover progress for the Rebate Micro pilot."""
    total_krw: Decimal = Decimal("0")
    target_krw: Decimal = Decimal("100000000000")
    orders: int = 0
    last_order_at: Optional[str] = None
    last_market: Optional[str] = None


def strategy_params_for_profile(profile: str) -> dict:
    """Return strategy parameters for a named live-trading profile."""
    if profile == "daily":
        return {"periods_per_year": 365}
    if profile == "intraday4h":
        return {
            "fast_window": 10,
            "slow_window": 60,
            "exit_window": 20,
            "momentum_window": 18,
            "momentum_min": 0.04,
            "rsi_window": 14,
            "rsi_min": 42,
            "rsi_max": 88,
            "vol_window": 18,
            "periods_per_year": 365 * 6,
            "max_entry_volatility": 1.45,
            "exit_volatility": 1.90,
            "target_volatility": 0.34,
            "breakout_window": 36,
            "breakout_tolerance": 0.04,
            "exit_momentum": -0.06,
            "exit_rsi": 36,
            "crash_5d": -0.08,
            "crash_10d": -0.14,
            "trailing_stop": 0.12,
            "max_position": 1.00,
            "min_position": 0.20,
        }
    if profile == "aggressive4h":
        return {
            "fast_window": 6,
            "slow_window": 36,
            "exit_window": 12,
            "momentum_window": 9,
            "momentum_min": 0.015,
            "rsi_window": 14,
            "rsi_min": 34,
            "rsi_max": 92,
            "vol_window": 12,
            "periods_per_year": 365 * 6,
            "max_entry_volatility": 2.20,
            "exit_volatility": 2.60,
            "target_volatility": 0.50,
            "breakout_window": 18,
            "breakout_tolerance": 0.10,
            "exit_momentum": -0.10,
            "exit_rsi": 25,
            "crash_5d": -0.12,
            "crash_10d": -0.18,
            "trailing_stop": 0.09,
            "max_position": 0.70,
            "min_position": 0.25,
            "use_exit_ma": False,
            "dip_probe_enabled": True,
            "dip_rsi_min": 26,
            "dip_rsi_max": 38,
            "dip_momentum_floor": -0.035,
            "dip_max_position": 0.55,
        }
    if profile == "stable_quality":
        return {
            "fast_ema": 8,
            "mid_ema": 36,
            "long_ema": 120,
            "donchian_window": 54,
            "exit_donchian_window": 24,
            "min_momentum": 0.040,
            "min_adx": 26,
            "entry_score": 0.70,
            "exit_score": 0.42,
            "target_volatility": 0.26,
            "max_position": 0.55,
            "cooldown_bars": 40,
            "max_realized_volatility": 0.90,
            "max_atr_pct": 0.085,
            "breakout_tolerance": 0.030,
            "hard_stop_pct": 0.035,
        }
    if profile == "scalp_active":
        return {
            "fast_ema": 5,
            "slow_ema": 21,
            "trend_ema": 55,
            "momentum_window": 3,
            "min_momentum": 0.0012,
            "rsi_entry_min": 42,
            "rsi_entry_max": 76,
            "min_atr_pct": 0.00045,
            "max_atr_pct": 0.018,
            "periods_per_year": 365 * 24 * 12,
            "max_realized_volatility": 2.80,
            "target_volatility": 0.75,
            "breakout_window": 18,
            "breakout_tolerance": 0.0035,
            "pullback_tolerance": 0.0060,
            "min_volume_ratio": 0.70,
            "take_profit_pct": 0.0060,
            "stop_loss_pct": 0.0045,
            "trailing_stop_pct": 0.0060,
            "crash_3bar": -0.014,
            "crash_6bar": -0.022,
            "min_position": 0.25,
            "max_position": 0.85,
            "averaging_enabled": True,
            "averaging_max_adds": 2,
            "averaging_min_loss_pct": 0.0040,
            "averaging_max_loss_pct": 0.0140,
            "averaging_step_weight": 0.15,
            "averaging_rsi_min": 30,
            "averaging_rsi_max": 58,
            "averaging_momentum_floor": -0.0025,
            "averaging_trend_floor": 0.994,
        }
    if profile == "scalp_1m":
        return {
            "fast_ema": 8,
            "slow_ema": 34,
            "trend_ema": 89,
            "momentum_window": 2,
            "min_momentum": 0.0005,
            "rsi_entry_min": 40,
            "rsi_entry_max": 78,
            "rsi_exit": 35,
            "atr_window": 14,
            "min_atr_pct": 0.00020,
            "max_atr_pct": 0.012,
            "vol_window": 48,
            "periods_per_year": 365 * 24 * 60,
            "max_realized_volatility": 3.80,
            "target_volatility": 0.70,
            "breakout_window": 20,
            "breakout_tolerance": 0.0025,
            "pullback_tolerance": 0.0045,
            "volume_window": 30,
            "min_volume_ratio": 0.65,
            "take_profit_pct": 0.0040,
            "stop_loss_pct": 0.0030,
            "trailing_stop_pct": 0.0040,
            "crash_3bar": -0.008,
            "crash_6bar": -0.014,
            "min_position": 0.25,
            "max_position": 0.85,
            "averaging_enabled": True,
            "averaging_max_adds": 2,
            "averaging_min_loss_pct": 0.0025,
            "averaging_max_loss_pct": 0.0100,
            "averaging_step_weight": 0.20,
            "averaging_rsi_min": 28,
            "averaging_rsi_max": 58,
            "averaging_momentum_floor": -0.0025,
            "averaging_trend_floor": 0.993,
        }
    if profile == "pulse_10s":
        return {
            "label": "pulse",
            "buy_interval_seconds": 600,
            "min_exit_hold_seconds": 3,
            "take_profit_pct": 0.0030,
            "stop_loss_pct": 0.0004,
            "max_position": 1.0,
            "required_history_days": 1,
        }
    if profile == "rebate_micro":
        return {
            "label": "rebate micro",
            "buy_interval_seconds": 0,
            "min_exit_hold_seconds": 0,
            "take_profit_pct": 0.0030,
            "take_profit_min_pct": 0.0030,
            "take_profit_max_pct": 0.0050,
            "take_profit_vol_multiplier": 1.20,
            "stop_loss_pct": 0.0035,
            "max_position": 1.0,
            "required_history_days": 1,
            "add_while_holding": False,
            "entry_signal_required": True,
            "min_micro_momentum": 0.00002,
            "min_micro_volatility": 0.00002,
            "momentum_lookback": 3,
            "volatility_window": 8,
            "reversal_entry_enabled": True,
            "reversal_lookback": 8,
            "reversal_min_rebound": 0.00020,
            "breakout_tolerance": 0.0060,
            "min_reentry_seconds": 3,
            "positive_ev_required": True,
            "max_entry_spread_pct": 0.00045,
            "fee_round_trip_pct": 0.0008,
            "rebate_round_trip_pct": 0.0008,
            "slippage_buffer_pct": 0.0002,
            "min_expected_net_pct": 0.0,
            "min_take_profit_after_cost_pct": 0.0015,
            "min_reward_to_risk": 0.50,
        }
    if profile == "night_micro":
        return {
            "label": "night micro",
            "buy_interval_seconds": 0,
            "min_exit_hold_seconds": 0,
            "take_profit_pct": 0.0010,
            "take_profit_min_pct": 0.0010,
            "take_profit_max_pct": 0.0010,
            "take_profit_vol_multiplier": 1.0,
            "stop_loss_pct": 0.0070,
            "max_position": 1.0,
            "required_history_days": 1,
            "add_while_holding": False,
            "averaging_enabled": True,
            "averaging_trigger_pct": 0.0020,
            "averaging_second_trigger_pct": 0.0035,
            "averaging_order_krw": 20000,
            "averaging_max_adds": 2,
            "tick_mode_enabled": True,
            "take_profit_min_ticks": 10,
            "averaging_min_ticks": 10,
            "averaging_second_min_ticks": 18,
            "stop_loss_min_ticks": 30,
            "tick_band_min_10tick_pct": 0.0010,
            "tick_band_max_10tick_pct": 0.0060,
            "entry_signal_required": True,
            "min_micro_momentum": 0.00002,
            "min_micro_volatility": 0.00002,
            "momentum_lookback": 3,
            "volatility_window": 8,
            "reversal_entry_enabled": True,
            "reversal_lookback": 8,
            "reversal_min_rebound": 0.00020,
            "breakout_tolerance": 0.0060,
            "min_reentry_seconds": 3,
            "positive_ev_required": True,
            "max_entry_spread_pct": 0.00045,
            "fee_round_trip_pct": 0.0008,
            "rebate_round_trip_pct": 0.0008,
            "slippage_buffer_pct": 0.0002,
            "min_expected_net_pct": 0.0,
            "min_take_profit_after_cost_pct": 0.0005,
            "min_reward_to_risk": 0.15,
        }
    raise ValueError(f"지원하지 않는 profile: {profile}")


def build_strategy_for_profile(profile: str):
    """Build the strategy implementation mapped to a live profile."""
    params = strategy_params_for_profile(profile)
    if profile in SELECTABLE_MARKET_PROFILES:
        return PulseScalperStrategy(params=params)
    if profile in SCALP_PROFILES:
        return ETHMicroScalperStrategy(params=params)
    if profile in ADAPTIVE_ALPHA_PROFILES:
        return ETHAdaptiveAlphaStrategy(params=params)
    return ETHCashRegimeStrategy(params=params)


def default_timeframe_for_profile(profile: str) -> str:
    """Return the intended candle unit for a strategy profile."""
    return PROFILE_DEFAULT_TIMEFRAMES.get(profile, DEFAULT_TIMEFRAME)


def default_candle_count_for_profile(profile: str, fallback: int) -> int:
    """Return a profile-specific candle count when one is required."""
    return PROFILE_DEFAULT_CANDLE_COUNTS.get(profile, fallback)


def validate_profile_timeframe(profile: str, timeframe: str) -> None:
    """Validate profile/timeframe combinations."""
    expected = default_timeframe_for_profile(profile)
    if profile == "daily" and timeframe != "1d":
        raise ValueError("daily profile은 --timeframe 1d와 함께 사용해야 합니다.")
    if profile in ("intraday4h", "aggressive4h", "stable_quality") and timeframe != "4h":
        raise ValueError(f"{profile} profile은 --timeframe 4h와 함께 사용해야 합니다.")
    if profile in SCALP_PROFILES and timeframe != expected:
        raise ValueError(f"{profile} profile은 --timeframe {expected}와 함께 사용해야 합니다.")
    if profile in SELECTABLE_MARKET_PROFILES and timeframe != expected:
        raise ValueError(f"{profile} profile은 --timeframe {expected}와 함께 사용해야 합니다.")


def load_runtime_profile_mode(path: Path) -> Optional[str]:
    """Load a Discord-switchable live profile from JSON or plain text."""
    if not path.exists():
        return None
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return None
    try:
        raw = json.loads(raw_text)
        profile = raw.get("profile")
    except json.JSONDecodeError:
        profile = raw_text
    if profile not in RUNTIME_PROFILE_CHOICES:
        raise ValueError(f"지원하지 않는 런타임 profile: {profile}")
    return profile


def save_runtime_profile_mode(path: Path, profile: str, *, updated_by: str = "local") -> None:
    """Persist the Discord-switchable live profile."""
    if profile not in RUNTIME_PROFILE_CHOICES:
        raise ValueError(f"지원하지 않는 런타임 profile: {profile}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile,
        "timeframe": default_timeframe_for_profile(profile),
        "updated_by": updated_by,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_krw_market(market: str) -> str:
    """Normalize and validate a Bithumb KRW market code."""
    normalized = market.strip().upper()
    parts = normalized.split("-")
    if len(parts) != 2 or parts[0] != "KRW" or not parts[1].isalnum():
        raise ValueError(f"KRW 마켓 형식이어야 합니다. 예: KRW-SOL (입력: {market})")
    return normalized


def validate_profile_market(profile: str, market: str) -> None:
    """Validate whether a market can run with the selected profile."""
    normalized = normalize_krw_market(market)
    if profile not in SELECTABLE_MARKET_PROFILES and normalized != MARKET:
        raise ValueError(f"{profile} profile은 ETH 전용입니다. --market {MARKET}만 허용합니다.")


def load_runtime_market_mode(path: Path) -> Optional[str]:
    """Load the Discord-switchable Pulse KRW market from JSON or plain text."""
    if not path.exists():
        return None
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return None
    try:
        raw = json.loads(raw_text)
        market = raw.get("market")
    except json.JSONDecodeError:
        market = raw_text
    if not market:
        return None
    return normalize_krw_market(str(market))


def save_runtime_market_mode(path: Path, market: str, *, updated_by: str = "local") -> None:
    """Persist the Discord-switchable Pulse KRW market."""
    normalized = normalize_krw_market(market)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "market": normalized,
        "updated_by": updated_by,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def auto_trade_enabled(path: Path) -> bool:
    """Return whether start/stop controlled auto-trading is enabled."""
    return path.exists()


def enable_auto_trade(path: Path, *, profile: str, updated_by: str = "local") -> None:
    """Persist start/stop controlled auto-trading as enabled."""
    if profile not in START_STOP_PROFILES:
        raise ValueError(f"start/stop 자동거래 대상 profile이 아닙니다: {profile}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": True,
        "profile": profile,
        "updated_by": updated_by,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def disable_auto_trade(path: Path) -> None:
    """Disable start/stop controlled auto-trading."""
    if path.exists():
        path.unlink()


def enable_risk_override(
    path: Path,
    *,
    profile: str,
    market: str,
    updated_by: str = "local",
    minutes: int = 60,
) -> None:
    """Enable a time-limited Rebate Micro override for loss-limit gates only."""
    if profile not in MICRO_PROFILES:
        raise ValueError(f"risk override는 micro profile 전용입니다: {profile}")
    if minutes <= 0:
        raise ValueError("risk override minutes must be positive")
    now = datetime.now()
    payload = {
        "enabled": True,
        "profile": profile,
        "market": normalize_krw_market(market),
        "updated_by": updated_by,
        "updated_at": now.isoformat(timespec="seconds"),
        "expires_at": (now + timedelta(minutes=minutes)).isoformat(timespec="seconds"),
        "scope": "daily_weekly_mdd_only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def disable_risk_override(path: Path) -> None:
    """Disable the Rebate Micro loss-limit override."""
    if path.exists():
        path.unlink()


def risk_override_enabled(path: Path, *, profile: str, market: str, now: datetime) -> bool:
    """Return whether a micro loss-limit override is active."""
    if profile not in MICRO_PROFILES or not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw.get("enabled"):
            return False
        if raw.get("profile") != profile:
            return False
        override_market = raw.get("market")
        if override_market and normalize_krw_market(str(override_market)) != normalize_krw_market(market):
            return False
        expires_at = raw.get("expires_at")
        if expires_at and now >= datetime.fromisoformat(str(expires_at)):
            return False
        return True
    except Exception:
        return False


def risk_override_allows(report) -> bool:
    """Loss-limit override never bypasses the kill switch."""
    if report.kill_switch_active:
        return False
    return bool(report.daily_breached or report.weekly_breached or report.drawdown_breached)


def loss_limit_bypass_enabled(path: Path, *, profile: str, market: str, now: datetime) -> bool:
    """Return whether daily/weekly/MDD loss gates should be bypassed."""
    if profile in MICRO_PROFILES:
        return True
    return risk_override_enabled(path, profile=profile, market=market, now=now)


def load_rebate_turnover_progress(
    path: Path,
    *,
    target_krw: Decimal = Decimal("100000000000"),
) -> RebateTurnoverProgress:
    """Load persistent Rebate Micro turnover progress."""
    if not path.exists():
        return RebateTurnoverProgress(target_krw=target_krw)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RebateTurnoverProgress(
            total_krw=Decimal(str(raw.get("total_krw", "0"))),
            target_krw=Decimal(str(raw.get("target_krw", str(target_krw)))),
            orders=int(raw.get("orders", 0)),
            last_order_at=raw.get("last_order_at"),
            last_market=raw.get("last_market"),
        )
    except Exception:
        return RebateTurnoverProgress(target_krw=target_krw)


def save_rebate_turnover_progress(path: Path, progress: RebateTurnoverProgress) -> None:
    """Persist Rebate Micro turnover progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_krw": str(quantize_krw(progress.total_krw)),
        "target_krw": str(quantize_krw(progress.target_krw)),
        "orders": progress.orders,
        "last_order_at": progress.last_order_at,
        "last_market": progress.last_market,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def record_rebate_turnover(
    path: Path,
    *,
    notional_krw: Decimal,
    market: str,
    now: datetime,
    target_krw: Decimal = Decimal("100000000000"),
) -> RebateTurnoverProgress:
    """Add one Rebate Micro order's approximate notional to progress."""
    progress = load_rebate_turnover_progress(path, target_krw=target_krw)
    updated = RebateTurnoverProgress(
        total_krw=progress.total_krw + max(notional_krw, Decimal("0")),
        target_krw=target_krw,
        orders=progress.orders + 1,
        last_order_at=now.isoformat(timespec="seconds"),
        last_market=market,
    )
    save_rebate_turnover_progress(path, updated)
    return updated


def format_rebate_turnover_progress(progress: RebateTurnoverProgress) -> str:
    """Format turnover progress against the configured target."""
    pct = (
        Decimal("0") if progress.target_krw <= 0
        else progress.total_krw / progress.target_krw * Decimal("100")
    )
    return (
        f"{quantize_krw(progress.total_krw):,} / "
        f"{quantize_krw(progress.target_krw):,} KRW ({pct:.4f}%)"
    )


def estimate_order_notional(plan: PilotPlan, trade_price: Decimal) -> Decimal:
    """Estimate KRW turnover from a submitted market order plan."""
    if plan.krw_amount > 0:
        return plan.krw_amount
    if plan.volume > 0:
        return quantize_krw(plan.volume * trade_price)
    return Decimal("0")


def resolve_runtime_profile(
    args: argparse.Namespace,
    profile_mode_path: Path,
    market_mode_path: Optional[Path] = None,
) -> RuntimeProfile:
    """Resolve the strategy/timeframe for the current cycle."""
    profile = load_runtime_profile_mode(profile_mode_path)
    source = str(profile_mode_path) if profile else "service args"
    if profile is None:
        profile = args.profile
        timeframe = args.timeframe
    else:
        timeframe = default_timeframe_for_profile(profile)
    validate_profile_timeframe(profile, timeframe)
    market = normalize_krw_market(args.market)
    market_source = "service args"
    if profile in SELECTABLE_MARKET_PROFILES and market_mode_path is not None:
        runtime_market = load_runtime_market_mode(market_mode_path)
        if runtime_market:
            market = runtime_market
            market_source = str(market_mode_path)
    validate_profile_market(profile, market)
    candle_count = default_candle_count_for_profile(profile, args.candle_count)
    return RuntimeProfile(
        profile=profile,
        market=market,
        timeframe=timeframe,
        candle_count=candle_count,
        source=source,
        market_source=market_source,
    )


def load_pilot_state(path: Path, now: datetime) -> PilotState:
    """Load persisted pilot state, resetting counters on a new local date."""
    today = now.date().isoformat()
    if not path.exists():
        return PilotState(date=today)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    state = PilotState(
        date=str(raw.get("date", today)),
        orders_today=int(raw.get("orders_today", 0)),
        last_order_at=raw.get("last_order_at"),
        last_pulse_buy_at=raw.get("last_pulse_buy_at"),
        last_action=raw.get("last_action"),
        last_status_alert_at=raw.get("last_status_alert_at"),
        averaging_count=int(raw.get("averaging_count", 0)),
    )
    if state.date != today:
        return PilotState(date=today)
    return state


def save_pilot_state(path: Path, state: PilotState) -> None:
    """Persist pilot state for restart-safe cooldown and order limits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": state.date,
                "orders_today": state.orders_today,
                "last_order_at": state.last_order_at,
                "last_pulse_buy_at": state.last_pulse_buy_at,
                "last_action": state.last_action,
                "last_status_alert_at": state.last_status_alert_at,
                "averaging_count": state.averaging_count,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def assert_order_quota(
    *,
    state: PilotState,
    now: datetime,
    max_orders_per_day: int,
    cooldown_minutes: int,
    cooldown_seconds: Optional[int] = None,
    action: Optional[str] = None,
    allow_sell_during_cooldown: bool = False,
) -> None:
    """Reject a live order when daily count or cooldown would be violated."""
    if state.orders_today >= max_orders_per_day:
        raise RuntimeError(
            f"일일 주문 횟수 한도 도달: {state.orders_today}/{max_orders_per_day}"
        )
    if action == "SELL" and allow_sell_during_cooldown:
        return
    if not state.last_order_at:
        return
    last_order_at = datetime.fromisoformat(state.last_order_at)
    cooldown = timedelta(seconds=cooldown_seconds) if cooldown_seconds is not None else timedelta(minutes=cooldown_minutes)
    if now - last_order_at < cooldown:
        remaining = cooldown - (now - last_order_at)
        remaining_seconds = remaining.total_seconds()
        if remaining_seconds < 60:
            wait = f"{remaining_seconds:.0f}초"
        else:
            wait = f"{remaining_seconds/60:.1f}분"
        raise RuntimeError(
            f"쿨다운 중: 다음 주문까지 약 {wait}"
        )


def record_order(
    state: PilotState,
    now: datetime,
    *,
    action: Optional[str] = None,
    is_averaging: bool = False,
    is_pulse: bool = False,
) -> PilotState:
    """Return state after one accepted order."""
    today = now.date().isoformat()
    orders_today = state.orders_today if state.date == today else 0
    averaging_count = state.averaging_count if state.date == today else 0
    last_pulse_buy_at = state.last_pulse_buy_at if state.date == today else None
    if action == "SELL":
        averaging_count = 0
    elif action == "BUY" and is_averaging:
        averaging_count += 1
    if action == "BUY" and is_pulse:
        last_pulse_buy_at = now.isoformat(timespec="seconds")
    return PilotState(
        date=today,
        orders_today=orders_today + 1,
        last_order_at=now.isoformat(timespec="seconds"),
        last_pulse_buy_at=last_pulse_buy_at,
        last_action=action,
        last_status_alert_at=state.last_status_alert_at,
        averaging_count=averaging_count,
    )


def record_status_alert(state: PilotState, now: datetime) -> PilotState:
    """Return state after sending a scheduled Discord status alert."""
    today = now.date().isoformat()
    orders_today = state.orders_today if state.date == today else 0
    return PilotState(
        date=today,
        orders_today=orders_today,
        last_order_at=state.last_order_at,
        last_pulse_buy_at=state.last_pulse_buy_at if state.date == today else None,
        last_action=state.last_action if state.date == today else None,
        last_status_alert_at=now.isoformat(timespec="seconds"),
        averaging_count=state.averaging_count if state.date == today else 0,
    )


def parse_decimal(value: str) -> Decimal:
    """Parse a positive Decimal from CLI text."""
    parsed = Decimal(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 값이어야 합니다.")
    return parsed


def market_base_currency(market: str) -> str:
    """Return base currency from a KRW market string such as KRW-ETH."""
    parts = market.split("-")
    if len(parts) != 2 or parts[0] != "KRW":
        raise ValueError(f"KRW 마켓만 지원합니다: {market}")
    return parts[1]


def candles_to_ohlcv(candles: Iterable[Candle]) -> pd.DataFrame:
    """Convert Bithumb candles to the strategy OHLCV DataFrame shape."""
    rows = []
    for candle in candles:
        timestamp = candle.candle_datetime_kst or candle.candle_datetime_utc
        if timestamp is None:
            continue
        rows.append({
            "timestamp": pd.Timestamp(timestamp).tz_localize(None),
            "open": float(candle.opening_price),
            "high": float(candle.high_price),
            "low": float(candle.low_price),
            "close": float(candle.trade_price),
            "volume": float(candle.candle_acc_trade_volume),
        })
    if not rows:
        raise ValueError("캔들 데이터가 비어 있습니다.")

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    required = ["open", "high", "low", "close", "volume"]
    if df[required].isnull().any().any():
        raise ValueError("캔들 데이터에 NaN이 포함되어 있습니다.")
    return df[required]


def validate_live_permission(config: Config, live: bool, confirmed: bool) -> None:
    """Reject real order mode unless every live-trading gate is explicit."""
    if not live:
        return
    if not confirmed:
        raise ValueError("--live 사용 시 --yes-i-understand-risk 확인 플래그가 필요합니다.")
    if not config.is_live_trading:
        raise ValueError(".env TRADING_MODE=LIVE 상태에서만 --live 주문을 허용합니다.")


def quantize_krw(amount: Decimal) -> Decimal:
    """Round down KRW amounts to whole won."""
    return amount.quantize(KRW_QUANT, rounding=ROUND_DOWN)


def quantize_volume(volume: Decimal) -> Decimal:
    """Round down coin volume to a conservative precision."""
    return volume.quantize(VOLUME_QUANT, rounding=ROUND_DOWN)


def krw_tick_size(price: Decimal) -> Decimal:
    """Return the current Bithumb KRW-market tick size for a price."""
    if price < Decimal("1"):
        return Decimal("0.0001")
    if price < Decimal("10"):
        return Decimal("0.001")
    if price < Decimal("100"):
        return Decimal("0.01")
    if price < Decimal("5000"):
        return Decimal("1")
    if price < Decimal("10000"):
        return Decimal("5")
    if price < Decimal("50000"):
        return Decimal("10")
    if price < Decimal("100000"):
        return Decimal("50")
    if price < Decimal("500000"):
        return Decimal("100")
    if price < Decimal("1000000"):
        return Decimal("500")
    return Decimal("1000")


def quantize_price_to_tick(price: Decimal, *, side: str) -> Decimal:
    """Round a KRW price to a valid tick without worsening side safety."""
    tick = krw_tick_size(price)
    units = price / tick
    if side == "ask":
        rounded_units = units.to_integral_value(rounding=ROUND_CEILING)
    else:
        rounded_units = units.to_integral_value(rounding=ROUND_DOWN)
    return max(rounded_units * tick, tick)


def rebate_limit_buy_price(
    best_bid: Decimal,
    *,
    offset_ticks: int = 1,
    max_offset_pct: Decimal = Decimal("0.0005"),
) -> Decimal:
    """Place buy limit slightly below best bid when one tick is still tiny."""
    if best_bid <= 0:
        raise ValueError("best_bid가 0보다 커야 합니다.")
    tick = krw_tick_size(best_bid)
    offset = tick * Decimal(offset_ticks)
    if offset / best_bid > max_offset_pct:
        offset = Decimal("0")
    return quantize_price_to_tick(best_bid - offset, side="bid")


def rebate_take_profit_limit_price(
    *,
    best_ask: Decimal,
    avg_buy_price: Decimal,
    take_profit_pct: Optional[float],
) -> Decimal:
    """Place take-profit sell limit at the better of target price and best ask."""
    if best_ask <= 0:
        raise ValueError("best_ask가 0보다 커야 합니다.")
    target = best_ask
    if avg_buy_price > 0 and take_profit_pct is not None:
        target = max(
            best_ask,
            avg_buy_price * (Decimal("1") + Decimal(str(take_profit_pct))),
        )
    return quantize_price_to_tick(target, side="ask")


def calculate_buy_amount(
    *,
    target_weight: float,
    portfolio_value_krw: Decimal,
    current_position_krw: Decimal,
    krw_balance: Decimal,
    max_order_krw: Decimal,
    max_position_krw: Decimal,
    min_order_krw: Decimal,
) -> Decimal:
    """Calculate the guarded KRW market-buy amount."""
    target_position_krw = quantize_krw(
        portfolio_value_krw * Decimal(str(target_weight))
    )
    desired_add = target_position_krw - current_position_krw
    remaining_position_cap = max_position_krw - current_position_krw
    amount = min(desired_add, remaining_position_cap, max_order_krw, krw_balance)
    amount = quantize_krw(max(amount, Decimal("0")))
    if amount < min_order_krw:
        return Decimal("0")
    return amount


def calculate_sell_volume(
    *,
    coin_balance: Decimal,
    trade_price: Decimal,
    min_order_krw: Decimal,
) -> Decimal:
    """Calculate the guarded market-sell volume."""
    estimated_value = coin_balance * trade_price
    if estimated_value < min_order_krw:
        return Decimal("0")
    return quantize_volume(coin_balance)


def build_strategy_context(
    *,
    market: str,
    ohlcv: pd.DataFrame,
    balances: Balances,
    trade_price: Decimal,
    metadata: Optional[dict] = None,
) -> StrategyContext:
    """Build a StrategyContext from live balances and recent candles."""
    base = market_base_currency(market)
    coin_balance = balances.get_total(base)
    krw_balance = balances.krw_balance
    position_value = coin_balance * trade_price
    portfolio_value = krw_balance + position_value
    account = balances.get(base)
    avg_buy_price = float(account.avg_buy_price) if account else 0.0

    return StrategyContext(
        timestamp=ohlcv.index[-1],
        ticker=market,
        ohlcv=ohlcv,
        current_position=float(coin_balance),
        avg_buy_price=avg_buy_price,
        cash=float(krw_balance),
        portfolio_value=float(portfolio_value),
        metadata=metadata or {},
    )


def calculate_portfolio_equity(
    *,
    market: str,
    balances: Balances,
    trade_price: Decimal,
) -> Decimal:
    """Return KRW cash plus live mark-to-market value for the traded coin."""
    base = market_base_currency(market)
    coin_balance = balances.get_total(base)
    return balances.krw_balance + (coin_balance * trade_price)


def build_pilot_plan(
    *,
    decision: TradeDecision,
    balances: Balances,
    trade_price: Decimal,
    market: str,
    max_order_krw: Decimal,
    max_position_krw: Decimal,
    min_order_krw: Decimal,
    orderbook: Optional[Orderbook] = None,
    use_limit_orders: bool = False,
    limit_wait_seconds: int = 0,
    dry_run: bool = True,
) -> PilotPlan:
    """Turn a strategy decision into one guarded pilot order plan."""
    base = market_base_currency(market)
    coin_balance = balances.get_total(base)
    krw_balance = balances.krw_balance
    current_position_krw = coin_balance * trade_price
    portfolio_value_krw = krw_balance + current_position_krw

    if decision.signal == Signal.BUY:
        if decision.order_krw is not None:
            requested_order = quantize_krw(Decimal(str(decision.order_krw)))
            remaining_position_cap = max_position_krw - current_position_krw
            krw_amount = min(requested_order, remaining_position_cap, max_order_krw, krw_balance)
            krw_amount = quantize_krw(max(krw_amount, Decimal("0")))
            if krw_amount < min_order_krw:
                krw_amount = Decimal("0")
        else:
            krw_amount = calculate_buy_amount(
                target_weight=decision.target_weight,
                portfolio_value_krw=portfolio_value_krw,
                current_position_krw=current_position_krw,
                krw_balance=krw_balance,
                max_order_krw=max_order_krw,
                max_position_krw=max_position_krw,
                min_order_krw=min_order_krw,
            )
        if krw_amount <= 0:
            if "controlled averaging" in decision.reason:
                sell_volume = calculate_sell_volume(
                    coin_balance=coin_balance,
                    trade_price=trade_price,
                    min_order_krw=min_order_krw,
                )
                if sell_volume > 0:
                    return PilotPlan(
                        action="SELL",
                        reason=(
                            "averaging order blocked; sell all and stop auto trade | "
                            f"{decision.reason}"
                        ),
                        volume=sell_volume,
                        dry_run=dry_run,
                    )
            return PilotPlan(
                action="SKIP",
                reason=(
                    "매수 신호는 있지만 최소 주문금액/현금/포지션 한도 때문에 주문하지 않음 | "
                    f"{decision.reason}"
                ),
                dry_run=dry_run,
            )
        if use_limit_orders and orderbook and orderbook.best_bid:
            limit_price = rebate_limit_buy_price(orderbook.best_bid)
            limit_volume = quantize_volume(krw_amount / limit_price)
            limit_notional = quantize_krw(limit_volume * limit_price)
            if limit_notional < min_order_krw:
                return PilotPlan(
                    action="SKIP",
                    reason=(
                        "지정가 매수 신호는 있지만 호가단위 반영 후 최소 주문금액 미만 | "
                        f"{decision.reason}"
                    ),
                    dry_run=dry_run,
                )
            return PilotPlan(
                action="BUY",
                reason=decision.reason,
                krw_amount=limit_notional,
                volume=limit_volume,
                order_type="limit",
                limit_price=limit_price,
                limit_wait_seconds=limit_wait_seconds,
                dry_run=dry_run,
            )
        return PilotPlan(
            action="BUY",
            reason=decision.reason,
            krw_amount=krw_amount,
            dry_run=dry_run,
        )

    if decision.signal == Signal.SELL:
        volume = calculate_sell_volume(
            coin_balance=coin_balance,
            trade_price=trade_price,
            min_order_krw=min_order_krw,
        )
        if volume <= 0:
            return PilotPlan(
                action="SKIP",
                reason=f"매도 신호는 있지만 보유 ETH 평가액이 최소 주문금액 미만 | {decision.reason}",
                dry_run=dry_run,
            )
        if (
            use_limit_orders and
            orderbook and
            orderbook.best_ask and
            "take-profit" in decision.reason
        ):
            account = balances.get(base)
            avg_buy_price = account.avg_buy_price if account else Decimal("0")
            limit_price = rebate_take_profit_limit_price(
                best_ask=orderbook.best_ask,
                avg_buy_price=avg_buy_price,
                take_profit_pct=decision.take_profit_pct,
            )
            return PilotPlan(
                action="SELL",
                reason=decision.reason,
                volume=volume,
                order_type="limit",
                limit_price=limit_price,
                limit_wait_seconds=limit_wait_seconds,
                dry_run=dry_run,
            )
        return PilotPlan(
            action="SELL",
            reason=decision.reason,
            volume=volume,
            dry_run=dry_run,
        )

    return PilotPlan(action="HOLD", reason=decision.reason, dry_run=dry_run)


def assert_spread_ok(spread_pct: Optional[Decimal], max_spread_pct: Decimal) -> None:
    """Reject trading when the orderbook spread is too wide."""
    if spread_pct is None:
        raise ValueError("호가 스프레드를 계산할 수 없습니다.")
    if spread_pct > max_spread_pct:
        raise ValueError(
            f"호가 스프레드 과다: {spread_pct * Decimal('100'):.3f}% "
            f"> 허용 {max_spread_pct * Decimal('100'):.3f}%"
        )


def wait_for_limit_order_fill(
    client: BithumbClient,
    order: Order,
    *,
    wait_seconds: int,
) -> Optional[Order]:
    """Wait briefly for a limit order, then cancel any unfilled remainder."""
    if client.dry_run or wait_seconds <= 0:
        return order

    deadline = time.monotonic() + wait_seconds
    latest = order
    while time.monotonic() < deadline:
        latest = client.get_order(order.uuid)
        if latest.is_filled:
            return latest
        if not latest.is_open:
            return latest if latest.executed_volume > 0 else None
        time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))

    latest = client.get_order(order.uuid)
    if latest.is_filled:
        return latest
    if latest.is_open:
        print(f"지정가 미체결 잔량 취소: uuid={order.uuid}")
        client.cancel_order(order.uuid)
        latest = client.get_order(order.uuid)
    if latest.executed_volume > 0:
        return latest
    print(f"지정가 {wait_seconds}초 미체결로 주문 취소 완료: uuid={order.uuid}")
    return None


def execute_plan(client: BithumbClient, market: str, plan: PilotPlan) -> Optional[Order]:
    """Execute one market or short-lived limit order for the pilot plan."""
    if plan.order_type == "limit":
        if plan.limit_price <= 0 or plan.volume <= 0:
            raise ValueError("지정가 주문에는 limit_price와 volume이 필요합니다.")
        side = OrderSide.BID if plan.action == "BUY" else OrderSide.ASK
        print(
            f"지정가 주문 제출: action={plan.action} price={plan.limit_price} "
            f"volume={plan.volume} wait={plan.limit_wait_seconds}s"
        )
        order = client.limit_order(
            market=market,
            side=side,
            price=plan.limit_price,
            volume=plan.volume,
        )
        return wait_for_limit_order_fill(
            client,
            order,
            wait_seconds=plan.limit_wait_seconds,
        )

    if plan.action == "BUY":
        return client.market_buy(market, plan.krw_amount)
    if plan.action == "SELL":
        return client.market_sell(market, plan.volume)
    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run one guarded ETH/Cash micro-live cycle.")
    parser.add_argument("--market", default=MARKET, help="KRW market. Default: KRW-ETH")
    parser.add_argument(
        "--timeframe",
        choices=["1m", "3m", "5m", "10m", "15m", "30m", "60m", "1h", "4h", "1d"],
        default=DEFAULT_TIMEFRAME,
    )
    parser.add_argument(
        "--profile",
        choices=[
            "intraday4h",
            "aggressive4h",
            "daily",
            "stable_quality",
            "scalp_active",
            "scalp_1m",
            "pulse_10s",
            "rebate_micro",
            "night_micro",
        ],
        default=DEFAULT_PROFILE,
    )
    parser.add_argument("--candle-count", type=int, default=DEFAULT_CANDLE_COUNT)
    parser.add_argument("--max-order-krw", type=parse_decimal, default=Decimal("10000"))
    parser.add_argument("--max-position-krw", type=parse_decimal, default=Decimal("30000"))
    parser.add_argument("--min-order-krw", type=parse_decimal, default=Decimal("5000"))
    parser.add_argument("--max-spread-pct", type=parse_decimal, default=Decimal("0.005"))
    parser.add_argument("--loop", action="store_true", help="Keep running guarded decision cycles.")
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=None,
        help="Override loop interval in seconds. Use for ticker-level pulse monitoring.",
    )
    parser.add_argument(
        "--align-interval-to-clock",
        action="store_true",
        help="Sleep to the next wall-clock interval boundary, e.g. 00/15/30/45.",
    )
    parser.add_argument("--cooldown-minutes", type=int, default=240)
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=None,
        help="Override cooldown in seconds. Use carefully for high-frequency pilots.",
    )
    parser.add_argument("--max-orders-per-day", type=int, default=4)
    parser.add_argument(
        "--status-alert-interval-minutes",
        type=int,
        default=0,
        help="Send Discord price/account/order status every N minutes. 0 disables.",
    )
    parser.add_argument(
        "--status-alert-on-hour",
        action="store_true",
        help="Only send scheduled status alerts on wall-clock interval boundaries.",
    )
    parser.add_argument(
        "--state-file",
        default="logs/eth_cash_micro_live_state.json",
        help="Persistent order counter/cooldown file.",
    )
    parser.add_argument(
        "--risk-state-file",
        default="logs/eth_cash_micro_live_risk_state.json",
        help="Persistent account risk baseline file.",
    )
    parser.add_argument(
        "--kill-switch-file",
        default="logs/KILL_SWITCH_ACTIVE",
        help="If this file exists, all BUY/SELL orders are blocked.",
    )
    parser.add_argument(
        "--profile-mode-file",
        default="logs/eth_cash_live_profile.json",
        help="Optional runtime profile file. Discord can switch stable/scalp by writing here.",
    )
    parser.add_argument(
        "--market-mode-file",
        default="logs/pulse_market.json",
        help="Optional runtime market file for pulse_10s. Discord can switch KRW alt markets here.",
    )
    parser.add_argument(
        "--auto-trade-file",
        default="logs/AUTO_TRADE_ENABLED",
        help="Start/stop flag for start-button controlled micro strategies.",
    )
    parser.add_argument(
        "--force-start-file",
        default="logs/REBATE_MICRO_FORCE_START.json",
        help="Time-limited Rebate Micro override for daily/weekly/MDD loss-limit gates.",
    )
    parser.add_argument(
        "--turnover-file",
        default="logs/rebate_micro_turnover.json",
        help="Persistent Rebate Micro turnover progress file.",
    )
    parser.add_argument(
        "--turnover-target-krw",
        type=parse_decimal,
        default=Decimal("100000000000"),
        help="Turnover progress target for Rebate Micro reporting.",
    )
    parser.add_argument("--live", action="store_true", help="Send real orders. Requires TRADING_MODE=LIVE.")
    parser.add_argument("--yes-i-understand-risk", action="store_true")
    parser.add_argument("--notify-dry-run", action="store_true", help="Send Discord alerts for dry-run orders/errors.")
    parser.add_argument("--no-file-log", action="store_true", help="Disable file logging for local permission-limited runs.")
    return parser.parse_args()


def validate_runner_args(args: argparse.Namespace) -> None:
    """Validate runtime argument combinations."""
    args.market = normalize_krw_market(args.market)
    validate_profile_market(args.profile, args.market)
    validate_profile_timeframe(args.profile, args.timeframe)
    if args.interval_minutes <= 0:
        raise ValueError("--interval-minutes는 1 이상이어야 합니다.")
    if args.interval_seconds is not None and args.interval_seconds <= 0:
        raise ValueError("--interval-seconds는 1 이상이어야 합니다.")
    if args.interval_seconds is not None and args.align_interval_to_clock:
        raise ValueError("--interval-seconds와 --align-interval-to-clock은 함께 사용할 수 없습니다.")
    if args.align_interval_to_clock and 1440 % args.interval_minutes != 0:
        raise ValueError("--align-interval-to-clock은 하루를 나누는 interval에서만 사용하세요.")
    if args.cooldown_minutes < 0:
        raise ValueError("--cooldown-minutes는 0 이상이어야 합니다.")
    if args.cooldown_seconds is not None and args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds는 0 이상이어야 합니다.")
    if args.max_orders_per_day <= 0:
        raise ValueError("--max-orders-per-day는 1 이상이어야 합니다.")
    if args.status_alert_interval_minutes < 0:
        raise ValueError("--status-alert-interval-minutes는 0 이상이어야 합니다.")
    if args.status_alert_on_hour:
        if args.status_alert_interval_minutes <= 0:
            raise ValueError("--status-alert-on-hour는 status interval이 필요합니다.")
        if 1440 % args.status_alert_interval_minutes != 0:
            raise ValueError("--status-alert-on-hour interval은 하루를 나누는 값이어야 합니다.")


def print_banner(args: argparse.Namespace, dry_run: bool) -> None:
    """Print runner settings at startup."""
    print("=" * 72)
    print("ETH/Cash Micro Live Runner")
    print("=" * 72)
    print(
        f"mode={'DRY-RUN' if dry_run else 'LIVE'} market={args.market} "
        f"timeframe={args.timeframe} profile={args.profile}"
    )
    print(
        f"max_order={args.max_order_krw:,} KRW "
        f"max_position={args.max_position_krw:,} KRW "
        f"min_order={args.min_order_krw:,} KRW"
    )
    print(
        f"loop={args.loop} interval={format_interval(args.interval_minutes, args.interval_seconds)} "
        f"cooldown={format_cooldown(args.cooldown_minutes, args.cooldown_seconds)} "
        f"max_orders/day={args.max_orders_per_day}"
    )
    clock_mode = " clock" if args.status_alert_on_hour else ""
    print(f"discord_status_interval={args.status_alert_interval_minutes}m{clock_mode}")
    print(f"sleep_mode={'clock-aligned' if args.align_interval_to_clock else 'fixed'}")
    print(f"risk_state={args.risk_state_file} kill_switch={args.kill_switch_file}")
    print(f"profile_mode_file={args.profile_mode_file}")
    print(f"market_mode_file={args.market_mode_file}")
    print(f"auto_trade_file={args.auto_trade_file}")
    print(f"force_start_file={args.force_start_file}")
    print(f"turnover_file={args.turnover_file} target={args.turnover_target_krw:,} KRW")


def notify_safely(
    *,
    notifier: DiscordNotifier,
    embed: DiscordEmbed,
    dry_run: bool,
    notify_dry_run: bool,
) -> None:
    """Send a Discord alert without letting alert failures affect trading flow."""
    if dry_run and not notify_dry_run:
        return
    if not notifier.enabled:
        return
    try:
        notifier.send(embeds=(embed,))
    except Exception as exc:
        print(f"[알림 오류] Discord 전송 실패: {type(exc).__name__}")


def build_order_alert_embed(
    *,
    plan: PilotPlan,
    order: Order,
    market: str,
    equity: Decimal,
    dry_run: bool,
) -> DiscordEmbed:
    """Build a Discord order alert."""
    mode = "DRY-RUN" if dry_run else "LIVE"
    color = COLOR_INFO if dry_run else COLOR_SUCCESS
    fields = [
        DiscordField("Mode", mode),
        DiscordField("Market", market),
        DiscordField("Action", plan.action),
        DiscordField("Order Type", plan.order_type),
        DiscordField("State", order.state.value),
        DiscordField("Equity", f"{quantize_krw(equity):,} KRW"),
        DiscordField("UUID", order.uuid, inline=False),
    ]
    if plan.limit_price > 0:
        fields.append(DiscordField("Limit Price", f"{plan.limit_price:,} KRW"))
    if plan.krw_amount > 0:
        fields.append(DiscordField("KRW Amount", f"{plan.krw_amount:,} KRW"))
    if plan.volume > 0:
        fields.append(DiscordField("Volume", str(plan.volume)))
    return DiscordEmbed(
        title=f"{mode} order requested",
        description=plan.reason,
        color=color,
        fields=tuple(fields),
    )


def build_risk_block_alert_embed(
    *,
    action: str,
    error: Exception,
    equity: Decimal,
    dry_run: bool,
) -> DiscordEmbed:
    """Build a Discord alert for a risk-gate block."""
    mode = "DRY-RUN" if dry_run else "LIVE"
    return DiscordEmbed(
        title=f"{mode} risk gate blocked {action}",
        description=str(error),
        color=COLOR_DANGER,
        fields=(
            DiscordField("Equity", f"{quantize_krw(equity):,} KRW"),
            DiscordField("Action", action),
        ),
    )


def build_error_alert_embed(*, error: Exception, dry_run: bool) -> DiscordEmbed:
    """Build a Discord alert for runtime errors."""
    mode = "DRY-RUN" if dry_run else "LIVE"
    color = COLOR_WARNING if dry_run else COLOR_DANGER
    return DiscordEmbed(
        title=f"{mode} runner error",
        description=f"{type(error).__name__}: {error}",
        color=color,
    )


def status_alert_due(state: PilotState, now: datetime, interval_minutes: int) -> bool:
    """Return True when a scheduled Discord status alert should be sent."""
    if interval_minutes <= 0:
        return False
    if not state.last_status_alert_at:
        return True
    last_sent = datetime.fromisoformat(state.last_status_alert_at)
    return now - last_sent >= timedelta(minutes=interval_minutes)


def clock_status_alert_due(
    state: PilotState,
    now: datetime,
    interval_minutes: int,
) -> bool:
    """Return True only on wall-clock status boundaries.

    For interval=60 this means one alert per hour, on the cycle that starts in
    minute 00. A restart during the same hour will not duplicate the alert.
    """
    if interval_minutes <= 0:
        return False
    minutes_since_midnight = now.hour * 60 + now.minute
    if minutes_since_midnight % interval_minutes != 0:
        return False

    boundary = now.replace(second=0, microsecond=0)
    if not state.last_status_alert_at:
        return True

    last_sent = datetime.fromisoformat(state.last_status_alert_at)
    return last_sent < boundary


def calculate_sleep_seconds(
    *,
    now: datetime,
    interval_minutes: int,
    align_to_clock: bool,
    interval_seconds: Optional[int] = None,
) -> int:
    """Calculate loop sleep seconds, optionally to the next clock boundary."""
    if interval_seconds is not None:
        return interval_seconds
    if not align_to_clock:
        return interval_minutes * 60

    interval_seconds = interval_minutes * 60
    seconds_since_midnight = (
        now.hour * 3600 +
        now.minute * 60 +
        now.second
    )
    next_boundary = (
        (seconds_since_midnight // interval_seconds) + 1
    ) * interval_seconds
    if next_boundary > 24 * 3600:
        next_boundary = 24 * 3600
    return max(next_boundary - seconds_since_midnight, 1)


def format_interval(interval_minutes: int, interval_seconds: Optional[int]) -> str:
    """Format loop interval for logs."""
    if interval_seconds is not None:
        if interval_seconds < 60:
            return f"{interval_seconds}s"
        if interval_seconds % 60 == 0:
            return f"{interval_seconds // 60}m"
        return f"{interval_seconds / 60:.1f}m"
    return f"{interval_minutes}m"


def format_cooldown(cooldown_minutes: int, cooldown_seconds: Optional[int]) -> str:
    """Format order cooldown for logs."""
    if cooldown_seconds is not None:
        if cooldown_seconds < 60:
            return f"{cooldown_seconds}s"
        if cooldown_seconds % 60 == 0:
            return f"{cooldown_seconds // 60}m"
        return f"{cooldown_seconds / 60:.1f}m"
    return f"{cooldown_minutes}m"


def format_sleep_wait(seconds: int) -> str:
    """Format loop sleep duration for logs."""
    if seconds < 60:
        return f"{seconds}초"
    minutes = seconds / 60
    if minutes.is_integer():
        return f"{int(minutes)}분"
    return f"{minutes:.1f}분"


def build_risk_summary(risk_report, risk_limits: RiskLimits) -> str:
    """Format risk state into one Discord field."""
    return (
        f"daily {format_pct(risk_report.daily_loss)}/{format_pct(risk_limits.daily_loss_limit)} | "
        f"weekly {format_pct(risk_report.weekly_loss)}/{format_pct(risk_limits.weekly_loss_limit)} | "
        f"MDD {format_pct(risk_report.drawdown)}/{format_pct(risk_limits.max_drawdown_limit)}"
    )


def send_status_alert_if_due(
    *,
    client: BithumbClient,
    notifier: DiscordNotifier,
    state: PilotState,
    now: datetime,
    args: argparse.Namespace,
    market: str,
    ticker,
    orderbook,
    balances: Balances,
    open_orders: list[Order],
    decision: TradeDecision,
    risk_report,
    risk_limits: RiskLimits,
) -> PilotState:
    """Send scheduled Discord price/account/order status without blocking trading."""
    if not notifier.enabled:
        return state
    if args.status_alert_on_hour:
        due = clock_status_alert_due(
            state,
            now,
            args.status_alert_interval_minutes,
        )
    else:
        due = status_alert_due(state, now, args.status_alert_interval_minutes)
    if not due:
        return state

    try:
        prices_by_currency = fetch_krw_prices_for_balances(client, balances)
        filled_orders = tuple(
            client.get_orders(market=market, state="done", limit=5)
        )
        canceled_orders = tuple(
            client.get_orders(market=market, state="cancel", limit=5)
        )
        snapshot = AccountSnapshot(
            market=market,
            ticker=ticker,
            orderbook=orderbook,
            balances=balances,
            open_orders=tuple(open_orders),
            filled_orders=filled_orders,
            canceled_orders=canceled_orders,
            prices_by_currency=prices_by_currency,
        )
        notifier.send(
            embeds=build_snapshot_embeds(
                snapshot,
                title_prefix=(
                    "Hourly LIVE status"
                    if args.status_alert_on_hour
                    else f"{args.status_alert_interval_minutes}m LIVE status"
                ),
                orders_today=state.orders_today,
                max_orders_per_day=args.max_orders_per_day,
                signal=decision.signal.name,
                target_weight=decision.target_weight,
                risk_summary=build_risk_summary(risk_report, risk_limits),
                kill_switch_active=risk_report.kill_switch_active,
                timestamp=now,
            )
        )
        if args.status_alert_on_hour:
            print("\nDiscord 정각 상태 알림 전송 완료")
        else:
            print(f"\nDiscord {args.status_alert_interval_minutes}분 상태 알림 전송 완료")
        return record_status_alert(state, now)
    except Exception as exc:
        print(f"\n[알림 오류] Discord 상태 알림 실패: {type(exc).__name__}")
        return state


def run_cycle(
    *,
    client: BithumbClient,
    args: argparse.Namespace,
    dry_run: bool,
    state_path: Path,
    risk_state_path: Path,
    kill_switch_path: Path,
    risk_limits: RiskLimits,
    notifier: DiscordNotifier,
) -> Optional[Order]:
    """Run one guarded decision cycle and optionally place one order."""
    now = datetime.now()
    state = load_pilot_state(state_path, now)
    runtime_profile = resolve_runtime_profile(
        args,
        Path(args.profile_mode_file),
        Path(args.market_mode_file),
    )

    strategy = build_strategy_for_profile(runtime_profile.profile)
    if runtime_profile.candle_count < strategy.warmup_period() + 5:
        raise ValueError("candle-count가 전략 warmup보다 작습니다.")

    if not client.health_check():
        raise RuntimeError("빗썸 public API 헬스체크 실패")

    market = runtime_profile.market
    open_orders = client.get_open_orders(market)
    if open_orders:
        raise RuntimeError(f"미체결 주문 {len(open_orders)}개 존재. 새 주문을 차단합니다.")

    candles = client.get_candles(
        market,
        unit=runtime_profile.timeframe,
        count=runtime_profile.candle_count,
    )
    ohlcv = candles_to_ohlcv(candles)
    ticker = client.get_ticker(market)
    balances = client.get_balances()
    orderbook = client.get_orderbook(market)
    assert_spread_ok(orderbook.spread_pct, args.max_spread_pct)
    base = market_base_currency(market)
    coin_total = balances.get_total(base)
    coin_value = coin_total * ticker.trade_price
    if coin_value < args.min_order_krw and state.averaging_count:
        state = PilotState(
            date=state.date,
            orders_today=state.orders_today,
            last_order_at=state.last_order_at,
            last_pulse_buy_at=state.last_pulse_buy_at,
            last_action=state.last_action,
            last_status_alert_at=state.last_status_alert_at,
            averaging_count=0,
        )

    portfolio_equity = calculate_portfolio_equity(
        market=market,
        balances=balances,
        trade_price=ticker.trade_price,
    )
    risk_state = load_risk_state(risk_state_path, now, portfolio_equity)
    risk_report = evaluate_risk(
        state=risk_state,
        equity=portfolio_equity,
        limits=risk_limits,
        kill_switch_file_active=kill_switch_path.exists(),
    )
    loss_limit_bypass = loss_limit_bypass_enabled(
        Path(args.force_start_file),
        profile=runtime_profile.profile,
        market=market,
        now=now,
    )

    ctx = build_strategy_context(
        market=market,
        ohlcv=ohlcv,
        balances=balances,
        trade_price=ticker.trade_price,
        metadata={
            "profile": runtime_profile.profile,
            "averaging_count": state.averaging_count,
            "last_pulse_buy_at": state.last_pulse_buy_at,
            "last_order_at": state.last_order_at,
            "last_action": state.last_action,
            "now": now.isoformat(timespec="seconds"),
            "trade_price": float(ticker.trade_price),
            "spread_pct": float(orderbook.spread_pct),
            "tick_size_krw": float(krw_tick_size(ticker.trade_price)),
            "min_order_krw": float(args.min_order_krw),
            "max_order_krw": float(args.max_order_krw),
            "max_position_krw": float(args.max_position_krw),
        },
    )
    start_stop_required = runtime_profile.profile in START_STOP_PROFILES
    start_stop_enabled = (
        not start_stop_required or
        auto_trade_enabled(Path(args.auto_trade_file))
    )
    if start_stop_enabled:
        decision = strategy.generate_signal(ctx)
    else:
        decision = TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason=(
                f"{runtime_profile.profile} stopped by Discord Start/Stop control | "
                f"create {args.auto_trade_file} to enable"
            ),
            confidence=0.0,
        )
    plan = build_pilot_plan(
        decision=decision,
        balances=balances,
        trade_price=ticker.trade_price,
        market=market,
        max_order_krw=args.max_order_krw,
        max_position_krw=args.max_position_krw,
        min_order_krw=args.min_order_krw,
        orderbook=orderbook,
        use_limit_orders=runtime_profile.profile == "night_micro",
        limit_wait_seconds=7,
        dry_run=dry_run,
    )

    print(f"\n[{now.isoformat(timespec='seconds')}]")
    print("[프로필]")
    print(
        f"profile={runtime_profile.profile} market={market} "
        f"timeframe={runtime_profile.timeframe} "
        f"candles={runtime_profile.candle_count} source={runtime_profile.source}"
    )
    if runtime_profile.market_source != "service args":
        print(f"market_source={runtime_profile.market_source}")
    print("[상태]")
    print(f"KRW 가용: {balances.krw_balance:,}")
    print(f"{base} 총량: {coin_total}")
    print(f"{base} 평가액: {quantize_krw(coin_value):,} KRW")
    print(f"현재가: {ticker.trade_price:,} KRW")
    print(f"스프레드: {orderbook.spread_pct * Decimal('100'):.4f}%")
    print(f"오늘 주문수: {state.orders_today}/{args.max_orders_per_day}")
    print(f"추매 카운트: {state.averaging_count}")
    if runtime_profile.profile in SELECTABLE_MARKET_PROFILES:
        print(f"마지막 micro 매수: {state.last_pulse_buy_at or '없음'}")
        print(f"마지막 주문 방향: {state.last_action or '없음'}")
    if start_stop_required:
        print(f"자동거래: {'enabled' if start_stop_enabled else 'stopped'}")
    if runtime_profile.profile in MICRO_PROFILES:
        progress = load_rebate_turnover_progress(
            Path(args.turnover_file),
            target_krw=args.turnover_target_krw,
        )
        print(f"거래대금 진행률: {format_rebate_turnover_progress(progress)}")
        print(f"손실률 게이트: {'bypassed' if loss_limit_bypass else 'active'}")

    print("\n[위험 게이트]")
    print(f"계좌 평가액: {quantize_krw(risk_report.equity):,} KRW")
    print(
        f"일일 손실: {format_pct(risk_report.daily_loss)} "
        f"/ 한도 {format_pct(risk_limits.daily_loss_limit)}"
    )
    print(
        f"주간 손실: {format_pct(risk_report.weekly_loss)} "
        f"/ 한도 {format_pct(risk_limits.weekly_loss_limit)}"
    )
    print(
        f"MDD: {format_pct(risk_report.drawdown)} "
        f"/ 한도 {format_pct(risk_limits.max_drawdown_limit)}"
    )
    if risk_report.kill_switch_active:
        print("킬스위치: ACTIVE")
    else:
        print("킬스위치: inactive")

    print("\n[전략 판단]")
    print(f"signal={decision.signal.name} target_weight={decision.target_weight:.3f}")
    print(decision.reason)

    print("\n[파일럿 주문 계획]")
    print(plan)

    updated_state = send_status_alert_if_due(
        client=client,
        notifier=notifier,
        state=state,
        now=now,
        args=args,
        market=market,
        ticker=ticker,
        orderbook=orderbook,
        balances=balances,
        open_orders=open_orders,
        decision=decision,
        risk_report=risk_report,
        risk_limits=risk_limits,
    )
    if updated_state != state:
        save_pilot_state(state_path, updated_state)
        state = updated_state

    if not dry_run:
        save_risk_state(risk_state_path, risk_report.state)

    if plan.action in ("BUY", "SELL"):
        try:
            assert_risk_allows_action(plan.action, risk_report)
        except RiskGateError as exc:
            if (
                runtime_profile.profile in MICRO_PROFILES and
                loss_limit_bypass and
                risk_override_allows(risk_report)
            ):
                print(f"\n[주의] Micro mode가 손실률 게이트를 우회합니다: {exc}")
            else:
                notify_safely(
                    notifier=notifier,
                    embed=build_risk_block_alert_embed(
                        action=plan.action,
                        error=exc,
                        equity=risk_report.equity,
                        dry_run=dry_run,
                    ),
                    dry_run=dry_run,
                    notify_dry_run=args.notify_dry_run,
                )
                raise
        if not dry_run:
            assert_order_quota(
                state=state,
                now=now,
                max_orders_per_day=args.max_orders_per_day,
                cooldown_minutes=args.cooldown_minutes,
                cooldown_seconds=args.cooldown_seconds,
                action=plan.action,
                allow_sell_during_cooldown=runtime_profile.profile in SELECTABLE_MARKET_PROFILES,
            )

    order = execute_plan(client, market, plan)
    if order is None:
        print("\n주문 없음.")
        return None

    if not dry_run:
        save_pilot_state(
            state_path,
            record_order(
                state,
                now,
                action=plan.action,
                is_averaging=(
                    plan.action == "BUY" and
                    "controlled averaging" in plan.reason
                ),
                is_pulse=(
                    plan.action == "BUY" and
                    runtime_profile.profile in SELECTABLE_MARKET_PROFILES
                ),
            ),
        )
        if runtime_profile.profile in MICRO_PROFILES:
            progress = record_rebate_turnover(
                Path(args.turnover_file),
                notional_krw=estimate_order_notional(plan, ticker.trade_price),
                market=market,
                now=now,
                target_krw=args.turnover_target_krw,
            )
            print(f"거래대금 진행률 업데이트: {format_rebate_turnover_progress(progress)}")
        if (
            runtime_profile.profile in START_STOP_PROFILES and
            plan.action == "SELL" and
            "stop auto trade" in plan.reason
        ):
            disable_auto_trade(Path(args.auto_trade_file))
            print(f"자동거래 플래그 제거: {args.auto_trade_file}")
    notify_safely(
        notifier=notifier,
        embed=build_order_alert_embed(
            plan=plan,
            order=order,
            market=market,
            equity=risk_report.equity,
            dry_run=dry_run,
        ),
        dry_run=dry_run,
        notify_dry_run=args.notify_dry_run,
    )
    print("\n주문 요청 완료")
    print(f"uuid={order.uuid} state={order.state.value} dry_run={dry_run}")
    return order


def main() -> None:
    """Run one or repeated guarded ETH/Cash pilot cycles."""
    args = parse_args()
    validate_runner_args(args)
    cfg = get_config()
    setup_logging(log_level=cfg.log_level, enable_file_log=not args.no_file_log)
    validate_live_permission(cfg, live=args.live, confirmed=args.yes_i_understand_risk)

    dry_run = not args.live
    state_path = Path(args.state_file)
    risk_state_path = Path(args.risk_state_file)
    kill_switch_path = Path(args.kill_switch_file)
    risk_limits = RiskLimits.from_config(cfg)
    notifier = DiscordNotifier.from_config(cfg)
    client = BithumbClient.from_config(
        cfg,
        dry_run=dry_run,
        allow_live_orders=args.live and cfg.is_live_trading,
    )

    try:
        print_banner(args, dry_run)
        while True:
            try:
                run_cycle(
                    client=client,
                    args=args,
                    dry_run=dry_run,
                    state_path=state_path,
                    risk_state_path=risk_state_path,
                    kill_switch_path=kill_switch_path,
                    risk_limits=risk_limits,
                    notifier=notifier,
                )
            except Exception as exc:
                print(f"\n[차단/오류] {type(exc).__name__}: {exc}")
                if not isinstance(exc, RiskGateError):
                    notify_safely(
                        notifier=notifier,
                        embed=build_error_alert_embed(error=exc, dry_run=dry_run),
                        dry_run=dry_run,
                        notify_dry_run=args.notify_dry_run,
                    )
                if not args.loop:
                    raise

            if not args.loop:
                return

            sleep_seconds = calculate_sleep_seconds(
                now=datetime.now(),
                interval_minutes=args.interval_minutes,
                align_to_clock=args.align_interval_to_clock,
                interval_seconds=args.interval_seconds,
            )
            print(f"\n다음 판단까지 {format_sleep_wait(sleep_seconds)} 대기...")
            time.sleep(sleep_seconds)
    finally:
        client.close()


if __name__ == "__main__":
    main()
