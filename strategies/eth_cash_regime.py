"""
ETH/Cash regime switch strategy.

This is an ETH-specialist research candidate. It is intentionally sparse:
hold cash most of the time, enter only when ETH shows a long-term trend,
recent momentum, and a breakout, then exit on trend decay or crash pressure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision


@dataclass(frozen=True)
class ETHCashRegimeSnapshot:
    """Point-in-time diagnostics for ETH/Cash decisions."""
    close: float
    fast_ma: float
    slow_ma: float
    exit_ma: float
    momentum: float
    rsi: float
    realized_volatility: float
    breakout_level: float
    crash_risk: bool
    trend_ready: bool
    breakout_ready: bool
    target_weight: float


class ETHCashRegimeStrategy(BaseStrategy):
    """Sparse ETH/cash switching model for the post-2024 regime study."""

    def __init__(self, params: Dict = None):
        params = params or {}
        params.setdefault("fast_window", 20)
        params.setdefault("slow_window", 180)
        params.setdefault("exit_window", 60)
        params.setdefault("momentum_window", 30)
        params.setdefault("momentum_min", 0.10)
        params.setdefault("rsi_window", 14)
        params.setdefault("rsi_min", 44)
        params.setdefault("rsi_max", 92)
        params.setdefault("vol_window", 20)
        params.setdefault("periods_per_year", 365)
        params.setdefault("max_entry_volatility", 1.30)
        params.setdefault("exit_volatility", 1.625)
        params.setdefault("target_volatility", 0.38)
        params.setdefault("breakout_window", 60)
        params.setdefault("breakout_tolerance", 0.08)
        params.setdefault("trend_spread", 0.0)
        params.setdefault("exit_momentum", -0.10)
        params.setdefault("exit_rsi", 38)
        params.setdefault("crash_5d", -0.20)
        params.setdefault("crash_10d", -0.25)
        params.setdefault("trailing_stop", 0.32)
        params.setdefault("max_position", 1.00)
        params.setdefault("min_position", 0.25)
        params.setdefault("required_history_days", max(
            params["slow_window"],
            params["exit_window"],
            params["momentum_window"],
            params["rsi_window"],
            params["vol_window"],
            params["breakout_window"],
        ) + 2)
        super().__init__(name="ETH_Cash_Regime", params=params)
        self.peak_close = 0.0

    def _validate_params(self) -> None:
        assert 0 < self.params["fast_window"] < self.params["slow_window"]
        assert 0 < self.params["exit_window"] <= self.params["slow_window"]
        assert self.params["momentum_window"] > 0
        assert self.params["rsi_window"] > 0
        assert self.params["vol_window"] > 0
        assert self.params["periods_per_year"] > 0
        assert self.params["breakout_window"] > 0
        assert 0 <= self.params["breakout_tolerance"] < 1
        assert 0 < self.params["min_position"] <= self.params["max_position"] <= 1
        assert self.params["target_volatility"] > 0
        assert self.params["max_entry_volatility"] > 0
        assert self.params["exit_volatility"] > 0
        assert 0 < self.params["trailing_stop"] < 1

    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        if len(ctx.ohlcv) < self.warmup_period():
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason="ETH/Cash regime warmup",
            )

        snapshot = self.evaluate(ctx.ohlcv)
        reason = self._format_reason(snapshot)

        if ctx.current_position > 0:
            self.peak_close = max(self.peak_close, snapshot.close)
            trailing_exit = (
                self.peak_close > 0 and
                snapshot.close < self.peak_close * (1 - self.params["trailing_stop"])
            )
            should_exit = (
                snapshot.close < snapshot.exit_ma or
                snapshot.momentum < self.params["exit_momentum"] or
                snapshot.rsi < self.params["exit_rsi"] or
                snapshot.realized_volatility > self.params["exit_volatility"] or
                snapshot.crash_risk or
                trailing_exit
            )
            if should_exit:
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=f"ETH/Cash risk-off | {reason}",
                    confidence=1.0,
                )
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason=f"ETH/Cash hold | {reason}",
                confidence=0.75,
            )

        self.peak_close = 0.0
        should_enter = (
            snapshot.trend_ready and
            snapshot.breakout_ready and
            snapshot.momentum > self.params["momentum_min"] and
            self.params["rsi_min"] < snapshot.rsi < self.params["rsi_max"] and
            snapshot.realized_volatility < self.params["max_entry_volatility"] and
            not snapshot.crash_risk and
            snapshot.target_weight > 0
        )
        if should_enter:
            self.peak_close = snapshot.close
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=snapshot.target_weight,
                reason=f"ETH/Cash breakout entry | {reason}",
                confidence=0.85,
            )

        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason=f"ETH/Cash cash | {reason}",
            confidence=0.0,
        )

    def evaluate(self, ohlcv) -> ETHCashRegimeSnapshot:
        close = ohlcv["close"].astype(float)
        returns = close.pct_change()
        p = self.params

        fast_ma = close.rolling(p["fast_window"]).mean().iloc[-1]
        slow_ma = close.rolling(p["slow_window"]).mean().iloc[-1]
        exit_ma = close.rolling(p["exit_window"]).mean().iloc[-1]
        momentum = close.pct_change(p["momentum_window"]).iloc[-1]
        momentum_5d = close.pct_change(5).iloc[-1]
        momentum_10d = close.pct_change(10).iloc[-1]
        rsi = _rsi(close, p["rsi_window"]).iloc[-1]
        realized_volatility = (
            returns.rolling(p["vol_window"]).std().iloc[-1] *
            np.sqrt(p["periods_per_year"])
        )
        breakout_level = close.shift(1).rolling(p["breakout_window"]).max().iloc[-1]

        current_close = close.iloc[-1]
        crash_risk = bool(
            momentum_5d < p["crash_5d"] or
            momentum_10d < p["crash_10d"]
        )
        trend_ready = bool(
            current_close > slow_ma and
            fast_ma > slow_ma * (1 + p["trend_spread"])
        )
        breakout_ready = bool(
            current_close > breakout_level * (1 - p["breakout_tolerance"])
        )

        target_weight = self._target_weight(realized_volatility)

        return ETHCashRegimeSnapshot(
            close=float(current_close),
            fast_ma=float(fast_ma),
            slow_ma=float(slow_ma),
            exit_ma=float(exit_ma),
            momentum=float(momentum),
            rsi=float(rsi),
            realized_volatility=float(realized_volatility),
            breakout_level=float(breakout_level),
            crash_risk=crash_risk,
            trend_ready=trend_ready,
            breakout_ready=breakout_ready,
            target_weight=target_weight,
        )

    def _target_weight(self, realized_volatility: float) -> float:
        if not np.isfinite(realized_volatility) or realized_volatility <= 0:
            return 0.0
        raw_weight = self.params["target_volatility"] / realized_volatility
        return float(np.clip(
            raw_weight,
            self.params["min_position"],
            self.params["max_position"],
        ))

    @staticmethod
    def _format_reason(snapshot: ETHCashRegimeSnapshot) -> str:
        return (
            f"close={snapshot.close:.0f}, mom={snapshot.momentum:.2f}, "
            f"rsi={snapshot.rsi:.1f}, vol={snapshot.realized_volatility:.2f}, "
            f"trend={snapshot.trend_ready}, breakout={snapshot.breakout_ready}, "
            f"crash={snapshot.crash_risk}, weight={snapshot.target_weight:.2f}"
        )


def _rsi(close, window: int):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    return rsi
