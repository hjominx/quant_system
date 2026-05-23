"""ETH micro scalping strategy.

This long-only ETH/KRW profile is intentionally more active than the
4h adaptive model. It uses short EMA structure, breakout/pullback triggers,
volume confirmation, ATR sanity checks, and explicit take-profit/stop-loss
exits. It is designed for tiny pilot sizing, not for large capital.
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
class ETHMicroScalpSnapshot:
    """Point-in-time diagnostics for scalping decisions."""
    close: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    momentum: float
    rsi: float
    atr_pct: float
    realized_volatility: float
    breakout_level: float
    volume_ratio: float
    trend_ready: bool
    breakout_ready: bool
    pullback_ready: bool
    volatility_ready: bool
    crash_risk: bool
    target_weight: float


class ETHMicroScalperStrategy(BaseStrategy):
    """Higher-frequency ETH/cash scalper for 5m candles."""

    def __init__(self, params: Dict = None):
        params = params or {}
        params.setdefault("fast_ema", 5)
        params.setdefault("slow_ema", 21)
        params.setdefault("trend_ema", 55)
        params.setdefault("momentum_window", 3)
        params.setdefault("min_momentum", 0.0012)
        params.setdefault("rsi_window", 14)
        params.setdefault("rsi_entry_min", 42)
        params.setdefault("rsi_entry_max", 76)
        params.setdefault("rsi_exit", 36)
        params.setdefault("atr_window", 14)
        params.setdefault("min_atr_pct", 0.00045)
        params.setdefault("max_atr_pct", 0.018)
        params.setdefault("vol_window", 36)
        params.setdefault("periods_per_year", 365 * 24 * 12)
        params.setdefault("max_realized_volatility", 2.80)
        params.setdefault("target_volatility", 0.75)
        params.setdefault("breakout_window", 18)
        params.setdefault("breakout_tolerance", 0.0035)
        params.setdefault("pullback_tolerance", 0.0060)
        params.setdefault("volume_window", 24)
        params.setdefault("min_volume_ratio", 0.70)
        params.setdefault("take_profit_pct", 0.0060)
        params.setdefault("stop_loss_pct", 0.0045)
        params.setdefault("trailing_stop_pct", 0.0060)
        params.setdefault("crash_3bar", -0.014)
        params.setdefault("crash_6bar", -0.022)
        params.setdefault("min_position", 0.25)
        params.setdefault("max_position", 0.85)
        params.setdefault("required_history_days", max(
            params["trend_ema"],
            params["momentum_window"],
            params["rsi_window"],
            params["atr_window"],
            params["vol_window"],
            params["breakout_window"],
            params["volume_window"],
        ) + 5)
        super().__init__(name="ETH_Micro_Scalper", params=params)
        self.peak_close = 0.0

    def _validate_params(self) -> None:
        assert 0 < self.params["fast_ema"] < self.params["slow_ema"] < self.params["trend_ema"]
        assert self.params["momentum_window"] > 0
        assert self.params["rsi_window"] > 0
        assert self.params["atr_window"] > 0
        assert 0 <= self.params["min_atr_pct"] < self.params["max_atr_pct"]
        assert self.params["vol_window"] > 0
        assert self.params["periods_per_year"] > 0
        assert self.params["breakout_window"] > 0
        assert 0 <= self.params["breakout_tolerance"] < 1
        assert 0 <= self.params["pullback_tolerance"] < 1
        assert self.params["volume_window"] > 0
        assert self.params["min_volume_ratio"] >= 0
        assert 0 < self.params["stop_loss_pct"] < 1
        assert 0 < self.params["take_profit_pct"] < 1
        assert 0 < self.params["trailing_stop_pct"] < 1
        assert 0 < self.params["min_position"] <= self.params["max_position"] <= 1

    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        if len(ctx.ohlcv) < self.warmup_period():
            return TradeDecision(Signal.HOLD, 0.0, "ETH scalp warmup", 0.0)

        snapshot = self.evaluate(ctx.ohlcv)
        reason = self._format_reason(snapshot)

        if ctx.current_position > 0:
            self.peak_close = max(self.peak_close, snapshot.close)
            pnl = 0.0
            if ctx.avg_buy_price > 0:
                pnl = snapshot.close / ctx.avg_buy_price - 1
            take_profit = pnl >= self.params["take_profit_pct"]
            stop_loss = pnl <= -self.params["stop_loss_pct"] if ctx.avg_buy_price > 0 else False
            trailing_exit = (
                self.peak_close > 0 and
                snapshot.close < self.peak_close * (1 - self.params["trailing_stop_pct"])
            )
            trend_exit = (
                snapshot.close < snapshot.ema_slow or
                snapshot.ema_fast < snapshot.ema_slow or
                snapshot.rsi < self.params["rsi_exit"]
            )
            if take_profit or stop_loss or trailing_exit or trend_exit or snapshot.crash_risk:
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=f"ETH scalp exit pnl={pnl:.3%} | {reason}",
                    confidence=1.0,
                    stop_loss_pct=-self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                )
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason=f"ETH scalp hold pnl={pnl:.3%} | {reason}",
                confidence=0.70,
            )

        self.peak_close = 0.0
        entry_ready = (
            snapshot.trend_ready and
            (snapshot.breakout_ready or snapshot.pullback_ready) and
            snapshot.volatility_ready and
            snapshot.volume_ratio >= self.params["min_volume_ratio"] and
            self.params["rsi_entry_min"] <= snapshot.rsi <= self.params["rsi_entry_max"] and
            snapshot.momentum > self.params["min_momentum"] and
            not snapshot.crash_risk and
            snapshot.target_weight > 0
        )
        if entry_ready:
            self.peak_close = snapshot.close
            trigger = "breakout" if snapshot.breakout_ready else "pullback"
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=snapshot.target_weight,
                reason=f"ETH scalp {trigger} entry | {reason}",
                confidence=0.72,
                stop_loss_pct=-self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
            )

        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason=f"ETH scalp cash | {reason}",
            confidence=0.0,
        )

    def evaluate(self, ohlcv) -> ETHMicroScalpSnapshot:
        p = self.params
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        volume = ohlcv["volume"].astype(float)
        returns = close.pct_change()

        ema_fast_series = close.ewm(span=p["fast_ema"], adjust=False).mean()
        ema_slow_series = close.ewm(span=p["slow_ema"], adjust=False).mean()
        ema_trend_series = close.ewm(span=p["trend_ema"], adjust=False).mean()
        ema_fast = ema_fast_series.iloc[-1]
        ema_slow = ema_slow_series.iloc[-1]
        ema_trend = ema_trend_series.iloc[-1]

        momentum = close.pct_change(p["momentum_window"]).iloc[-1]
        rsi = _rsi(close, p["rsi_window"]).iloc[-1]
        atr = _atr(high, low, close, p["atr_window"]).iloc[-1]
        atr_pct = atr / close.iloc[-1]
        realized_volatility = returns.rolling(p["vol_window"]).std().iloc[-1] * np.sqrt(p["periods_per_year"])
        breakout_level = high.shift(1).rolling(p["breakout_window"]).max().iloc[-1]
        avg_volume = volume.shift(1).rolling(p["volume_window"]).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / avg_volume if avg_volume and avg_volume > 0 else 0.0
        current_close = close.iloc[-1]

        momentum_3bar = close.pct_change(3).iloc[-1]
        momentum_6bar = close.pct_change(6).iloc[-1]
        crash_risk = bool(
            momentum_3bar < p["crash_3bar"] or
            momentum_6bar < p["crash_6bar"]
        )
        trend_ready = bool(
            current_close > ema_trend and
            ema_fast > ema_slow and
            ema_slow >= ema_trend * 0.998
        )
        breakout_ready = bool(
            current_close >= breakout_level * (1 - p["breakout_tolerance"])
        )
        pullback_ready = bool(
            current_close > ema_fast and
            abs(current_close / ema_slow - 1) <= p["pullback_tolerance"] and
            momentum > 0
        )
        volatility_ready = bool(
            p["min_atr_pct"] <= atr_pct <= p["max_atr_pct"] and
            realized_volatility <= p["max_realized_volatility"]
        )
        target_weight = self._target_weight(realized_volatility)

        return ETHMicroScalpSnapshot(
            close=float(current_close),
            ema_fast=float(ema_fast),
            ema_slow=float(ema_slow),
            ema_trend=float(ema_trend),
            momentum=float(momentum),
            rsi=float(rsi),
            atr_pct=float(atr_pct),
            realized_volatility=float(realized_volatility),
            breakout_level=float(breakout_level),
            volume_ratio=float(volume_ratio),
            trend_ready=trend_ready,
            breakout_ready=breakout_ready,
            pullback_ready=pullback_ready,
            volatility_ready=volatility_ready,
            crash_risk=crash_risk,
            target_weight=float(target_weight),
        )

    def _target_weight(self, realized_volatility: float) -> float:
        if not np.isfinite(realized_volatility) or realized_volatility <= 0:
            return 0.0
        raw = self.params["target_volatility"] / realized_volatility
        return float(np.clip(raw, self.params["min_position"], self.params["max_position"]))

    @staticmethod
    def _format_reason(snapshot: ETHMicroScalpSnapshot) -> str:
        return (
            f"close={snapshot.close:.0f}, mom={snapshot.momentum:.3%}, "
            f"rsi={snapshot.rsi:.1f}, atr={snapshot.atr_pct:.3%}, "
            f"vol={snapshot.realized_volatility:.2f}, vr={snapshot.volume_ratio:.2f}, "
            f"trend={snapshot.trend_ready}, breakout={snapshot.breakout_ready}, "
            f"pullback={snapshot.pullback_ready}, crash={snapshot.crash_risk}, "
            f"weight={snapshot.target_weight:.2f}"
        )


def _rsi(close, window: int):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50)


def _atr(high, low, close, window: int):
    previous_close = close.shift(1)
    true_range = np.maximum.reduce([
        (high - low).to_numpy(),
        (high - previous_close).abs().to_numpy(),
        (low - previous_close).abs().to_numpy(),
    ])
    return close.copy().mul(0).add(true_range).rolling(window).mean()
