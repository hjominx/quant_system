"""ETH adaptive alpha strategy.

This ETH-specialist candidate avoids simple moving-average crossovers. It uses
trend strength, Donchian breakout pressure, ATR volatility, RSI, and an explicit
cost edge filter to decide when ETH exposure is worth taking.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision


@dataclass(frozen=True)
class ETHAdaptiveAlphaSnapshot:
    """Point-in-time diagnostics for adaptive ETH decisions."""
    close: float
    ema_fast: float
    ema_mid: float
    ema_long: float
    ema_slope: float
    momentum: float
    rsi: float
    adx: float
    atr_pct: float
    realized_volatility: float
    donchian_high: float
    donchian_low: float
    breakout_ready: bool
    trend_ready: bool
    volatility_ready: bool
    crash_risk: bool
    quality_score: float
    target_weight: float


class ETHAdaptiveAlphaStrategy(BaseStrategy):
    """ETH/Cash model using trend quality instead of SMA crossover."""

    def __init__(self, params: Dict = None):
        params = params or {}
        params.setdefault("fast_ema", 12)
        params.setdefault("mid_ema", 48)
        params.setdefault("long_ema", 144)
        params.setdefault("slope_window", 18)
        params.setdefault("momentum_window", 24)
        params.setdefault("min_momentum", 0.035)
        params.setdefault("rsi_window", 14)
        params.setdefault("rsi_min", 42)
        params.setdefault("rsi_max", 78)
        params.setdefault("adx_window", 14)
        params.setdefault("min_adx", 20)
        params.setdefault("atr_window", 14)
        params.setdefault("max_atr_pct", 0.085)
        params.setdefault("vol_window", 36)
        params.setdefault("periods_per_year", 365 * 6)
        params.setdefault("max_realized_volatility", 1.65)
        params.setdefault("target_volatility", 0.28)
        params.setdefault("donchian_window", 72)
        params.setdefault("breakout_tolerance", 0.035)
        params.setdefault("exit_donchian_window", 30)
        params.setdefault("exit_score", 0.40)
        params.setdefault("entry_score", 0.66)
        params.setdefault("atr_stop_multiple", 3.2)
        params.setdefault("hard_stop_pct", 0.085)
        params.setdefault("crash_3bar", -0.075)
        params.setdefault("crash_6bar", -0.110)
        params.setdefault("round_trip_cost_estimate", 0.0014)
        params.setdefault("cost_edge_multiple", 4.0)
        params.setdefault("min_position", 0.15)
        params.setdefault("max_position", 0.60)
        params.setdefault("cooldown_bars", 18)
        params.setdefault("required_history_days", max(
            params["long_ema"],
            params["momentum_window"],
            params["rsi_window"],
            params["adx_window"],
            params["atr_window"],
            params["vol_window"],
            params["donchian_window"],
            params["exit_donchian_window"],
        ) + params["slope_window"] + 5)
        super().__init__(name="ETH_Adaptive_Alpha", params=params)
        self.peak_close = 0.0
        self.last_exit_timestamp = None

    def _validate_params(self) -> None:
        assert 0 < self.params["fast_ema"] < self.params["mid_ema"] < self.params["long_ema"]
        assert self.params["slope_window"] > 0
        assert self.params["momentum_window"] > 0
        assert self.params["rsi_window"] > 0
        assert self.params["adx_window"] > 0
        assert self.params["atr_window"] > 0
        assert self.params["vol_window"] > 0
        assert self.params["periods_per_year"] > 0
        assert self.params["donchian_window"] > 0
        assert self.params["exit_donchian_window"] > 0
        assert 0 <= self.params["breakout_tolerance"] < 1
        assert self.params["entry_score"] > self.params["exit_score"]
        assert self.params["atr_stop_multiple"] > 0
        assert 0 < self.params["hard_stop_pct"] < 1
        assert self.params["round_trip_cost_estimate"] >= 0
        assert self.params["cost_edge_multiple"] >= 0
        assert 0 < self.params["min_position"] <= self.params["max_position"] <= 1
        assert self.params["cooldown_bars"] >= 0

    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        if len(ctx.ohlcv) < self.warmup_period():
            return TradeDecision(Signal.HOLD, 0.0, "ETH adaptive warmup", 0.0)

        snapshot = self.evaluate(ctx.ohlcv)
        reason = self._format_reason(snapshot)

        if ctx.current_position > 0:
            self.peak_close = max(self.peak_close, snapshot.close)
            atr_stop = self.peak_close * (1 - self.params["hard_stop_pct"])
            if np.isfinite(snapshot.atr_pct):
                atr_stop = max(
                    atr_stop,
                    self.peak_close - snapshot.close * snapshot.atr_pct * self.params["atr_stop_multiple"],
                )
            trend_exit = (
                snapshot.quality_score < self.params["exit_score"] or
                snapshot.close < snapshot.ema_mid or
                snapshot.close < snapshot.donchian_low
            )
            stop_exit = snapshot.close < atr_stop
            weak_exit = snapshot.rsi < self.params["rsi_min"] - 6
            if trend_exit or stop_exit or weak_exit or snapshot.crash_risk:
                self.last_exit_timestamp = ctx.timestamp
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=f"ETH adaptive risk-off | {reason}",
                    confidence=1.0,
                )
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason=f"ETH adaptive hold | {reason}",
                confidence=0.75,
            )

        self.peak_close = 0.0
        if not self._cooldown_complete(ctx.timestamp):
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason=f"ETH adaptive cooldown | {reason}",
                confidence=0.0,
            )

        min_edge = self.params["round_trip_cost_estimate"] * self.params["cost_edge_multiple"]
        cost_edge_ready = snapshot.momentum > max(self.params["min_momentum"], min_edge)
        entry_ready = (
            snapshot.quality_score >= self.params["entry_score"] and
            snapshot.trend_ready and
            snapshot.breakout_ready and
            snapshot.volatility_ready and
            cost_edge_ready and
            self.params["rsi_min"] <= snapshot.rsi <= self.params["rsi_max"] and
            not snapshot.crash_risk and
            snapshot.target_weight > 0
        )
        if entry_ready:
            self.peak_close = snapshot.close
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=snapshot.target_weight,
                reason=f"ETH adaptive quality entry | {reason}",
                confidence=min(1.0, snapshot.quality_score),
            )

        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason=f"ETH adaptive cash | {reason}",
            confidence=0.0,
        )

    def evaluate(self, ohlcv) -> ETHAdaptiveAlphaSnapshot:
        p = self.params
        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        returns = close.pct_change()

        ema_fast_series = close.ewm(span=p["fast_ema"], adjust=False).mean()
        ema_mid_series = close.ewm(span=p["mid_ema"], adjust=False).mean()
        ema_long_series = close.ewm(span=p["long_ema"], adjust=False).mean()
        ema_fast = ema_fast_series.iloc[-1]
        ema_mid = ema_mid_series.iloc[-1]
        ema_long = ema_long_series.iloc[-1]
        ema_slope = (ema_long / ema_long_series.iloc[-p["slope_window"]] - 1)
        momentum = close.pct_change(p["momentum_window"]).iloc[-1]
        rsi = _rsi(close, p["rsi_window"]).iloc[-1]
        atr = _atr(high, low, close, p["atr_window"]).iloc[-1]
        atr_pct = atr / close.iloc[-1]
        adx = _adx(high, low, close, p["adx_window"]).iloc[-1]
        realized_volatility = returns.rolling(p["vol_window"]).std().iloc[-1] * np.sqrt(p["periods_per_year"])
        donchian_high = high.shift(1).rolling(p["donchian_window"]).max().iloc[-1]
        donchian_low = low.shift(1).rolling(p["exit_donchian_window"]).min().iloc[-1]

        momentum_3bar = close.pct_change(3).iloc[-1]
        momentum_6bar = close.pct_change(6).iloc[-1]
        current_close = close.iloc[-1]
        crash_risk = bool(
            momentum_3bar < p["crash_3bar"] or
            momentum_6bar < p["crash_6bar"]
        )
        trend_ready = bool(
            current_close > ema_long and
            ema_fast > ema_mid > ema_long and
            ema_slope > 0
        )
        breakout_ready = bool(
            current_close >= donchian_high * (1 - p["breakout_tolerance"])
        )
        volatility_ready = bool(
            atr_pct < p["max_atr_pct"] and
            realized_volatility < p["max_realized_volatility"]
        )
        quality_score = _quality_score(
            trend_ready=trend_ready,
            breakout_ready=breakout_ready,
            volatility_ready=volatility_ready,
            momentum=momentum,
            min_momentum=p["min_momentum"],
            rsi=rsi,
            adx=adx,
            min_adx=p["min_adx"],
            ema_slope=ema_slope,
            crash_risk=crash_risk,
        )
        target_weight = self._target_weight(realized_volatility, quality_score)

        return ETHAdaptiveAlphaSnapshot(
            close=float(current_close),
            ema_fast=float(ema_fast),
            ema_mid=float(ema_mid),
            ema_long=float(ema_long),
            ema_slope=float(ema_slope),
            momentum=float(momentum),
            rsi=float(rsi),
            adx=float(adx),
            atr_pct=float(atr_pct),
            realized_volatility=float(realized_volatility),
            donchian_high=float(donchian_high),
            donchian_low=float(donchian_low),
            breakout_ready=breakout_ready,
            trend_ready=trend_ready,
            volatility_ready=volatility_ready,
            crash_risk=crash_risk,
            quality_score=float(quality_score),
            target_weight=float(target_weight),
        )

    def _target_weight(self, realized_volatility: float, quality_score: float) -> float:
        if not np.isfinite(realized_volatility) or realized_volatility <= 0:
            return 0.0
        raw = self.params["target_volatility"] / realized_volatility
        quality_adjusted = raw * np.clip(quality_score, 0.50, 1.00)
        return float(np.clip(
            quality_adjusted,
            self.params["min_position"],
            self.params["max_position"],
        ))

    def _cooldown_complete(self, timestamp) -> bool:
        if self.last_exit_timestamp is None:
            return True
        cooldown = timedelta(hours=4 * self.params["cooldown_bars"])
        return timestamp - self.last_exit_timestamp >= cooldown

    @staticmethod
    def _format_reason(snapshot: ETHAdaptiveAlphaSnapshot) -> str:
        return (
            f"close={snapshot.close:.0f}, score={snapshot.quality_score:.2f}, "
            f"mom={snapshot.momentum:.2f}, rsi={snapshot.rsi:.1f}, "
            f"adx={snapshot.adx:.1f}, atr={snapshot.atr_pct*100:.2f}%, "
            f"vol={snapshot.realized_volatility:.2f}, trend={snapshot.trend_ready}, "
            f"breakout={snapshot.breakout_ready}, crash={snapshot.crash_risk}, "
            f"weight={snapshot.target_weight:.2f}"
        )


def _quality_score(
    *,
    trend_ready: bool,
    breakout_ready: bool,
    volatility_ready: bool,
    momentum: float,
    min_momentum: float,
    rsi: float,
    adx: float,
    min_adx: float,
    ema_slope: float,
    crash_risk: bool,
) -> float:
    momentum_score = np.clip(momentum / max(min_momentum * 3, 1e-9), 0, 1)
    adx_score = np.clip((adx - min_adx) / 22, 0, 1)
    rsi_score = 1 - abs(np.clip(rsi, 30, 85) - 58) / 35
    slope_score = np.clip(ema_slope / 0.035, 0, 1)
    score = (
        0.24 * float(trend_ready) +
        0.18 * float(breakout_ready) +
        0.14 * float(volatility_ready) +
        0.18 * momentum_score +
        0.12 * adx_score +
        0.08 * rsi_score +
        0.06 * slope_score
    )
    if crash_risk:
        score -= 0.40
    return float(np.clip(score, 0, 1))


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


def _adx(high, low, close, window: int):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(high, low, close, window).replace(0, np.nan)
    plus_di = 100 * close.copy().mul(0).add(plus_dm).ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * close.copy().mul(0).add(minus_dm).ewm(alpha=1 / window, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / window, adjust=False).mean().fillna(0)
