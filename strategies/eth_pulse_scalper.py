"""Pulse scalping strategy for a selected KRW market.

This profile is deliberately mechanical:
- start a 100k KRW buy pulse every configured interval
- while holding the selected coin, check live ticker price for tiny TP/SL exits

It is extremely fee and spread sensitive, so it should remain isolated from
the indicator-driven stable/scalp profiles.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision


class PulseScalperStrategy(BaseStrategy):
    """Time-pulsed KRW-market scalper using live ticker price for exits."""

    def __init__(self, params: Dict = None):
        params = params or {}
        params.setdefault("buy_interval_seconds", 600)
        params.setdefault("min_exit_hold_seconds", 3)
        params.setdefault("take_profit_pct", 0.0030)
        params.setdefault("stop_loss_pct", 0.0004)
        params.setdefault("max_position", 1.0)
        params.setdefault("required_history_days", 1)
        super().__init__(name="Pulse_Scalper", params=params)

    def _validate_params(self) -> None:
        assert self.params["buy_interval_seconds"] > 0
        assert self.params["min_exit_hold_seconds"] >= 0
        assert 0 < self.params["take_profit_pct"] < 1
        assert 0 < self.params["stop_loss_pct"] < 1
        assert 0 < self.params["max_position"] <= 1

    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        price = self._current_price(ctx)
        now = self._current_time(ctx)
        last_buy_at = self._last_pulse_buy_at(ctx)
        seconds_since_buy = (
            None if last_buy_at is None else (now - last_buy_at).total_seconds()
        )
        buy_due = (
            last_buy_at is None or
            seconds_since_buy >= self.params["buy_interval_seconds"]
        )
        exit_armed = (
            last_buy_at is None or
            seconds_since_buy >= self.params["min_exit_hold_seconds"]
        )

        if ctx.current_position > 0 and ctx.avg_buy_price > 0:
            pnl = price / ctx.avg_buy_price - 1
            if exit_armed and pnl >= self.params["take_profit_pct"]:
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=self._reason(ctx.ticker, "take-profit", price, pnl, seconds_since_buy),
                    confidence=1.0,
                    take_profit_pct=self.params["take_profit_pct"],
                    stop_loss_pct=-self.params["stop_loss_pct"],
                )
            if exit_armed and pnl <= -self.params["stop_loss_pct"]:
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=self._reason(ctx.ticker, "stop-loss", price, pnl, seconds_since_buy),
                    confidence=1.0,
                    take_profit_pct=self.params["take_profit_pct"],
                    stop_loss_pct=-self.params["stop_loss_pct"],
                )
            if buy_due:
                return TradeDecision(
                    signal=Signal.BUY,
                    target_weight=self.params["max_position"],
                    reason=self._reason(ctx.ticker, "10m pulse add", price, pnl, seconds_since_buy),
                    confidence=1.0,
                    take_profit_pct=self.params["take_profit_pct"],
                    stop_loss_pct=-self.params["stop_loss_pct"],
                )
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason=self._reason(ctx.ticker, "monitor", price, pnl, seconds_since_buy),
                confidence=0.0,
            )

        if buy_due:
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=self.params["max_position"],
                reason=self._reason(ctx.ticker, "10m pulse entry", price, 0.0, seconds_since_buy),
                confidence=1.0,
                take_profit_pct=self.params["take_profit_pct"],
                stop_loss_pct=-self.params["stop_loss_pct"],
            )

        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason=self._reason(ctx.ticker, "cash wait", price, 0.0, seconds_since_buy),
            confidence=0.0,
        )

    def _current_price(self, ctx: StrategyContext) -> float:
        price = ctx.metadata.get("trade_price")
        if price is not None:
            return float(price)
        return float(ctx.ohlcv["close"].iloc[-1])

    def _current_time(self, ctx: StrategyContext) -> datetime:
        raw = ctx.metadata.get("now")
        if raw:
            return datetime.fromisoformat(str(raw))
        ts = ctx.timestamp
        if isinstance(ts, datetime):
            return ts
        return datetime.fromisoformat(str(ts))

    def _last_pulse_buy_at(self, ctx: StrategyContext) -> Optional[datetime]:
        raw = ctx.metadata.get("last_pulse_buy_at")
        if not raw:
            return None
        return datetime.fromisoformat(str(raw))

    def _reason(
        self,
        ticker: str,
        action: str,
        price: float,
        pnl: float,
        seconds_since_buy: Optional[float],
    ) -> str:
        if seconds_since_buy is None:
            age = "never"
        elif seconds_since_buy < 60:
            age = f"{seconds_since_buy:.0f}s"
        else:
            age = f"{seconds_since_buy / 60:.1f}m"
        return (
            f"{ticker} pulse {action} | price={price:.0f}, pnl={pnl:.3%}, "
            f"last_pulse_age={age}, interval={self.params['buy_interval_seconds']}s, "
            f"exit_after={self.params['min_exit_hold_seconds']}s, "
            f"tp={self.params['take_profit_pct']:.2%}, sl={self.params['stop_loss_pct']:.2%}"
        )


ETHPulseScalperStrategy = PulseScalperStrategy
