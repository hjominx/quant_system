"""Pulse scalping strategy for a selected KRW market.

This profile is deliberately mechanical:
- start a 100k KRW buy pulse every configured interval
- while holding the selected coin, check live ticker price for tiny TP/SL exits

It is extremely fee and spread sensitive, so it should remain isolated from
the indicator-driven stable/scalp profiles.
"""
from __future__ import annotations

from datetime import datetime
from statistics import pstdev
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision


class PulseScalperStrategy(BaseStrategy):
    """Time-pulsed KRW-market scalper using live ticker price for exits."""

    def __init__(self, params: Dict = None):
        params = params or {}
        params.setdefault("label", "pulse")
        params.setdefault("buy_interval_seconds", 600)
        params.setdefault("min_exit_hold_seconds", 3)
        params.setdefault("take_profit_pct", 0.0030)
        params.setdefault("stop_loss_pct", 0.0004)
        params.setdefault("max_position", 1.0)
        params.setdefault("required_history_days", 1)
        params.setdefault("add_while_holding", True)
        params.setdefault("entry_signal_required", False)
        params.setdefault("min_micro_momentum", 0.0)
        params.setdefault("min_micro_volatility", 0.0)
        params.setdefault("momentum_lookback", 3)
        params.setdefault("volatility_window", 8)
        params.setdefault("reversal_entry_enabled", False)
        params.setdefault("reversal_lookback", 8)
        params.setdefault("reversal_min_rebound", 0.0010)
        params.setdefault("breakout_tolerance", 0.0010)
        params.setdefault("min_reentry_seconds", 0)
        params.setdefault("take_profit_min_pct", params["take_profit_pct"])
        params.setdefault("take_profit_max_pct", params["take_profit_pct"])
        params.setdefault("take_profit_vol_multiplier", 1.0)
        params.setdefault("positive_ev_required", False)
        params.setdefault("max_entry_spread_pct", 1.0)
        params.setdefault("fee_round_trip_pct", 0.0)
        params.setdefault("rebate_round_trip_pct", 0.0)
        params.setdefault("slippage_buffer_pct", 0.0)
        params.setdefault("min_expected_net_pct", 0.0)
        params.setdefault("min_take_profit_after_cost_pct", 0.0)
        params.setdefault("min_reward_to_risk", 0.0)
        super().__init__(name="Pulse_Scalper", params=params)

    def _validate_params(self) -> None:
        assert self.params["buy_interval_seconds"] >= 0
        assert self.params["min_exit_hold_seconds"] >= 0
        assert 0 < self.params["take_profit_pct"] < 1
        assert 0 < self.params["take_profit_min_pct"] <= self.params["take_profit_max_pct"] < 1
        assert 0 < self.params["stop_loss_pct"] < 1
        assert 0 < self.params["max_position"] <= 1
        assert self.params["min_reentry_seconds"] >= 0
        assert self.params["max_entry_spread_pct"] >= 0
        assert self.params["fee_round_trip_pct"] >= 0
        assert self.params["rebate_round_trip_pct"] >= 0
        assert self.params["slippage_buffer_pct"] >= 0
        assert self.params["min_expected_net_pct"] >= 0
        assert self.params["min_take_profit_after_cost_pct"] >= 0
        assert self.params["min_reward_to_risk"] >= 0

    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        price = self._current_price(ctx)
        now = self._current_time(ctx)
        last_buy_at = self._last_pulse_buy_at(ctx)
        take_profit_pct = self._take_profit_pct(ctx)
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
            if exit_armed and pnl >= take_profit_pct:
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=self._reason(ctx.ticker, "take-profit", price, pnl, seconds_since_buy, take_profit_pct),
                    confidence=1.0,
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=-self.params["stop_loss_pct"],
                )
            if exit_armed and pnl <= -self.params["stop_loss_pct"]:
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=self._reason(ctx.ticker, "stop-loss", price, pnl, seconds_since_buy, take_profit_pct),
                    confidence=1.0,
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=-self.params["stop_loss_pct"],
                )
            entry_allowed, entry_reason = self._entry_allowed(ctx, now)
            if buy_due and self.params["add_while_holding"] and entry_allowed:
                return TradeDecision(
                    signal=Signal.BUY,
                    target_weight=self.params["max_position"],
                    reason=self._reason(
                        ctx.ticker,
                        f"pulse add | {entry_reason}",
                        price,
                        pnl,
                        seconds_since_buy,
                        take_profit_pct,
                    ),
                    confidence=1.0,
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=-self.params["stop_loss_pct"],
                )
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason=self._reason(
                    ctx.ticker,
                    "monitor" if entry_allowed else f"entry wait: {entry_reason}",
                    price,
                    pnl,
                    seconds_since_buy,
                    take_profit_pct,
                ),
                confidence=0.0,
            )

        entry_allowed, entry_reason = self._entry_allowed(ctx, now)
        if buy_due and entry_allowed:
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=self.params["max_position"],
                reason=self._reason(
                    ctx.ticker,
                    f"pulse entry | {entry_reason}",
                    price,
                    0.0,
                    seconds_since_buy,
                    take_profit_pct,
                ),
                confidence=1.0,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=-self.params["stop_loss_pct"],
            )

        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason=self._reason(
                ctx.ticker,
                "cash wait" if not buy_due else f"entry wait: {entry_reason}",
                price,
                0.0,
                seconds_since_buy,
                take_profit_pct,
            ),
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

    def _last_order_at(self, ctx: StrategyContext) -> Optional[datetime]:
        raw = ctx.metadata.get("last_order_at")
        if not raw:
            return None
        return datetime.fromisoformat(str(raw))

    def _entry_allowed(self, ctx: StrategyContext, now: datetime) -> tuple[bool, str]:
        if not self.params["entry_signal_required"]:
            return True, "entry filter disabled"

        last_action = str(ctx.metadata.get("last_action") or "").upper()
        last_order_at = self._last_order_at(ctx)
        if last_action == "SELL" and last_order_at is not None:
            elapsed = (now - last_order_at).total_seconds()
            if elapsed < self.params["min_reentry_seconds"]:
                return False, f"post-sell reentry guard {elapsed:.0f}s"

        closes = [float(value) for value in ctx.ohlcv["close"].tolist()]
        if len(closes) < max(self.params["momentum_lookback"] + 1, self.params["volatility_window"] + 1):
            return False, "insufficient micro history"
        if min(closes) <= 0:
            return False, "invalid price history"

        returns = [
            closes[idx] / closes[idx - 1] - 1
            for idx in range(1, len(closes))
            if closes[idx - 1] > 0
        ]
        window = int(self.params["volatility_window"])
        volatility = pstdev(returns[-window:]) if len(returns) >= 2 else 0.0
        if volatility < self.params["min_micro_volatility"]:
            return False, f"vol too low {volatility:.3%}"

        lookback = int(self.params["momentum_lookback"])
        momentum = closes[-1] / closes[-1 - lookback] - 1
        recent_return = closes[-1] / closes[-2] - 1
        recent_high = max(closes[-max(lookback * 2, 3):])
        breakout = closes[-1] >= recent_high * (1 - self.params["breakout_tolerance"])
        momentum_ok = (
            momentum >= self.params["min_micro_momentum"] and
            recent_return > -self.params["stop_loss_pct"] and
            breakout
        )

        reversal = 0.0
        reversal_ok = False
        if self.params["reversal_entry_enabled"]:
            reversal_window = int(self.params["reversal_lookback"])
            recent_low = min(closes[-reversal_window:])
            reversal = closes[-1] / recent_low - 1
            reversal_ok = (
                reversal >= self.params["reversal_min_rebound"] and
                recent_return > 0
            )

        if momentum_ok or reversal_ok:
            setup = "momentum" if momentum_ok else "reversal"
            if self.params["positive_ev_required"]:
                ev_allowed, ev_reason = self._positive_ev_allowed(
                    ctx,
                    momentum=momentum,
                    reversal=reversal,
                    volatility=volatility,
                    take_profit_pct=self._take_profit_pct(ctx),
                )
                if not ev_allowed:
                    return False, ev_reason
                return True, f"{setup} mom={momentum:.3%} vol={volatility:.3%} | {ev_reason}"
            return True, f"{setup} mom={momentum:.3%} vol={volatility:.3%}"
        return False, f"no micro edge mom={momentum:.3%} vol={volatility:.3%}"

    def _positive_ev_allowed(
        self,
        ctx: StrategyContext,
        *,
        momentum: float,
        reversal: float,
        volatility: float,
        take_profit_pct: float,
    ) -> tuple[bool, str]:
        """Require a positive post-cost edge before opening a Rebate Micro trade."""
        spread_pct = float(ctx.metadata.get("spread_pct") or 0.0)
        max_spread_pct = float(self.params["max_entry_spread_pct"])
        if spread_pct > max_spread_pct:
            return False, f"spread too wide {spread_pct:.3%}>{max_spread_pct:.3%}"

        fee = float(self.params["fee_round_trip_pct"])
        rebate = float(self.params["rebate_round_trip_pct"])
        slippage = float(self.params["slippage_buffer_pct"])
        round_trip_cost = spread_pct + fee + slippage
        expected_alpha = (
            max(momentum, 0.0) * 0.35 +
            max(reversal, 0.0) * 0.10 +
            max(volatility, 0.0) * 0.45
        )
        expected_net = expected_alpha + rebate - round_trip_cost
        min_expected_net = float(self.params["min_expected_net_pct"])
        if expected_net < min_expected_net:
            return (
                False,
                f"expected EV too low {expected_net:.3%}<{min_expected_net:.3%} "
                f"(spread={spread_pct:.3%}, rebate={rebate:.3%})",
            )

        take_profit_after_cost = take_profit_pct + rebate - round_trip_cost
        min_take_profit_after_cost = float(self.params["min_take_profit_after_cost_pct"])
        if take_profit_after_cost < min_take_profit_after_cost:
            return (
                False,
                f"TP after cost too thin {take_profit_after_cost:.3%}<{min_take_profit_after_cost:.3%}",
            )

        effective_loss = float(self.params["stop_loss_pct"]) + max(round_trip_cost - rebate, 0.0)
        reward_to_risk = take_profit_after_cost / effective_loss if effective_loss > 0 else float("inf")
        min_reward_to_risk = float(self.params["min_reward_to_risk"])
        if reward_to_risk < min_reward_to_risk:
            return False, f"reward/risk too low {reward_to_risk:.2f}<{min_reward_to_risk:.2f}"

        return (
            True,
            f"EV={expected_net:.3%}, TPnet={take_profit_after_cost:.3%}, "
            f"RR={reward_to_risk:.2f}, spread={spread_pct:.3%}",
        )

    def _take_profit_pct(self, ctx: StrategyContext) -> float:
        closes = [float(value) for value in ctx.ohlcv["close"].tolist()]
        if len(closes) < 3:
            return float(self.params["take_profit_min_pct"])
        returns = [
            closes[idx] / closes[idx - 1] - 1
            for idx in range(1, len(closes))
            if closes[idx - 1] > 0
        ]
        window = int(self.params["volatility_window"])
        vol = pstdev(returns[-window:]) if len(returns) >= 2 else 0.0
        raw_target = vol * float(self.params["take_profit_vol_multiplier"])
        return max(
            float(self.params["take_profit_min_pct"]),
            min(float(self.params["take_profit_max_pct"]), raw_target),
        )

    def _reason(
        self,
        ticker: str,
        action: str,
        price: float,
        pnl: float,
        seconds_since_buy: Optional[float],
        take_profit_pct: float,
    ) -> str:
        if seconds_since_buy is None:
            age = "never"
        elif seconds_since_buy < 60:
            age = f"{seconds_since_buy:.0f}s"
        else:
            age = f"{seconds_since_buy / 60:.1f}m"
        return (
            f"{ticker} {self.params['label']} {action} | price={price:.0f}, pnl={pnl:.3%}, "
            f"last_pulse_age={age}, interval={self.params['buy_interval_seconds']}s, "
            f"exit_after={self.params['min_exit_hold_seconds']}s, "
            f"tp={take_profit_pct:.2%}, sl={self.params['stop_loss_pct']:.2%}"
        )


ETHPulseScalperStrategy = PulseScalperStrategy
