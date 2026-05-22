"""
QPE Trend Overlay 전략

목표:
- SMA 기준선의 상승 포착력을 유지한다.
- QPE 시장상태 확률을 이용해 급락/불확실 구간 노출을 줄인다.
- QPE v1보다 공격적으로 수익을 확보해 Sharpe 개선 후보를 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.market_features import MarketFeatureExtractor, MarketFeatureSnapshot
from regime.market_state import MarketStateProbabilities, QuantumMarketStateEngine
from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision


@dataclass(frozen=True)
class QPETrendOverlaySnapshot:
    """QPE Trend Overlay 의사결정 스냅샷"""
    overlay_score: float
    trend_score: float
    breakout_score: float
    qpe_constructive: float
    crash_risk: float
    target_weight: float
    fast_above_slow: bool
    close_above_slow: bool
    features: MarketFeatureSnapshot
    states: MarketStateProbabilities


class QPETrendOverlayStrategy(BaseStrategy):
    """SMA/모멘텀 공격 엔진 + QPE 리스크 필터"""

    def __init__(self, params: Dict = None):
        params = params or {}
        params.setdefault("short_window", 20)
        params.setdefault("long_window", 100)
        params.setdefault("momentum_window", 60)
        params.setdefault("breakout_window", 80)
        params.setdefault("entry_threshold", 0.50)
        params.setdefault("exit_threshold", 0.34)
        params.setdefault("trend_entry_threshold", 0.52)
        params.setdefault("crash_entry_threshold", 0.52)
        params.setdefault("crash_exit_threshold", 0.66)
        params.setdefault("max_position", 0.95)
        params.setdefault("min_position", 0.10)
        params.setdefault("target_volatility", 0.55)
        params.setdefault("uncertainty_penalty", 0.22)
        params.setdefault("crash_penalty", 0.50)
        params.setdefault("required_history_days", max(
            params["long_window"],
            params["momentum_window"],
            params["breakout_window"],
            260,
        ) + 5)
        super().__init__(name="QPE_Trend_Overlay", params=params)

        self.feature_extractor = MarketFeatureExtractor(params)
        self.state_engine = QuantumMarketStateEngine()
        self.required_history_days = max(
            self.required_history_days,
            self.feature_extractor.min_history,
        )

    def _validate_params(self) -> None:
        assert 0 < self.params["short_window"] < self.params["long_window"]
        assert self.params["momentum_window"] > 0
        assert self.params["breakout_window"] > 0
        assert 0 < self.params["exit_threshold"] < self.params["entry_threshold"] < 1
        assert 0 < self.params["trend_entry_threshold"] < 1
        assert 0 < self.params["crash_entry_threshold"] < 1
        assert 0 < self.params["crash_exit_threshold"] < 1
        assert 0 < self.params["min_position"] <= self.params["max_position"] <= 1
        assert self.params["target_volatility"] > 0
        assert self.params["uncertainty_penalty"] >= 0
        assert self.params["crash_penalty"] >= 0

    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        if len(ctx.ohlcv) < self.warmup_period():
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason="QPE Trend Overlay warmup 기간",
            )

        snapshot = self.evaluate(ctx.ohlcv)
        reason = self._format_reason(snapshot)

        if ctx.current_position > 0:
            should_exit = (
                snapshot.overlay_score <= self.params["exit_threshold"] or
                snapshot.crash_risk >= self.params["crash_exit_threshold"] or
                not snapshot.fast_above_slow or
                not snapshot.close_above_slow
            )
            if should_exit:
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=f"QPE overlay risk-off | {reason}",
                    confidence=1.0 - snapshot.overlay_score,
                )
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason=f"QPE overlay hold | {reason}",
                confidence=snapshot.overlay_score,
            )

        should_enter = (
            snapshot.overlay_score >= self.params["entry_threshold"] and
            snapshot.trend_score >= self.params["trend_entry_threshold"] and
            snapshot.crash_risk < self.params["crash_entry_threshold"] and
            snapshot.fast_above_slow and
            snapshot.target_weight > 0
        )
        if should_enter:
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=snapshot.target_weight,
                reason=f"QPE overlay trend entry | {reason}",
                confidence=snapshot.overlay_score,
            )

        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason=f"QPE overlay no-entry | {reason}",
            confidence=snapshot.overlay_score,
        )

    def evaluate(self, ohlcv) -> QPETrendOverlaySnapshot:
        features = self.feature_extractor.extract(ohlcv)
        states = self.state_engine.infer(features)

        close = ohlcv["close"].astype(float)
        short_window = self.params["short_window"]
        long_window = self.params["long_window"]
        momentum_window = self.params["momentum_window"]
        breakout_window = self.params["breakout_window"]

        fast_ma = close.rolling(short_window).mean().iloc[-1]
        slow_ma = close.rolling(long_window).mean().iloc[-1]
        trend_spread = (fast_ma / max(slow_ma, 1e-12)) - 1.0
        price_spread = (close.iloc[-1] / max(slow_ma, 1e-12)) - 1.0
        momentum = close.pct_change(momentum_window).iloc[-1]

        fast_above_slow = bool(fast_ma > slow_ma)
        close_above_slow = bool(close.iloc[-1] > slow_ma)
        trend_score = _clip01(
            0.45 * _sigmoid(35.0 * trend_spread) +
            0.35 * _sigmoid(7.5 * momentum) +
            0.20 * _sigmoid(30.0 * price_spread)
        )

        prior_high = close.shift(1).rolling(breakout_window).max().iloc[-1]
        breakout_distance = (close.iloc[-1] / max(prior_high, 1e-12)) - 1.0
        breakout_score = _clip01(_sigmoid(45.0 * breakout_distance))

        qpe_constructive = _clip01(
            0.46 * states.trend_up +
            0.26 * states.volatility_breakout +
            0.16 * states.mean_reversion +
            0.12 * features.relative_strength
        )
        crash_risk = max(states.crash_risk, features.crash_risk)

        overlay_score = _clip01(
            0.48 * trend_score +
            0.20 * breakout_score +
            0.24 * qpe_constructive +
            0.08 * features.volume_pressure -
            0.18 * crash_risk -
            0.06 * states.uncertainty
        )
        target_weight = self._target_weight(overlay_score, crash_risk, states.uncertainty, features.realized_volatility)

        return QPETrendOverlaySnapshot(
            overlay_score=overlay_score,
            trend_score=trend_score,
            breakout_score=breakout_score,
            qpe_constructive=qpe_constructive,
            crash_risk=crash_risk,
            target_weight=target_weight,
            fast_above_slow=fast_above_slow,
            close_above_slow=close_above_slow,
            features=features,
            states=states,
        )

    def _target_weight(
        self,
        overlay_score: float,
        crash_risk: float,
        uncertainty: float,
        realized_volatility: float,
    ) -> float:
        realized_volatility = max(float(realized_volatility), 1e-6)
        vol_scalar = np.clip(self.params["target_volatility"] / realized_volatility, 0.55, 1.20)
        risk_scalar = (
            (1.0 - self.params["uncertainty_penalty"] * uncertainty) *
            (1.0 - self.params["crash_penalty"] * crash_risk)
        )
        conviction_scalar = 0.62 + 0.38 * _clip01(overlay_score)
        weight = self.params["max_position"] * conviction_scalar * vol_scalar * max(risk_scalar, 0.0)
        if weight < self.params["min_position"]:
            return 0.0
        return float(np.clip(weight, 0.0, self.params["max_position"]))

    @staticmethod
    def _format_reason(snapshot: QPETrendOverlaySnapshot) -> str:
        return (
            f"score={snapshot.overlay_score:.2f}, trend={snapshot.trend_score:.2f}, "
            f"breakout={snapshot.breakout_score:.2f}, qpe={snapshot.qpe_constructive:.2f}, "
            f"crash={snapshot.crash_risk:.2f}, weight={snapshot.target_weight:.2f}, "
            f"stateTrend={snapshot.states.trend_up:.2f}, uncertainty={snapshot.states.uncertainty:.2f}"
        )


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -50, 50))
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
