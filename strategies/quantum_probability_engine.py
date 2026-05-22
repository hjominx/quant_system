"""
QPE: Quantum Probability Engine 전략

양자역학의 사고방식을 차용한 시장상태 적응형 복합 알파 엔진.
실제 양자컴퓨팅이 아니라, 시장 상태를 확률 벡터로 들고 있다가
충분한 확신이 생길 때만 포지션으로 "붕괴"시키는 전략이다.
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
from risk.position_sizer import PositionSizingInput, QuantumPositionSizer
from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision


@dataclass(frozen=True)
class QuantumEdgeSnapshot:
    """QPE 의사결정 스냅샷"""
    edge_score: float
    target_weight: float
    features: MarketFeatureSnapshot
    states: MarketStateProbabilities


class QuantumProbabilityEngineStrategy(BaseStrategy):
    """시장 상태 확률 + 복합 알파 점수 기반 현물 롱 전략"""

    def __init__(self, params: Dict = None):
        params = params or {}
        params.setdefault("entry_threshold", 0.48)
        params.setdefault("exit_threshold", 0.32)
        params.setdefault("crash_entry_threshold", 0.36)
        params.setdefault("crash_exit_threshold", 0.46)
        params.setdefault("max_position", 0.85)
        params.setdefault("min_position", 0.05)
        params.setdefault("target_volatility", 0.35)
        params.setdefault("uncertainty_penalty", 0.55)
        params.setdefault("crash_penalty", 0.80)
        params.setdefault("required_history_days", 260)
        super().__init__(name="Quantum_Probability_Engine", params=params)

        self.feature_extractor = MarketFeatureExtractor(params)
        self.state_engine = QuantumMarketStateEngine()
        self.position_sizer = QuantumPositionSizer(
            max_position=self.params["max_position"],
            min_position=self.params["min_position"],
            target_volatility=self.params["target_volatility"],
            uncertainty_penalty=self.params["uncertainty_penalty"],
            crash_penalty=self.params["crash_penalty"],
        )
        self.required_history_days = max(
            self.required_history_days,
            self.feature_extractor.min_history,
        )

    def _validate_params(self) -> None:
        assert 0 < self.params["exit_threshold"] < self.params["entry_threshold"] < 1
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
                reason="QPE warmup 기간",
            )

        edge = self.evaluate(ctx.ohlcv)
        reason = self._format_reason(edge)

        if ctx.current_position > 0:
            if (
                edge.edge_score <= self.params["exit_threshold"] or
                edge.states.crash_risk >= self.params["crash_exit_threshold"]
            ):
                return TradeDecision(
                    signal=Signal.SELL,
                    target_weight=0.0,
                    reason=f"QPE collapse exit | {reason}",
                    confidence=1.0 - edge.edge_score,
                )
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason=f"QPE hold | {reason}",
                confidence=edge.edge_score,
            )

        if (
            edge.edge_score >= self.params["entry_threshold"] and
            edge.states.crash_risk < self.params["crash_entry_threshold"] and
            edge.target_weight > 0
        ):
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=edge.target_weight,
                reason=f"QPE probability collapse entry | {reason}",
                confidence=edge.edge_score,
            )

        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason=f"QPE no-entry | {reason}",
            confidence=edge.edge_score,
        )

    def evaluate(self, ohlcv) -> QuantumEdgeSnapshot:
        features = self.feature_extractor.extract(ohlcv)
        states = self.state_engine.infer(features)
        edge_score = self._calculate_edge_score(features, states)
        target_weight = self.position_sizer.size(PositionSizingInput(
            edge_score=edge_score,
            uncertainty=states.uncertainty,
            crash_risk=max(states.crash_risk, features.crash_risk),
            realized_volatility=features.realized_volatility,
        ))
        return QuantumEdgeSnapshot(
            edge_score=edge_score,
            target_weight=target_weight,
            features=features,
            states=states,
        )

    def _calculate_edge_score(
        self,
        features: MarketFeatureSnapshot,
        states: MarketStateProbabilities,
    ) -> float:
        alpha_quality = (
            0.26 * features.trend_quality +
            0.20 * features.relative_strength +
            0.18 * features.volume_pressure +
            0.18 * features.volatility_compression +
            0.18 * features.mean_reversion_asymmetry
        )
        constructive_state = (
            0.48 * states.trend_up +
            0.28 * states.volatility_breakout +
            0.18 * states.mean_reversion +
            0.06 * states.neutral
        )
        risk_drag = (
            0.45 * states.crash_risk +
            0.35 * features.crash_risk +
            0.20 * states.uncertainty
        )
        edge = 0.72 * alpha_quality + 0.55 * constructive_state - 0.30 * risk_drag
        return float(np.clip(edge, 0.0, 1.0))

    @staticmethod
    def _format_reason(edge: QuantumEdgeSnapshot) -> str:
        states = edge.states
        features = edge.features
        return (
            f"edge={edge.edge_score:.2f}, weight={edge.target_weight:.2f}, "
            f"trend={states.trend_up:.2f}, breakout={states.volatility_breakout:.2f}, "
            f"meanrev={states.mean_reversion:.2f}, crash={states.crash_risk:.2f}, "
            f"uncertainty={states.uncertainty:.2f}, "
            f"trendQ={features.trend_quality:.2f}, volComp={features.volatility_compression:.2f}"
        )
