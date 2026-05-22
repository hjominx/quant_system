"""
양자역학적 시장상태 확률 엔진

시장을 하나의 상태로 단정하지 않고 여러 상태의 확률 벡터로 표현한다.
QPE에서는 이 확률 벡터를 기반으로 전략 선택과 포지션 크기를 결정한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from features.market_features import MarketFeatureSnapshot


@dataclass(frozen=True)
class MarketStateProbabilities:
    """시장 상태 확률 벡터"""
    trend_up: float
    mean_reversion: float
    volatility_breakout: float
    crash_risk: float
    neutral: float
    uncertainty: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "trend_up": self.trend_up,
            "mean_reversion": self.mean_reversion,
            "volatility_breakout": self.volatility_breakout,
            "crash_risk": self.crash_risk,
            "neutral": self.neutral,
            "uncertainty": self.uncertainty,
        }


class QuantumMarketStateEngine:
    """정규화된 피처를 시장 상태 확률로 변환"""

    def infer(self, features: MarketFeatureSnapshot) -> MarketStateProbabilities:
        f = features

        trend_raw = (
            0.45 * f.trend_quality +
            0.25 * f.relative_strength +
            0.15 * f.volume_pressure +
            0.15 * (1.0 - f.crash_risk)
        )
        mean_reversion_raw = (
            0.45 * f.mean_reversion_asymmetry +
            0.25 * f.volatility_compression +
            0.20 * (1.0 - f.crash_risk) +
            0.10 * (1.0 - f.trend_quality)
        )
        breakout_raw = (
            0.40 * f.volatility_compression +
            0.25 * f.volume_pressure +
            0.20 * f.trend_quality +
            0.15 * f.relative_strength
        )
        crash_raw = (
            0.70 * f.crash_risk +
            0.20 * (1.0 - f.trend_quality) +
            0.10 * (1.0 - f.volume_pressure)
        )
        neutral_raw = (
            0.25 +
            0.25 * (1.0 - abs(f.trend_quality - 0.5) * 2.0) +
            0.25 * (1.0 - abs(f.relative_strength - 0.5) * 2.0) +
            0.25 * (1.0 - f.crash_risk)
        )

        raw = np.array([
            max(trend_raw, 0.0),
            max(mean_reversion_raw, 0.0),
            max(breakout_raw, 0.0),
            max(crash_raw, 0.0),
            max(neutral_raw, 0.0),
        ], dtype=float)

        # 상태 확률이 과도하게 균등해지면 모든 구간이 "불확실"로만 보인다.
        # power sharpening은 가장 강한 상태를 조금 더 선명하게 만들어
        # 확률 벡터가 실제 의사결정에 쓰이도록 한다.
        raw = np.power(raw + 1e-12, 1.45)

        total = raw.sum()
        if total <= 0:
            probs = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        else:
            probs = raw / total

        entropy = -np.sum(probs * np.log(probs + 1e-12))
        uncertainty = float(entropy / np.log(len(probs)))

        return MarketStateProbabilities(
            trend_up=float(probs[0]),
            mean_reversion=float(probs[1]),
            volatility_breakout=float(probs[2]),
            crash_risk=float(probs[3]),
            neutral=float(probs[4]),
            uncertainty=uncertainty,
        )
