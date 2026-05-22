"""
확률 기반 포지션 사이징

QPE의 "붕괴" 단계:
EdgeScore, 불확실성, 급락위험을 실제 목표 비중으로 변환한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionSizingInput:
    edge_score: float
    uncertainty: float
    crash_risk: float
    realized_volatility: float


class QuantumPositionSizer:
    """확률/위험 기반 목표 비중 계산기"""

    def __init__(
        self,
        max_position: float = 0.90,
        min_position: float = 0.10,
        target_volatility: float = 0.35,
        uncertainty_penalty: float = 0.65,
        crash_penalty: float = 0.85,
    ):
        self.max_position = max_position
        self.min_position = min_position
        self.target_volatility = target_volatility
        self.uncertainty_penalty = uncertainty_penalty
        self.crash_penalty = crash_penalty
        self._validate()

    def size(self, sizing_input: PositionSizingInput) -> float:
        edge = _clip01(sizing_input.edge_score)
        uncertainty = _clip01(sizing_input.uncertainty)
        crash = _clip01(sizing_input.crash_risk)
        realized_vol = max(float(sizing_input.realized_volatility), 1e-6)

        vol_scalar = np.clip(self.target_volatility / realized_vol, 0.25, 1.25)
        risk_scalar = (
            (1.0 - self.uncertainty_penalty * uncertainty) *
            (1.0 - self.crash_penalty * crash)
        )
        weight = self.max_position * edge * vol_scalar * max(risk_scalar, 0.0)

        if weight < self.min_position:
            return 0.0
        return float(np.clip(weight, 0.0, self.max_position))

    def _validate(self) -> None:
        if not (0 < self.min_position <= self.max_position <= 1.0):
            raise ValueError("포지션 한도는 0 < min_position <= max_position <= 1 이어야 합니다.")
        if self.target_volatility <= 0:
            raise ValueError("target_volatility는 양수여야 합니다.")
        if self.uncertainty_penalty < 0 or self.crash_penalty < 0:
            raise ValueError("위험 페널티는 음수일 수 없습니다.")


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
