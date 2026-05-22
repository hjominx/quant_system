"""
시장 관측 피처 계산

QPE(Quantum Probability Engine)의 첫 단계는 시장을 단정하지 않고
관측 가능한 값들을 0~1 범위의 안정적인 피처로 바꾸는 것이다.

모든 계산은 전달받은 OHLCV의 마지막 시점까지의 데이터만 사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class MarketFeatureSnapshot:
    """단일 시점의 정규화된 시장 피처"""
    timestamp: pd.Timestamp
    trend_quality: float
    volatility_compression: float
    relative_strength: float
    volume_pressure: float
    mean_reversion_asymmetry: float
    crash_risk: float
    realized_volatility: float
    drawdown_60d: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "trend_quality": self.trend_quality,
            "volatility_compression": self.volatility_compression,
            "relative_strength": self.relative_strength,
            "volume_pressure": self.volume_pressure,
            "mean_reversion_asymmetry": self.mean_reversion_asymmetry,
            "crash_risk": self.crash_risk,
            "realized_volatility": self.realized_volatility,
            "drawdown_60d": self.drawdown_60d,
        }


class MarketFeatureExtractor:
    """OHLCV에서 QPE 입력 피처를 계산"""

    def __init__(self, params: Dict | None = None):
        params = params or {}
        self.fast_window = int(params.get("fast_window", 20))
        self.slow_window = int(params.get("slow_window", 60))
        self.regime_window = int(params.get("regime_window", 120))
        self.volume_window = int(params.get("volume_window", 20))
        self.rs_window = int(params.get("relative_strength_window", 252))

    @property
    def min_history(self) -> int:
        return max(self.regime_window, self.rs_window, self.slow_window) + 5

    def extract(self, ohlcv: pd.DataFrame) -> MarketFeatureSnapshot:
        if len(ohlcv) < self.min_history:
            raise ValueError(
                f"QPE 피처 계산에 데이터가 부족합니다. 최소 {self.min_history}봉 필요, 현재 {len(ohlcv)}봉"
            )

        close = ohlcv["close"].astype(float)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)
        volume = ohlcv["volume"].astype(float)
        returns = close.pct_change().fillna(0.0)

        fast_ret = close.pct_change(self.fast_window).iloc[-1]
        slow_ret = close.pct_change(self.slow_window).iloc[-1]

        fast_vol = returns.rolling(self.fast_window).std().iloc[-1] * np.sqrt(365)
        slow_vol = returns.rolling(self.slow_window).std().iloc[-1] * np.sqrt(365)
        regime_vol = returns.rolling(self.regime_window).std().iloc[-1] * np.sqrt(365)
        realized_vol = float(_safe_number(slow_vol))

        sma_fast = close.rolling(self.fast_window).mean().iloc[-1]
        sma_slow = close.rolling(self.slow_window).mean().iloc[-1]
        sma_spread = (sma_fast / max(sma_slow, EPS)) - 1
        trend_consistency = (returns.tail(self.fast_window) > 0).mean()

        momentum_score = _sigmoid(4.0 * _safe_number(slow_ret) / max(realized_vol, 0.05))
        spread_score = _sigmoid(20.0 * _safe_number(sma_spread))
        trend_quality = _clip01(
            0.45 * momentum_score +
            0.35 * spread_score +
            0.20 * trend_consistency
        )

        vol_ratio = _safe_number(fast_vol) / max(_safe_number(regime_vol), 0.05)
        volatility_compression = _clip01(1.0 - (vol_ratio / 1.35))

        rs_series = close.pct_change(self.slow_window).dropna().tail(self.rs_window)
        current_rs = rs_series.iloc[-1]
        relative_strength = float((rs_series <= current_rs).mean()) if len(rs_series) > 0 else 0.5

        signed_volume = np.sign(returns.tail(self.volume_window)) * volume.tail(self.volume_window)
        pressure = signed_volume.sum() / max(volume.tail(self.volume_window).sum(), EPS)
        volume_ma = volume.rolling(self.volume_window).mean().iloc[-1]
        volume_expansion = (volume.iloc[-1] / max(volume_ma, EPS)) - 1
        volume_pressure = _clip01(
            0.70 * _sigmoid(3.0 * _safe_number(pressure)) +
            0.30 * _sigmoid(_safe_number(volume_expansion))
        )

        mean = close.rolling(self.slow_window).mean().iloc[-1]
        std = close.rolling(self.slow_window).std().iloc[-1]
        z_score = (close.iloc[-1] - mean) / max(std, EPS)
        # 과매도(-1 부근)이면서 구조가 무너지지 않은 구간을 높게 평가한다.
        mean_reversion_shape = np.exp(-((z_score + 1.0) ** 2) / 2.0)
        mean_reversion_asymmetry = _clip01(mean_reversion_shape * (0.55 + 0.45 * trend_quality))

        rolling_high = close.rolling(self.slow_window).max().iloc[-1]
        drawdown_60d = (close.iloc[-1] / max(rolling_high, EPS)) - 1
        downside_speed = -close.pct_change(7).iloc[-1]
        range_pct = ((high - low) / close).rolling(7).mean().iloc[-1]
        base_range = ((high - low) / close).rolling(self.slow_window).mean().iloc[-1]
        range_spike = range_pct / max(base_range, EPS)
        vol_spike = _safe_number(fast_vol) / max(_safe_number(slow_vol), 0.05)

        drawdown_component = _clip01(abs(min(_safe_number(drawdown_60d), 0.0)) / 0.35)
        downside_component = _clip01(max(_safe_number(downside_speed), 0.0) / 0.16)
        vol_spike_component = _clip01((vol_spike - 1.0) / 1.75)
        range_spike_component = _clip01((range_spike - 1.0) / 1.75)

        crash_risk = _clip01(
            0.38 * drawdown_component +
            0.32 * downside_component +
            0.18 * vol_spike_component +
            0.12 * range_spike_component
        )

        return MarketFeatureSnapshot(
            timestamp=ohlcv.index[-1],
            trend_quality=trend_quality,
            volatility_compression=volatility_compression,
            relative_strength=_clip01(relative_strength),
            volume_pressure=volume_pressure,
            mean_reversion_asymmetry=mean_reversion_asymmetry,
            crash_risk=crash_risk,
            realized_volatility=realized_vol,
            drawdown_60d=float(_safe_number(drawdown_60d)),
        )


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -50, 50))
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(x: float) -> float:
    return float(np.clip(_safe_number(x), 0.0, 1.0))


def _safe_number(x: float) -> float:
    if pd.isna(x) or not np.isfinite(x):
        return 0.0
    return float(x)
