"""
Walk-Forward 최적화 objective 모음

단일 지표 최적화는 전략을 한쪽으로 비틀 수 있다.
QPE composite score는 Sharpe, Sortino, Calmar, 수익률, MDD, 거래 빈도를
함께 보면서 과도한 변동성/과매매/무거래 후보를 벌점 처리한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from backtest.metrics import PerformanceMetrics


@dataclass(frozen=True)
class ObjectiveResult:
    """objective 점수와 진단용 세부 항목"""
    score: float
    details: Dict[str, float]


def calculate_objective(
    metrics: PerformanceMetrics,
    objective: str,
) -> ObjectiveResult:
    """objective 이름에 맞는 최적화 점수 계산"""
    if objective in ("qpe_composite_score", "composite_score"):
        return calculate_qpe_composite_score(metrics)

    if not hasattr(metrics, objective):
        raise ValueError(f"지원하지 않는 objective: {objective}")

    value = float(getattr(metrics, objective))
    return ObjectiveResult(
        score=value,
        details={"objective_metric_value": value},
    )


def calculate_qpe_composite_score(metrics: PerformanceMetrics) -> ObjectiveResult:
    """QPE용 복합 점수

    설계 의도:
    - Sharpe/Sortino/Calmar를 가장 크게 본다.
    - 누적/연환산 수익은 중요하지만, 고변동성 몰빵을 막기 위해 tanh로 눌러준다.
    - MDD가 -20%를 넘어가면 강하게 벌점 처리한다.
    - 거래가 너무 없으면 무의미한 현금 보유 후보로 보고 벌점 처리한다.
    - 거래가 너무 많으면 수수료/슬리피지 민감도가 커지므로 벌점 처리한다.
    """
    years = max(metrics.days / 365.25, 1 / 365.25)
    trades_per_year = metrics.total_trades / years
    profit_factor = _finite_cap(metrics.profit_factor, cap=5.0)

    sharpe_component = _bounded(metrics.sharpe_ratio, target=2.0)
    sortino_component = _bounded(metrics.sortino_ratio, target=2.5)
    calmar_component = _bounded(metrics.calmar_ratio, target=2.0)
    total_return_component = _bounded(metrics.total_return, target=1.0)
    annual_return_component = _bounded(metrics.annualized_return, target=0.50)
    profit_factor_component = _bounded(profit_factor - 1.0, target=2.0)
    win_rate_component = np.clip((metrics.win_rate - 0.45) / 0.25, -1.0, 1.0)

    mdd_abs = abs(min(metrics.max_drawdown, 0.0))
    mdd_penalty = np.clip((mdd_abs - 0.20) / 0.20, 0.0, 2.0)
    avg_drawdown_penalty = np.clip((abs(min(metrics.avg_drawdown, 0.0)) - 0.08) / 0.12, 0.0, 1.5)

    inactivity_penalty = _shortfall_penalty(trades_per_year, minimum=1.5)
    turnover_penalty = _excess_penalty(trades_per_year, maximum=24.0)

    score = (
        0.26 * sharpe_component +
        0.24 * sortino_component +
        0.20 * calmar_component +
        0.10 * total_return_component +
        0.04 * annual_return_component +
        0.08 * profit_factor_component +
        0.04 * win_rate_component -
        0.18 * mdd_penalty -
        0.07 * avg_drawdown_penalty -
        0.12 * inactivity_penalty -
        0.08 * turnover_penalty
    )

    details = {
        "objective_sharpe_component": float(sharpe_component),
        "objective_sortino_component": float(sortino_component),
        "objective_calmar_component": float(calmar_component),
        "objective_total_return_component": float(total_return_component),
        "objective_annual_return_component": float(annual_return_component),
        "objective_profit_factor_component": float(profit_factor_component),
        "objective_win_rate_component": float(win_rate_component),
        "objective_mdd_penalty": float(mdd_penalty),
        "objective_avg_drawdown_penalty": float(avg_drawdown_penalty),
        "objective_inactivity_penalty": float(inactivity_penalty),
        "objective_turnover_penalty": float(turnover_penalty),
        "objective_trades_per_year": float(trades_per_year),
    }

    return ObjectiveResult(score=float(score), details=details)


def _bounded(value: float, target: float) -> float:
    """목표치를 기준으로 점수를 -1~1 근처로 압축"""
    if target <= 0:
        raise ValueError("target은 양수여야 합니다.")
    return float(np.tanh(_safe_number(value) / target))


def _finite_cap(value: float, cap: float) -> float:
    if not np.isfinite(value):
        return cap
    return float(np.clip(value, -cap, cap))


def _shortfall_penalty(value: float, minimum: float) -> float:
    if value >= minimum:
        return 0.0
    return float(np.clip((minimum - value) / minimum, 0.0, 1.0))


def _excess_penalty(value: float, maximum: float) -> float:
    if value <= maximum:
        return 0.0
    return float(np.clip((value - maximum) / maximum, 0.0, 2.0))


def _safe_number(value: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)
