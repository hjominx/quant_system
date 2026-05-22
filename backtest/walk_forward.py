"""
Walk-Forward 검증 엔진

목적:
1. 과최적화 방지: 학습 구간에서 고른 파라미터를 다음 미사용 구간에만 평가
2. Out-of-Sample 성과 집중: 각 검증 구간의 성과만 이어 붙여 최종 판단
3. 전략 독립성: BaseStrategy 팩토리와 파라미터 그리드만 주면 재사용 가능

기본 흐름:
    2021 데이터로 최적화 -> 2022 데이터 검증
    2021~2022 데이터로 최적화 -> 2023 데이터 검증
    ...
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics, calculate_metrics
from backtest.objectives import ObjectiveResult, calculate_objective
from strategies.base import BaseStrategy


StrategyFactory = Callable[[Dict], BaseStrategy]


@dataclass
class WalkForwardConfig:
    """Walk-Forward 검증 설정"""
    initial_train_years: int = 1
    test_years: int = 1
    step_years: int = 1
    mode: str = "expanding"  # expanding 또는 rolling
    objective: str = "sharpe_ratio"
    min_train_bars: int = 200
    min_test_bars: int = 60
    allow_partial_final_window: bool = True

    # Out-of-Sample 최소 합격 기준
    min_oos_sharpe: float = 1.0
    min_oos_calmar: float = 1.5
    max_oos_mdd: float = -0.20

    def __post_init__(self) -> None:
        if self.initial_train_years <= 0:
            raise ValueError("initial_train_years는 1 이상이어야 합니다.")
        if self.test_years <= 0:
            raise ValueError("test_years는 1 이상이어야 합니다.")
        if self.step_years <= 0:
            raise ValueError("step_years는 1 이상이어야 합니다.")
        if self.mode not in ("expanding", "rolling"):
            raise ValueError("mode는 'expanding' 또는 'rolling'이어야 합니다.")


@dataclass
class WalkForwardWindowResult:
    """단일 Walk-Forward 윈도우 결과"""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int
    best_params: Dict
    objective_value: float
    train_result: BacktestResult
    test_result: BacktestResult

    @property
    def oos_passed(self) -> bool:
        return self.test_result.metrics.is_acceptable()


@dataclass
class WalkForwardResult:
    """Walk-Forward 전체 결과"""
    windows: List[WalkForwardWindowResult]
    oos_equity_curve: pd.Series
    oos_trades: pd.DataFrame
    oos_metrics: PerformanceMetrics
    optimization_log: pd.DataFrame
    config: WalkForwardConfig

    @property
    def pass_rate(self) -> float:
        if not self.windows:
            return 0.0
        return sum(self.window_passed(w) for w in self.windows) / len(self.windows)

    def window_passed(self, window: WalkForwardWindowResult) -> bool:
        """설정된 OOS 기준으로 단일 윈도우 통과 여부 판단"""
        return window.test_result.metrics.is_acceptable(
            min_sharpe=self.config.min_oos_sharpe,
            min_calmar=self.config.min_oos_calmar,
            max_mdd=self.config.max_oos_mdd,
        )

    def windows_dataframe(self) -> pd.DataFrame:
        """윈도우별 핵심 성과 요약"""
        rows = []
        for w in self.windows:
            train_m = w.train_result.metrics
            test_m = w.test_result.metrics
            rows.append({
                "window_id": w.window_id,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "train_rows": w.train_rows,
                "test_rows": w.test_rows,
                "best_params": w.best_params,
                "objective_value": w.objective_value,
                "train_return": train_m.total_return,
                "train_sharpe": train_m.sharpe_ratio,
                "train_calmar": train_m.calmar_ratio,
                "oos_return": test_m.total_return,
                "oos_mdd": test_m.max_drawdown,
                "oos_sharpe": test_m.sharpe_ratio,
                "oos_calmar": test_m.calmar_ratio,
                "oos_trades": test_m.total_trades,
                "oos_passed": self.window_passed(w),
            })
        return pd.DataFrame(rows)

    def selected_params_dataframe(self) -> pd.DataFrame:
        """윈도우별 선택 파라미터를 DataFrame으로 반환"""
        rows = []
        for w in self.windows:
            row = {"window_id": w.window_id}
            row.update(w.best_params)
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        m = self.oos_metrics
        return f"""
=== Walk-Forward Out-of-Sample 요약 ===
윈도우 수:        {len(self.windows)}
윈도우 통과율:    {self.pass_rate * 100:>7.2f}%

[OOS 수익성]
누적 수익률:      {m.total_return * 100:>7.2f}%
연환산 수익률:    {m.annualized_return * 100:>7.2f}%

[OOS 위험]
MDD:             {m.max_drawdown * 100:>7.2f}%
변동성:          {m.volatility * 100:>7.2f}%

[OOS 위험조정수익]
Sharpe:          {m.sharpe_ratio:>7.2f}
Sortino:         {m.sortino_ratio:>7.2f}
Calmar:          {m.calmar_ratio:>7.2f}

[OOS 매매]
총 거래수:        {m.total_trades:>7d}
승률:             {m.win_rate * 100:>7.2f}%
손익비:           {m.profit_factor:>7.2f}
"""


class WalkForwardEngine:
    """단일 자산 Walk-Forward 검증 엔진"""

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        backtest_config: Optional[BacktestConfig] = None,
        config: Optional[WalkForwardConfig] = None,
    ):
        self.strategy_factory = strategy_factory
        self.backtest_config = backtest_config or BacktestConfig()
        self.config = config or WalkForwardConfig()

    def run(
        self,
        ohlcv: pd.DataFrame,
        ticker: str,
        parameter_grid: Sequence[Dict] | Dict[str, Sequence],
    ) -> WalkForwardResult:
        """Walk-Forward 검증 실행"""
        param_sets = self._expand_parameter_grid(parameter_grid)
        if not param_sets:
            raise ValueError("parameter_grid가 비어 있습니다.")
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise ValueError("OHLCV 인덱스는 DatetimeIndex여야 합니다.")
        if not ohlcv.index.is_monotonic_increasing:
            raise ValueError("OHLCV 인덱스가 시간순으로 정렬되지 않았습니다.")

        windows: List[WalkForwardWindowResult] = []
        optimization_rows = []

        for window_id, train_df, test_df in self._iter_windows(ohlcv):
            best_params, train_result, objective_value, window_opt_rows = self._optimize_window(
                train_df=train_df,
                ticker=ticker,
                param_sets=param_sets,
                window_id=window_id,
            )
            optimization_rows.extend(window_opt_rows)

            test_result = self._run_oos_window(
                train_df=train_df,
                test_df=test_df,
                ticker=ticker,
                params=best_params,
            )

            windows.append(WalkForwardWindowResult(
                window_id=window_id,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                train_rows=len(train_df),
                test_rows=len(test_df),
                best_params=best_params,
                objective_value=objective_value,
                train_result=train_result,
                test_result=test_result,
            ))

        if not windows:
            raise ValueError("생성된 Walk-Forward 윈도우가 없습니다. 기간/최소 봉 수 설정을 확인하세요.")

        oos_equity_curve = self._stitch_oos_equity(windows)
        oos_trades = self._combine_oos_trades(windows)
        oos_metrics = calculate_metrics(oos_equity_curve, oos_trades)

        return WalkForwardResult(
            windows=windows,
            oos_equity_curve=oos_equity_curve,
            oos_trades=oos_trades,
            oos_metrics=oos_metrics,
            optimization_log=pd.DataFrame(optimization_rows),
            config=self.config,
        )

    def _iter_windows(self, ohlcv: pd.DataFrame):
        cfg = self.config
        first = ohlcv.index[0]
        last = ohlcv.index[-1]
        step = pd.DateOffset(years=cfg.step_years)

        train_start = first
        train_end_exclusive = first + pd.DateOffset(years=cfg.initial_train_years)
        window_id = 1

        while True:
            test_start = train_end_exclusive
            test_end_exclusive = test_start + pd.DateOffset(years=cfg.test_years)

            if test_start > last:
                break
            if not cfg.allow_partial_final_window and test_end_exclusive > last:
                break

            train_df = ohlcv[(ohlcv.index >= train_start) & (ohlcv.index < train_end_exclusive)]
            test_df = ohlcv[(ohlcv.index >= test_start) & (ohlcv.index < test_end_exclusive)]

            if len(train_df) >= cfg.min_train_bars and len(test_df) >= cfg.min_test_bars:
                yield window_id, train_df, test_df
                window_id += 1

            if cfg.mode == "expanding":
                train_end_exclusive = train_end_exclusive + step
            else:
                train_start = train_start + step
                train_end_exclusive = train_start + pd.DateOffset(years=cfg.initial_train_years)

    def _optimize_window(
        self,
        train_df: pd.DataFrame,
        ticker: str,
        param_sets: List[Dict],
        window_id: int,
    ):
        best_params = None
        best_result = None
        best_score = -np.inf
        rows = []

        for params in param_sets:
            try:
                strategy = self.strategy_factory(dict(params))
                result = BacktestEngine(self.backtest_config).run(strategy, train_df, ticker)
                objective_result = self._objective_value(result.metrics)
                score = objective_result.score
                objective_details = objective_result.details
                status = "ok"
            except Exception as e:
                result = None
                score = np.nan
                objective_details = {}
                status = f"error: {type(e).__name__}: {e}"

            row = {
                "window_id": window_id,
                "params": dict(params),
                "objective": self.config.objective,
                "objective_value": score,
                "status": status,
            }
            row.update(objective_details)
            rows.append(row)

            if result is not None and np.isfinite(score) and score > best_score:
                best_score = score
                best_params = dict(params)
                best_result = result

        if best_params is None or best_result is None:
            raise ValueError(
                f"윈도우 {window_id}에서 유효한 파라미터를 찾지 못했습니다."
            )

        return best_params, best_result, best_score, rows

    def _run_oos_window(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        ticker: str,
        params: Dict,
    ) -> BacktestResult:
        strategy = self.strategy_factory(dict(params))
        warmup = strategy.warmup_period()
        if len(train_df) < warmup:
            raise ValueError(
                f"OOS 실행에 필요한 warmup 부족: 필요 {warmup}, 학습 구간 {len(train_df)}"
            )

        # 학습 구간의 마지막 warmup 봉만 컨텍스트로 제공한다.
        # BacktestEngine은 warmup 이후부터 equity를 기록하므로 성과는 test_df만 반영된다.
        oos_input = pd.concat([train_df.tail(warmup), test_df])
        return BacktestEngine(self.backtest_config).run(strategy, oos_input, ticker)

    def _objective_value(self, metrics: PerformanceMetrics) -> ObjectiveResult:
        return calculate_objective(metrics, self.config.objective)

    def _stitch_oos_equity(self, windows: List[WalkForwardWindowResult]) -> pd.Series:
        curves = []
        current_equity = self.backtest_config.initial_capital

        for window in windows:
            curve = window.test_result.equity_curve.copy()
            if curve.empty:
                continue
            scaled = (curve / curve.iloc[0]) * current_equity
            current_equity = scaled.iloc[-1]
            curves.append(scaled)

        if not curves:
            raise ValueError("이어 붙일 OOS 자산곡선이 없습니다.")

        stitched = pd.concat(curves)
        stitched = stitched[~stitched.index.duplicated(keep="last")]
        stitched = stitched.sort_index()
        stitched.name = "equity"
        return stitched

    @staticmethod
    def _combine_oos_trades(windows: List[WalkForwardWindowResult]) -> pd.DataFrame:
        frames = []
        for window in windows:
            trades = window.test_result.trades.copy()
            if trades.empty:
                continue
            trades.insert(0, "window_id", window.window_id)
            trades["best_params"] = [dict(window.best_params)] * len(trades)
            frames.append(trades)

        if not frames:
            return pd.DataFrame(columns=[
                "window_id", "entry_time", "exit_time", "ticker", "entry_price",
                "exit_price", "quantity", "pnl", "pnl_pct", "entry_reason",
                "exit_reason", "best_params",
            ])

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _expand_parameter_grid(parameter_grid: Sequence[Dict] | Dict[str, Sequence]) -> List[Dict]:
        if isinstance(parameter_grid, dict):
            keys = list(parameter_grid.keys())
            values = [list(parameter_grid[k]) for k in keys]
            if not keys or any(len(v) == 0 for v in values):
                return []
            return [dict(zip(keys, combo)) for combo in product(*values)]

        return [dict(params) for params in parameter_grid]
