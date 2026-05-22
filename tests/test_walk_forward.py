"""
Walk-Forward 검증 엔진 테스트

검증 시나리오 6가지:
1. Expanding 방식 Walk-Forward 실행
2. Rolling 방식에서 학습 구간 시작점이 이동하는지 확인
3. OOS 자산곡선이 윈도우별로 이어 붙는지 확인
4. 잘못된 파라미터 후보를 스킵하는지 확인
5. 복합 objective가 최적화 로그에 기록되는지 확인
6. 빈 파라미터 그리드를 거부하는지 확인
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig
from backtest.walk_forward import WalkForwardConfig, WalkForwardEngine
from strategies.sma_crossover import SMACrossoverStrategy


def make_test_ohlcv(n_days: int = 1100, seed: int = 7) -> pd.DataFrame:
    """Walk-Forward 테스트용 합성 OHLCV"""
    np.random.seed(seed)
    t = np.arange(n_days)
    cycle = 0.0018 * np.sin(2 * np.pi * t / 160)
    noise = np.random.normal(0, 0.015, n_days)
    drift = 0.0002
    log_returns = drift + cycle + noise
    close = 100_000 * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.random.uniform(1000, 5000, n_days)

    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)


def make_strategy(params):
    return SMACrossoverStrategy(params=params)


def make_engine(mode: str = 'expanding') -> WalkForwardEngine:
    return WalkForwardEngine(
        strategy_factory=make_strategy,
        backtest_config=BacktestConfig(
            initial_capital=10_000_000,
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
        config=WalkForwardConfig(
            initial_train_years=1,
            test_years=1,
            step_years=1,
            mode=mode,
            min_train_bars=250,
            min_test_bars=120,
            objective='sharpe_ratio',
            allow_partial_final_window=True,
        ),
    )


PARAM_GRID = {
    'short_window': [10, 20],
    'long_window': [40, 60],
    'target_weight': [0.8],
}


def test_expanding_walk_forward_runs():
    print("\n[테스트 1] Expanding Walk-Forward 실행")
    print("-" * 60)

    result = make_engine('expanding').run(
        make_test_ohlcv(),
        ticker='TEST',
        parameter_grid=PARAM_GRID,
    )

    assert len(result.windows) >= 2
    assert not result.windows_dataframe().empty
    assert not result.optimization_log.empty

    print(f"  ✓ 윈도우 수: {len(result.windows)}")
    print(f"  ✓ 최적화 로그: {len(result.optimization_log)}행")
    print(f"  ✓ OOS Sharpe: {result.oos_metrics.sharpe_ratio:.2f}")


def test_rolling_windows_advance_train_start():
    print("\n[테스트 2] Rolling 학습 구간 이동")
    print("-" * 60)

    result = make_engine('rolling').run(
        make_test_ohlcv(),
        ticker='TEST',
        parameter_grid=PARAM_GRID,
    )

    assert len(result.windows) >= 2
    assert result.windows[1].train_start > result.windows[0].train_start

    print(f"  ✓ 첫 학습 시작: {result.windows[0].train_start.date()}")
    print(f"  ✓ 둘째 학습 시작: {result.windows[1].train_start.date()}")


def test_oos_equity_is_stitched():
    print("\n[테스트 3] OOS 자산곡선 연결")
    print("-" * 60)

    engine = make_engine('expanding')
    result = engine.run(make_test_ohlcv(), ticker='TEST', parameter_grid=PARAM_GRID)

    assert len(result.oos_equity_curve) > 0
    assert abs(result.oos_equity_curve.iloc[0] - engine.backtest_config.initial_capital) < 1.0
    assert result.oos_equity_curve.index.is_monotonic_increasing

    print(f"  ✓ OOS 자산곡선 길이: {len(result.oos_equity_curve)}")
    print(f"  ✓ 시작 자산: {result.oos_equity_curve.iloc[0]:,.0f}원")
    print(f"  ✓ 종료 자산: {result.oos_equity_curve.iloc[-1]:,.0f}원")


def test_invalid_parameters_are_skipped():
    print("\n[테스트 4] 잘못된 파라미터 후보 스킵")
    print("-" * 60)

    param_grid = [
        {'short_window': 60, 'long_window': 20, 'target_weight': 0.8},  # invalid
        {'short_window': 10, 'long_window': 40, 'target_weight': 0.8},
    ]
    result = make_engine('expanding').run(
        make_test_ohlcv(),
        ticker='TEST',
        parameter_grid=param_grid,
    )

    error_rows = result.optimization_log[
        result.optimization_log['status'].str.startswith('error')
    ]

    assert len(error_rows) >= 1
    assert len(result.windows) >= 1

    print(f"  ✓ 오류 후보 기록: {len(error_rows)}행")
    print(f"  ✓ 유효 후보로 검증 계속 진행")


def test_composite_objective_supported():
    print("\n[테스트 5] 복합 objective 지원")
    print("-" * 60)

    engine = make_engine('expanding')
    engine.config.objective = 'qpe_composite_score'
    result = engine.run(
        make_test_ohlcv(n_days=760),
        ticker='TEST',
        parameter_grid=PARAM_GRID,
    )

    assert len(result.windows) >= 1
    assert 'objective_sharpe_component' in result.optimization_log.columns
    assert 'objective_turnover_penalty' in result.optimization_log.columns
    assert result.optimization_log['objective_value'].notnull().any()

    best = result.windows_dataframe().iloc[0]
    print(f"  ✓ objective: {engine.config.objective}")
    print(f"  ✓ 선택 점수: {best['objective_value']:.3f}")
    print(f"  ✓ 세부 항목 기록됨")


def test_empty_parameter_grid_rejected():
    print("\n[테스트 6] 빈 파라미터 그리드 거부")
    print("-" * 60)

    try:
        make_engine('expanding').run(
            make_test_ohlcv(),
            ticker='TEST',
            parameter_grid={},
        )
        raise AssertionError("빈 그리드가 통과됨!")
    except ValueError as e:
        assert "parameter_grid" in str(e)

    print("  ✓ 빈 그리드 거부")


def main():
    print("=" * 60)
    print("Walk-Forward 검증 엔진 테스트")
    print("=" * 60)

    tests = [
        test_expanding_walk_forward_runs,
        test_rolling_windows_advance_train_start,
        test_oos_equity_is_stitched,
        test_invalid_parameters_are_skipped,
        test_composite_objective_supported,
        test_empty_parameter_grid_rejected,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 실패: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 예외: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과 / {failed}개 실패 (총 {len(tests)}개)")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
