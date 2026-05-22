"""
ETH/Cash Regime Switch 검증 스크립트

목적:
- 2024년 말 이후 ETH 단순보유가 깨진 구간에서 cash 전환 필터가 유효했는지 확인한다.
- 같은 고정 파라미터를 전체 5년 Walk-Forward에도 넣어 최근 구간 착시를 분리한다.

실행:
    python scripts/walk_forward_eth_cash_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestConfig, BacktestEngine
from backtest.metrics import calculate_metrics
from backtest.walk_forward import WalkForwardConfig, WalkForwardEngine
from data.loader import get_cache_summary, load_for_backtest
from strategies.eth_cash_regime import ETHCashRegimeStrategy
from utils.logger import setup_logging


DEFAULT_PARAMS = {
    "fast_window": 20,
    "slow_window": 180,
    "exit_window": 60,
    "momentum_window": 30,
    "momentum_min": 0.10,
    "rsi_min": 44,
    "rsi_max": 92,
    "max_entry_volatility": 1.30,
    "target_volatility": 0.38,
    "trailing_stop": 0.32,
    "breakout_window": 60,
    "breakout_tolerance": 0.08,
    "crash_5d": -0.20,
    "crash_10d": -0.25,
    "max_position": 1.00,
    "min_position": 0.25,
}

RECENT_STARTS = [
    "2024-09-01",
    "2024-10-01",
    "2024-11-01",
    "2024-12-01",
    "2025-01-01",
]


def make_strategy(params=None):
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    return ETHCashRegimeStrategy(params=merged)


def buy_and_hold_metrics(df: pd.DataFrame, start: str):
    start_ts = pd.Timestamp(start)
    sliced = df[df.index >= start_ts]
    equity = sliced["close"] / sliced["close"].iloc[0] * 10_000_000
    return calculate_metrics(equity.rename("equity"), pd.DataFrame())


def run_strategy_from_start(df: pd.DataFrame, start: str):
    start_ts = pd.Timestamp(start)
    strategy = make_strategy()
    history = df[df.index < start_ts].tail(strategy.warmup_period())
    live_slice = df[df.index >= start_ts]
    input_df = pd.concat([history, live_slice]).copy()

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000_000,
            commission_rate=0.0025,
            slippage_rate=0.002,
        )
    ).run(
        strategy,
        input_df,
        ticker="KRW-ETH",
    )
    equity = result.equity_curve[result.equity_curve.index >= start_ts]
    trades = result.trades.copy()
    if not trades.empty:
        trades = trades[trades["exit_time"] >= start_ts]
    return calculate_metrics(equity, trades), trades


def run_recent_fixed_study(df: pd.DataFrame):
    print(f"\n{'='*96}")
    print(" ETH/Cash Regime — 2024년 말 이후 고정 파라미터 검증")
    print(f"{'='*96}\n")
    print(
        f"{'시작일':<12} {'모델수익':>10} {'모델MDD':>9} {'모델Sharpe':>11} "
        f"{'B&H수익':>10} {'B&HMDD':>9} {'B&HSharpe':>10} {'거래수':>7}"
    )
    print("─" * 96)

    for start in RECENT_STARTS:
        model_m, trades = run_strategy_from_start(df, start)
        hold_m = buy_and_hold_metrics(df, start)
        print(
            f"{start:<12} "
            f"{model_m.total_return*100:>+9.2f}% "
            f"{model_m.max_drawdown*100:>+8.2f}% "
            f"{model_m.sharpe_ratio:>11.2f} "
            f"{hold_m.total_return*100:>+9.2f}% "
            f"{hold_m.max_drawdown*100:>+8.2f}% "
            f"{hold_m.sharpe_ratio:>10.2f} "
            f"{len(trades):>7d}"
        )

    print("\n[주의]")
    print("  - 위 표는 최근 레짐 고정 파라미터 검증입니다.")
    print("  - 전체 5년 Walk-Forward OOS와 분리해서 봐야 합니다.")


def run_fixed_walk_forward(df: pd.DataFrame):
    print(f"\n{'='*96}")
    print(" ETH/Cash Regime — 전체 기간 고정 파라미터 Walk-Forward")
    print(f"{'='*96}\n")

    wf_config = WalkForwardConfig(
        initial_train_years=1,
        test_years=1,
        step_years=1,
        mode="expanding",
        objective="qpe_composite_score",
        min_train_bars=300,
        min_test_bars=180,
        allow_partial_final_window=True,
    )
    backtest_config = BacktestConfig(
        initial_capital=10_000_000,
        commission_rate=0.0025,
        slippage_rate=0.002,
    )
    engine = WalkForwardEngine(
        strategy_factory=make_strategy,
        backtest_config=backtest_config,
        config=wf_config,
    )
    result = engine.run(df, ticker="KRW-ETH", parameter_grid=[DEFAULT_PARAMS])

    print(result.summary())
    print("\n[윈도우별 OOS]")
    print(f"{'ID':>2} {'Test':<23} {'OOS Ret':>9} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Trades':>6}")
    print("─" * 72)
    for _, row in result.windows_dataframe().iterrows():
        test_range = f"{row['test_start'].date()}~{row['test_end'].date()}"
        print(
            f"{int(row['window_id']):>2} {test_range:<23} "
            f"{row['oos_return']*100:>+8.2f}% "
            f"{row['oos_mdd']*100:>+7.2f}% "
            f"{row['oos_sharpe']:>7.2f} "
            f"{row['oos_calmar']:>7.2f} "
            f"{int(row['oos_trades']):>6d}"
        )
    return result


def main():
    setup_logging(log_level="WARNING", enable_file_log=False)
    print("=" * 96)
    print(" ETH/Cash Regime Switch — 최근 레짐 연구")
    print("=" * 96)

    summary = get_cache_summary()
    if summary.empty:
        print("✗ 캐시 없음. 먼저 scripts/download_historical.py 실행 필요.")
        sys.exit(1)

    df = load_for_backtest("KRW-ETH", "1d")
    print(f"데이터 로드: {len(df)}봉 ({df.index[0].date()} ~ {df.index[-1].date()})")

    run_recent_fixed_study(df)
    run_fixed_walk_forward(df)

    print("\n해석:")
    print("  - 최근 구간에서는 ETH/cash 전환이 QPE Trend Overlay보다 높은 Sharpe 후보를 만들었습니다.")
    print("  - 하지만 전체 5년 OOS 기준에서는 아직 최종 실전 전략이 아닙니다.")
    print("  - 다음 단계는 이 필터를 포트폴리오 레벨 리스크 게이트로 쓰는지 검증하는 것입니다.")


if __name__ == "__main__":
    main()
