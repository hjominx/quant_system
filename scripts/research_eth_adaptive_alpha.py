"""Research ETH adaptive alpha presets on recent windows.

This script compares the current aggressive4h live profile, ETH buy-and-hold,
and a stricter indicator stack:
- Donchian breakout pressure
- ADX trend strength
- ATR volatility and trailing stop
- RSI overheat/weakness filter
- explicit fee/slippage edge filter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestConfig, BacktestEngine
from backtest.metrics import calculate_metrics
from data.loader import load_for_backtest
from scripts.run_eth_cash_micro_live import strategy_params_for_profile
from strategies.eth_adaptive_alpha import ETHAdaptiveAlphaStrategy
from strategies.eth_cash_regime import ETHCashRegimeStrategy
from utils.logger import setup_logging


INITIAL_CAPITAL = 10_000_000
PERIODS_PER_YEAR_4H = 365 * 6


PRESETS = {
    "balanced": {
        "fast_ema": 12,
        "mid_ema": 48,
        "long_ema": 144,
        "donchian_window": 72,
        "exit_donchian_window": 30,
        "min_momentum": 0.035,
        "min_adx": 20,
        "entry_score": 0.66,
        "exit_score": 0.40,
        "target_volatility": 0.28,
        "max_position": 0.60,
        "cooldown_bars": 18,
    },
    "strict": {
        "fast_ema": 16,
        "mid_ema": 60,
        "long_ema": 180,
        "donchian_window": 96,
        "exit_donchian_window": 42,
        "min_momentum": 0.050,
        "min_adx": 24,
        "entry_score": 0.72,
        "exit_score": 0.44,
        "target_volatility": 0.24,
        "max_position": 0.50,
        "cooldown_bars": 24,
    },
    "defensive": {
        "fast_ema": 18,
        "mid_ema": 72,
        "long_ema": 216,
        "donchian_window": 120,
        "exit_donchian_window": 36,
        "min_momentum": 0.060,
        "min_adx": 25,
        "entry_score": 0.74,
        "exit_score": 0.46,
        "target_volatility": 0.20,
        "max_position": 0.42,
        "cooldown_bars": 30,
        "max_realized_volatility": 1.35,
    },
    "stable_quality": {
        "fast_ema": 8,
        "mid_ema": 36,
        "long_ema": 120,
        "donchian_window": 54,
        "exit_donchian_window": 24,
        "min_momentum": 0.040,
        "min_adx": 26,
        "entry_score": 0.70,
        "exit_score": 0.42,
        "target_volatility": 0.26,
        "max_position": 0.55,
        "cooldown_bars": 40,
        "max_realized_volatility": 0.90,
        "max_atr_pct": 0.085,
        "breakout_tolerance": 0.030,
        "hard_stop_pct": 0.035,
    },
    "fast_quality": {
        "fast_ema": 8,
        "mid_ema": 36,
        "long_ema": 120,
        "donchian_window": 54,
        "exit_donchian_window": 24,
        "min_momentum": 0.040,
        "min_adx": 22,
        "entry_score": 0.70,
        "exit_score": 0.42,
        "target_volatility": 0.26,
        "max_position": 0.55,
        "cooldown_bars": 20,
    },
}


def run_strategy(df, strategy, *, months: int, commission: float, slippage: float):
    end = df.index[-1]
    start = end - pd.DateOffset(months=months)
    history = df[df.index < start].tail(strategy.warmup_period())
    live = df[df.index >= start]
    input_df = pd.concat([history, live]).copy()
    result = BacktestEngine(
        BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            commission_rate=commission,
            slippage_rate=slippage,
            execution_lag=1,
            use_open_for_execution=True,
        )
    ).run(strategy, input_df, ticker="KRW-ETH")
    equity = result.equity_curve[result.equity_curve.index >= start]
    trades = result.trades.copy()
    if not trades.empty:
        trades = trades[trades["exit_time"] >= start]
    metrics = calculate_metrics(
        equity,
        trades,
        periods_per_year=PERIODS_PER_YEAR_4H,
    )
    return metrics, trades, equity


def buy_and_hold(df, months: int):
    end = df.index[-1]
    start = end - pd.DateOffset(months=months)
    live = df[df.index >= start]
    equity = live["close"] / live["close"].iloc[0] * INITIAL_CAPITAL
    return calculate_metrics(
        equity.rename("equity"),
        pd.DataFrame(),
        periods_per_year=PERIODS_PER_YEAR_4H,
    )


def turnover(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    notional = trades["quantity"] * (
        trades["entry_price"] + trades["exit_price"].fillna(trades["entry_price"])
    )
    return float(notional.sum() / INITIAL_CAPITAL)


def format_row(name: str, months: int, metrics, trades: pd.DataFrame) -> str:
    return (
        f"{name:<18} {months:>3d}m "
        f"{metrics.total_return*100:>+8.2f}% "
        f"{metrics.max_drawdown*100:>+8.2f}% "
        f"{metrics.sharpe_ratio:>7.2f} "
        f"{metrics.sortino_ratio:>7.2f} "
        f"{metrics.calmar_ratio:>7.2f} "
        f"{metrics.total_trades:>6d} "
        f"{metrics.win_rate*100:>6.1f}% "
        f"{metrics.profit_factor:>6.2f} "
        f"{turnover(trades):>7.2f}x"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research ETH adaptive alpha recent-window metrics.")
    parser.add_argument("--market", default="KRW-ETH")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--months", default="18,12,6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_level="WARNING", enable_file_log=False)
    months_list = [int(item.strip()) for item in args.months.split(",") if item.strip()]
    df = load_for_backtest(args.market, args.timeframe)

    print("=" * 118)
    print("ETH adaptive alpha research")
    print("=" * 118)
    print(f"data: {args.market} {args.timeframe} | {len(df)} bars | {df.index[0]} ~ {df.index[-1]}")
    print(f"cost: fee {args.fee*100:.2f}% + slippage {args.slippage*100:.2f}% per side")
    print("fill: signal at 4h close -> next 4h open")
    print("-" * 118)
    print(
        f"{'model':<18} {'win':>4} {'return':>9} {'MDD':>9} "
        f"{'Sharpe':>7} {'Sortino':>7} {'Calmar':>7} "
        f"{'trades':>6} {'win%':>6} {'PF':>6} {'turnover':>8}"
    )
    print("-" * 118)

    for months in months_list:
        bh = buy_and_hold(df, months)
        print(format_row("ETH buy&hold", months, bh, pd.DataFrame()))

        current = ETHCashRegimeStrategy(params=strategy_params_for_profile("aggressive4h"))
        current_m, current_trades, _ = run_strategy(
            df,
            current,
            months=months,
            commission=args.fee,
            slippage=args.slippage,
        )
        print(format_row("current aggr4h", months, current_m, current_trades))

        for name, params in PRESETS.items():
            strategy = ETHAdaptiveAlphaStrategy(params=params)
            metrics, trades, _ = run_strategy(
                df,
                strategy,
                months=months,
                commission=args.fee,
                slippage=args.slippage,
            )
            print(format_row(name, months, metrics, trades))
        print("-" * 118)

    print("\nNotes:")
    print("  - This is research output, not a live-trading approval.")
    print("  - Prefer candidates that improve 18m without collapsing on 12m and 6m.")
    print("  - Turnover should stay far below the current aggressive4h profile.")


if __name__ == "__main__":
    main()
