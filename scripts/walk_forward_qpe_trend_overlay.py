"""
QPE Trend Overlay Walk-Forward 검증

목적:
- SMA의 공격적인 상승장 포착력을 유지한다.
- QPE 시장상태/급락위험 필터로 나쁜 구간 노출을 낮춘다.
- QPE v1과 SMA 기준선 대비 Sharpe 개선 여부를 확인한다.

실행:
    python scripts/walk_forward_qpe_trend_overlay.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging
from data.loader import load_for_backtest, get_cache_summary
from strategies.qpe_trend_overlay import QPETrendOverlayStrategy
from backtest.engine import BacktestConfig
from backtest.walk_forward import WalkForwardConfig, WalkForwardEngine


PARAMETER_GRID = {
    'short_window': [10, 20, 30],
    'long_window': [60, 100, 150],
    'momentum_window': [60],
    'breakout_window': [80],
    'entry_threshold': [0.46, 0.52],
    'exit_threshold': [0.30, 0.38],
    'trend_entry_threshold': [0.50],
    'crash_entry_threshold': [0.50],
    'crash_exit_threshold': [0.62],
    'max_position': [0.95],
    'min_position': [0.08],
    'target_volatility': [0.55],
    'uncertainty_penalty': [0.20],
    'crash_penalty': [0.45],
}


def make_overlay_strategy(params):
    return QPETrendOverlayStrategy(params=params)


def run_single_walk_forward(market: str, timeframe: str = '1d'):
    print(f"\n{'='*96}")
    print(f"  {market} {timeframe} — QPE Trend Overlay Walk-Forward 검증")
    print(f"{'='*96}\n")

    df = load_for_backtest(market, timeframe)
    print(f"데이터 로드: {len(df)}봉 ({df.index[0].date()} ~ {df.index[-1].date()})")

    wf_config = WalkForwardConfig(
        initial_train_years=1,
        test_years=1,
        step_years=1,
        mode='expanding',
        objective='qpe_composite_score',
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
        strategy_factory=make_overlay_strategy,
        backtest_config=backtest_config,
        config=wf_config,
    )
    result = engine.run(df, ticker=market, parameter_grid=PARAMETER_GRID)

    print(result.summary())

    print("\n[윈도우별 OOS 결과]")
    print(
        f"{'ID':>2} {'Train':<23} {'Test':<23} "
        f"{'Best Params':<62} {'Score':>7} {'OOS Ret':>9} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Trades':>6}"
    )
    print("─" * 166)

    for _, row in result.windows_dataframe().iterrows():
        params = row['best_params']
        param_text = (
            f"S{params['short_window']}/L{params['long_window']}/"
            f"E{params['entry_threshold']:.2f}/X{params['exit_threshold']:.2f}/"
            f"CE{params['crash_entry_threshold']:.2f}/CX{params['crash_exit_threshold']:.2f}/"
            f"W{params['max_position']:.2f}"
        )
        train_range = f"{row['train_start'].date()}~{row['train_end'].date()}"
        test_range = f"{row['test_start'].date()}~{row['test_end'].date()}"
        print(
            f"{int(row['window_id']):>2} "
            f"{train_range:<23} {test_range:<23} "
            f"{param_text:<62} "
            f"{row['objective_value']:>7.3f} "
            f"{row['oos_return']*100:>+8.2f}% "
            f"{row['oos_mdd']*100:>+7.2f}% "
            f"{row['oos_sharpe']:>7.2f} "
            f"{row['oos_calmar']:>7.2f} "
            f"{int(row['oos_trades']):>6d}"
        )

    return result


def main():
    setup_logging(log_level='WARNING')

    print("=" * 96)
    print(" QPE Trend Overlay — Sharpe 개선 후보 검증")
    print("=" * 96)

    summary = get_cache_summary()
    if summary.empty:
        print("✗ 캐시 없음. 먼저 scripts/download_historical.py 실행 필요.")
        sys.exit(1)

    targets = [
        ('KRW-BTC', '1d'),
        ('KRW-ETH', '1d'),
    ]

    results = []
    for market, timeframe in targets:
        results.append(run_single_walk_forward(market, timeframe))

    print(f"\n{'='*96}")
    print(" QPE Trend Overlay 종합 OOS 비교")
    print(f"{'='*96}\n")
    print(f"{'종목':<12} {'윈도우':>6} {'OOS 수익':>10} {'OOS MDD':>10} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'거래수':>8} {'통과율':>8}")
    print("─" * 96)

    for (market, _), result in zip(targets, results):
        m = result.oos_metrics
        print(
            f"{market:<12} "
            f"{len(result.windows):>6d} "
            f"{m.total_return*100:>+9.2f}% "
            f"{m.max_drawdown*100:>+9.2f}% "
            f"{m.sharpe_ratio:>8.2f} "
            f"{m.sortino_ratio:>8.2f} "
            f"{m.calmar_ratio:>8.2f} "
            f"{m.total_trades:>8d} "
            f"{result.pass_rate*100:>7.2f}%"
        )

    print("\n해석:")
    print("  - 목표는 SMA의 수익 포착력과 QPE의 위험 차단을 결합하는 것입니다.")
    print("  - Sharpe가 SMA 기준선보다 낮으면 아직 실전 후보가 아닙니다.")
    print("  - 결과가 좋더라도 다종목 포트폴리오 검증 전까지는 연구 후보입니다.")


if __name__ == '__main__':
    main()
