"""
실데이터 Walk-Forward 검증 실행

목적:
- SMA 교차 전략을 기준선으로 삼아 과최적화 방지 검증 흐름 확인
- 학습 구간 최적화 결과가 다음 미사용 구간에서도 유지되는지 확인

실행:
    python scripts/walk_forward_sma.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging
from data.loader import load_for_backtest, get_cache_summary
from strategies.sma_crossover import SMACrossoverStrategy
from backtest.engine import BacktestConfig
from backtest.walk_forward import WalkForwardConfig, WalkForwardEngine


PARAMETER_GRID = {
    'short_window': [10, 20, 30],
    'long_window': [50, 60, 100, 150],
    'target_weight': [0.80, 0.95],
}


def make_sma_strategy(params):
    return SMACrossoverStrategy(params=params)


def run_single_walk_forward(market: str, timeframe: str = '1d'):
    print(f"\n{'='*80}")
    print(f"  {market} {timeframe} — SMA Walk-Forward 검증")
    print(f"{'='*80}\n")

    df = load_for_backtest(market, timeframe)
    print(f"데이터 로드: {len(df)}봉 ({df.index[0].date()} ~ {df.index[-1].date()})")

    wf_config = WalkForwardConfig(
        initial_train_years=1,
        test_years=1,
        step_years=1,
        mode='expanding',
        objective='sharpe_ratio',
        min_train_bars=250,
        min_test_bars=180,
        allow_partial_final_window=True,
    )
    backtest_config = BacktestConfig(
        initial_capital=10_000_000,
        commission_rate=0.0025,
        slippage_rate=0.002,
    )

    engine = WalkForwardEngine(
        strategy_factory=make_sma_strategy,
        backtest_config=backtest_config,
        config=wf_config,
    )
    result = engine.run(df, ticker=market, parameter_grid=PARAMETER_GRID)

    print(result.summary())

    print("\n[윈도우별 OOS 결과]")
    print(
        f"{'ID':>2} {'Train':<23} {'Test':<23} "
        f"{'Best Params':<42} {'OOS Ret':>9} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7}"
    )
    print("─" * 130)

    for _, row in result.windows_dataframe().iterrows():
        params = row['best_params']
        param_text = (
            f"S{params['short_window']}/L{params['long_window']}/"
            f"W{params['target_weight']:.2f}"
        )
        train_range = f"{row['train_start'].date()}~{row['train_end'].date()}"
        test_range = f"{row['test_start'].date()}~{row['test_end'].date()}"
        print(
            f"{int(row['window_id']):>2} "
            f"{train_range:<23} {test_range:<23} "
            f"{param_text:<42} "
            f"{row['oos_return']*100:>+8.2f}% "
            f"{row['oos_mdd']*100:>+7.2f}% "
            f"{row['oos_sharpe']:>7.2f} "
            f"{row['oos_calmar']:>7.2f}"
        )

    return result


def main():
    setup_logging(log_level='WARNING')

    print("=" * 80)
    print(" Walk-Forward 검증 — 과최적화 방지용 OOS 성과 확인")
    print("=" * 80)

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

    print(f"\n{'='*80}")
    print(" 종합 OOS 비교")
    print(f"{'='*80}\n")
    print(f"{'종목':<12} {'윈도우':>6} {'OOS 수익':>10} {'OOS MDD':>10} {'Sharpe':>8} {'Calmar':>8} {'통과율':>8}")
    print("─" * 80)

    for (market, _), result in zip(targets, results):
        m = result.oos_metrics
        print(
            f"{market:<12} "
            f"{len(result.windows):>6d} "
            f"{m.total_return*100:>+9.2f}% "
            f"{m.max_drawdown*100:>+9.2f}% "
            f"{m.sharpe_ratio:>8.2f} "
            f"{m.calmar_ratio:>8.2f} "
            f"{result.pass_rate*100:>7.2f}%"
        )

    print("\n해석:")
    print("  - 여기서 좋은 결과가 나와야 실전 전략 후보로 볼 수 있습니다.")
    print("  - In-Sample이 좋아도 OOS가 무너지면 과최적화 가능성이 큽니다.")
    print("  - SMA는 여전히 기준선이며, 다음 실전 전략도 이 엔진으로 검증해야 합니다.")


if __name__ == '__main__':
    main()
