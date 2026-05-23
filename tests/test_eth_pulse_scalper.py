"""ETH pulse scalper strategy tests."""
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import Signal, StrategyContext
from strategies.eth_pulse_scalper import PulseScalperStrategy


def make_context(
    *,
    now: datetime,
    trade_price: float = 3_000_000.0,
    current_position: float = 0.0,
    avg_buy_price: float = 0.0,
    last_pulse_buy_at: datetime | None = None,
) -> StrategyContext:
    """Create a tiny live-like context using ticker price metadata."""
    ohlcv = pd.DataFrame(
        {
            "open": [trade_price],
            "high": [trade_price],
            "low": [trade_price],
            "close": [trade_price],
            "volume": [1.0],
        },
        index=[now],
    )
    return StrategyContext(
        timestamp=now,
        ticker="KRW-XRP",
        ohlcv=ohlcv,
        current_position=current_position,
        avg_buy_price=avg_buy_price,
        cash=200_000.0,
        portfolio_value=200_000.0,
        metadata={
            "now": now.isoformat(timespec="seconds"),
            "trade_price": trade_price,
            "last_pulse_buy_at": (
                last_pulse_buy_at.isoformat(timespec="seconds")
                if last_pulse_buy_at
                else None
            ),
        },
    )


def test_pulse_buys_when_no_prior_buy():
    print("\n[테스트 1] 첫 10분 pulse 매수")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    decision = strategy.generate_signal(make_context(now=datetime(2026, 5, 23, 9, 0, 0)))

    assert decision.signal == Signal.BUY
    assert decision.target_weight == 1.0
    assert "pulse entry" in decision.reason
    print("  ✓ 이전 매수 기록이 없으면 즉시 10만원 pulse 시작")


def test_pulse_waits_until_ten_minutes():
    print("\n[테스트 2] 10분 간격 강제")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 5, 0)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            last_pulse_buy_at=now - timedelta(minutes=5),
        )
    )

    assert decision.signal == Signal.HOLD
    assert "cash wait" in decision.reason
    print("  ✓ 마지막 pulse 이후 10분 전에는 추가 매수 없음")


def test_pulse_adds_every_ten_minutes_even_while_holding():
    print("\n[테스트 3] 보유 중 10분마다 추가 pulse")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 10, 0)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=3_000_500.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(minutes=10),
        )
    )

    assert decision.signal == Signal.BUY
    assert "pulse add" in decision.reason
    print("  ✓ 손익이 TP/SL 사이면 10분마다 10만원 추가 pulse")


def test_pulse_takes_profit_from_live_ticker():
    print("\n[테스트 4] 3초 이후 +0.3% 익절")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 0, 3)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=3_009_100.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=3),
        )
    )

    assert decision.signal == Signal.SELL
    assert "take-profit" in decision.reason
    print("  ✓ 실시간 ticker 기준 +0.3% 도달 시 전량 매도")


def test_pulse_stops_loss_from_live_ticker():
    print("\n[테스트 5] -0.04% 손절")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 0, 3)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=2_998_700.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=3),
        )
    )

    assert decision.signal == Signal.SELL
    assert "stop-loss" in decision.reason
    print("  ✓ 실시간 ticker 기준 -0.04% 도달 시 전량 매도")


def test_pulse_does_not_exit_before_three_seconds():
    print("\n[테스트 6] 첫 3초 보호 구간")
    print("-" * 60)

    strategy = PulseScalperStrategy()
    now = datetime(2026, 5, 23, 9, 0, 2)
    decision = strategy.generate_signal(
        make_context(
            now=now,
            trade_price=3_009_000.0,
            current_position=0.03,
            avg_buy_price=3_000_000.0,
            last_pulse_buy_at=now - timedelta(seconds=2),
        )
    )

    assert decision.signal == Signal.HOLD
    assert "monitor" in decision.reason
    print("  ✓ 매수 후 3초 전에는 TP/SL 청산을 시작하지 않음")


def main():
    print("=" * 60)
    print("Pulse Scalper 테스트")
    print("=" * 60)

    tests = [
        test_pulse_buys_when_no_prior_buy,
        test_pulse_waits_until_ten_minutes,
        test_pulse_adds_every_ten_minutes_even_while_holding,
        test_pulse_takes_profit_from_live_ticker,
        test_pulse_stops_loss_from_live_ticker,
        test_pulse_does_not_exit_before_three_seconds,
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
            failed += 1

    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과 / {failed}개 실패 (총 {len(tests)}개)")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
