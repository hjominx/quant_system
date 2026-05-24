"""
Rebate Micro market scanner tests.

These tests use pure scoring inputs only and do not call Bithumb.
"""
from decimal import Decimal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.rebate_market_scanner import (
    scan_rebate_micro_candidates,
    score_rebate_market_candidate,
)


class FakeMarket:
    def __init__(self, market: str):
        self.market = market
        self.market_warning = None


class FakeTicker:
    def __init__(self, market: str, trade_price_24h: str):
        self.market = market
        self.acc_trade_price_24h = Decimal(trade_price_24h)


class FakeOrderbook:
    def __init__(self, spread_pct: str):
        self.spread_pct = Decimal(spread_pct)


class FakeCandle:
    def __init__(self, price: float):
        self.trade_price = Decimal(str(price))
        self.candle_datetime_kst = None
        self.candle_datetime_utc = None


class FakeClient:
    def __init__(self):
        self.series = {
            "KRW-LOW": [100, 100.01, 100.00, 100.01, 100.00, 100.01, 100.00, 100.01, 100.00, 100.01],
            "KRW-HIGH": [100, 100.8, 99.8, 101.0, 99.7, 101.2, 99.9, 101.4, 100.2, 101.6],
            "KRW-MID": [100, 100.2, 100.0, 100.3, 100.1, 100.4, 100.2, 100.5, 100.3, 100.6],
        }

    def get_krw_markets(self):
        return [FakeMarket(market) for market in self.series]

    def get_tickers(self, markets):
        return [FakeTicker(market, "50000000000") for market in markets]

    def get_orderbook(self, market):
        return FakeOrderbook("0.0002")

    def get_candles(self, market, unit="1m", count=30):
        return [FakeCandle(price) for price in self.series[market]]


def test_candidate_prefers_liquid_tight_positive_signal():
    print("\n[테스트 1] Rebate Micro 후보 점수")
    print("-" * 60)

    candidate = score_rebate_market_candidate(
        market="KRW-SOL",
        liquidity_krw=Decimal("50000000000"),
        spread_pct=Decimal("0.0002"),
        closes=[
            100.0, 100.1, 100.0, 100.2, 100.4,
            100.3, 100.5, 100.8, 101.0, 101.2,
            101.1, 101.4,
        ],
        min_liquidity_krw=Decimal("1000000000"),
        max_spread_pct=Decimal("0.0015"),
        min_volatility_pct=Decimal("0.00025"),
        fee_round_trip_pct=Decimal("0.0008"),
        rebate_round_trip_pct=Decimal("0.0008"),
        slippage_buffer_pct=Decimal("0.0002"),
    )

    assert candidate.score > 0.5
    assert candidate.expected_net_pct > 0
    assert "positive EV" in candidate.reason

    print("  ✓ 유동성/스프레드/변동성/신호/기대값을 종합해 양수 후보로 평가")


def test_scanner_can_sort_by_live_volatility():
    print("\n[테스트 2] 변동성 순 마켓 선택")
    print("-" * 60)

    candidates = scan_rebate_micro_candidates(
        FakeClient(),
        min_liquidity_krw=Decimal("1000000000"),
        max_spread_pct=Decimal("0.0015"),
        min_volatility_pct=Decimal("0"),
        result_limit=3,
        sort_by="volatility",
    )

    assert [candidate.market for candidate in candidates] == ["KRW-HIGH", "KRW-MID", "KRW-LOW"]
    assert candidates[0].volatility_pct > candidates[1].volatility_pct > candidates[2].volatility_pct

    print("  ✓ Set Market 후보를 1분봉 변동성 높은 순서로 정렬")


def main():
    print("=" * 60)
    print("Rebate Micro Scanner 테스트")
    print("=" * 60)

    tests = [
        test_candidate_prefers_liquid_tight_positive_signal,
        test_scanner_can_sort_by_live_volatility,
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


if __name__ == "__main__":
    main()
