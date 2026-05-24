"""
Rebate Micro market scanner tests.

These tests use pure scoring inputs only and do not call Bithumb.
"""
from decimal import Decimal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.rebate_market_scanner import score_rebate_market_candidate


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


def main():
    print("=" * 60)
    print("Rebate Micro Scanner 테스트")
    print("=" * 60)

    tests = [
        test_candidate_prefers_liquid_tight_positive_signal,
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
