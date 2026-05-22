"""Discord status embed tests."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.models import (
    Account,
    Balances,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    Orderbook,
    OrderbookUnit,
    Ticker,
)
from notifications.discord_status import (
    AccountSnapshot,
    build_snapshot_embeds,
    estimate_total_equity,
    format_order_lines,
)


def make_snapshot() -> AccountSnapshot:
    ticker = Ticker(
        market="KRW-ETH",
        trade_price=Decimal("3000000"),
        opening_price=Decimal("2900000"),
        high_price=Decimal("3100000"),
        low_price=Decimal("2800000"),
        prev_closing_price=Decimal("2900000"),
        change="RISE",
        change_price=Decimal("100000"),
        change_rate=Decimal("0.034"),
        signed_change_price=Decimal("100000"),
        signed_change_rate=Decimal("0.034"),
        acc_trade_volume_24h=Decimal("1000"),
        acc_trade_price_24h=Decimal("3000000000"),
        timestamp=1705305600000,
    )
    orderbook = Orderbook(
        market="KRW-ETH",
        timestamp=1705305600000,
        total_ask_size=Decimal("10"),
        total_bid_size=Decimal("11"),
        units=[
            OrderbookUnit(
                ask_price=Decimal("3001000"),
                bid_price=Decimal("2999000"),
                ask_size=Decimal("1"),
                bid_size=Decimal("1"),
            )
        ],
    )
    balances = Balances(accounts=[
        Account(
            currency="KRW",
            balance=Decimal("100000"),
            locked=Decimal("0"),
            avg_buy_price=Decimal("0"),
            avg_buy_price_modified=False,
            unit_currency="KRW",
        ),
        Account(
            currency="ETH",
            balance=Decimal("0.01"),
            locked=Decimal("0"),
            avg_buy_price=Decimal("2500000"),
            avg_buy_price_modified=False,
            unit_currency="KRW",
        ),
    ])
    order = Order(
        uuid="abc",
        market="KRW-ETH",
        side=OrderSide.BID,
        ord_type=OrderType.MARKET_BUY,
        state=OrderState.DONE,
        price=Decimal("5000"),
        volume=None,
        remaining_volume=Decimal("0"),
        executed_volume=Decimal("0.001"),
        paid_fee=Decimal("2"),
        locked=Decimal("0"),
        trades_count=1,
        created_at=datetime(2026, 5, 22, 9, 0, tzinfo=timezone.utc),
        raw_data={"executed_funds": "5000"},
    )
    return AccountSnapshot(
        market="KRW-ETH",
        ticker=ticker,
        orderbook=orderbook,
        balances=balances,
        open_orders=(),
        filled_orders=(order,),
        canceled_orders=(),
        prices_by_currency={"ETH": Decimal("3000000")},
    )


def test_total_equity_and_embeds():
    print("\n[테스트 1] Discord 상태 embed 생성")
    print("-" * 60)

    snapshot = make_snapshot()
    equity = estimate_total_equity(snapshot.balances, snapshot.prices_by_currency)
    embeds = build_snapshot_embeds(
        snapshot,
        title_prefix="30m LIVE status",
        orders_today=1,
        max_orders_per_day=12,
        signal="HOLD",
        target_weight=0.0,
        risk_summary="daily 0.00%/3.00%",
        kill_switch_active=False,
        timestamp=datetime(2026, 5, 22, 18, 30),
    )

    assert equity == Decimal("130000.00")
    assert len(embeds) == 3
    assert embeds[0].title == "30m LIVE status - KRW-ETH"
    assert "orders=1/12" in embeds[0].description
    assert "Estimated equity" in embeds[1].description
    assert "bid/done" in embeds[2].fields[0].value

    print("  ✓ 가격/자산/주문 embed 3개 생성")
    print("  ✓ 총 자산 추정과 주문 이력 표시 정상")


def test_order_line_uses_executed_funds():
    print("\n[테스트 2] 주문 체결금액 표시")
    print("-" * 60)

    snapshot = make_snapshot()
    line = format_order_lines(snapshot.filled_orders)

    assert "5,000 KRW" in line
    assert "exec 0.001" in line

    print("  ✓ executed_funds 기반 체결금액 표시")


def main():
    print("=" * 60)
    print("Discord 상태 리포트 테스트")
    print("=" * 60)

    tests = [
        test_total_equity_and_embeds,
        test_order_line_uses_executed_funds,
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
