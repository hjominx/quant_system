"""
ETH/Cash micro live runner.

This script runs guarded decision cycles for KRW-ETH:
1. Pull recent 4h or daily candles from Bithumb.
2. Evaluate ETHCashRegimeStrategy.
3. Check balances, open orders, spread, account risk, and micro-live caps.
4. Place at most one market order.

Default mode is dry-run. Real orders require all three:
    - .env TRADING_MODE=LIVE
    - --live
    - --yes-i-understand-risk

Example dry-run:
    python scripts/run_eth_cash_micro_live.py --loop --max-order-krw 10000 --max-position-krw 30000

Example real micro-live:
    python scripts/run_eth_cash_micro_live.py --loop --live --yes-i-understand-risk --max-order-krw 10000 --max-position-krw 30000
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Iterable, Optional
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.bithumb_client import BithumbClient
from execution.models import Balances, Candle, Order
from risk.live_guard import (
    RiskLimits,
    assert_risk_allows_action,
    evaluate_risk,
    format_pct,
    load_risk_state,
    save_risk_state,
)
from strategies.base import Signal, StrategyContext, TradeDecision
from strategies.eth_cash_regime import ETHCashRegimeStrategy
from utils.config import Config, get_config
from utils.logger import setup_logging


MARKET = "KRW-ETH"
DEFAULT_TIMEFRAME = "4h"
DEFAULT_PROFILE = "intraday4h"
DEFAULT_CANDLE_COUNT = 200
KRW_QUANT = Decimal("1")
VOLUME_QUANT = Decimal("0.00000001")


@dataclass(frozen=True)
class PilotPlan:
    """Order plan produced after strategy and safety checks."""
    action: str
    reason: str
    krw_amount: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    dry_run: bool = True


@dataclass(frozen=True)
class PilotState:
    """Persistent micro-live state used for cooldown and daily order caps."""
    date: str
    orders_today: int = 0
    last_order_at: Optional[str] = None


def strategy_params_for_profile(profile: str) -> dict:
    """Return ETHCashRegime parameters for a named live-trading profile."""
    if profile == "daily":
        return {"periods_per_year": 365}
    if profile == "intraday4h":
        return {
            "fast_window": 10,
            "slow_window": 60,
            "exit_window": 20,
            "momentum_window": 18,
            "momentum_min": 0.04,
            "rsi_window": 14,
            "rsi_min": 42,
            "rsi_max": 88,
            "vol_window": 18,
            "periods_per_year": 365 * 6,
            "max_entry_volatility": 1.45,
            "exit_volatility": 1.90,
            "target_volatility": 0.34,
            "breakout_window": 36,
            "breakout_tolerance": 0.04,
            "exit_momentum": -0.06,
            "exit_rsi": 36,
            "crash_5d": -0.08,
            "crash_10d": -0.14,
            "trailing_stop": 0.12,
            "max_position": 1.00,
            "min_position": 0.20,
        }
    raise ValueError(f"지원하지 않는 profile: {profile}")


def load_pilot_state(path: Path, now: datetime) -> PilotState:
    """Load persisted pilot state, resetting counters on a new local date."""
    today = now.date().isoformat()
    if not path.exists():
        return PilotState(date=today)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    state = PilotState(
        date=str(raw.get("date", today)),
        orders_today=int(raw.get("orders_today", 0)),
        last_order_at=raw.get("last_order_at"),
    )
    if state.date != today:
        return PilotState(date=today)
    return state


def save_pilot_state(path: Path, state: PilotState) -> None:
    """Persist pilot state for restart-safe cooldown and order limits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "date": state.date,
            "orders_today": state.orders_today,
            "last_order_at": state.last_order_at,
        }, f, ensure_ascii=False, indent=2)


def assert_order_quota(
    *,
    state: PilotState,
    now: datetime,
    max_orders_per_day: int,
    cooldown_minutes: int,
) -> None:
    """Reject a live order when daily count or cooldown would be violated."""
    if state.orders_today >= max_orders_per_day:
        raise RuntimeError(
            f"일일 주문 횟수 한도 도달: {state.orders_today}/{max_orders_per_day}"
        )
    if not state.last_order_at:
        return
    last_order_at = datetime.fromisoformat(state.last_order_at)
    cooldown = timedelta(minutes=cooldown_minutes)
    if now - last_order_at < cooldown:
        remaining = cooldown - (now - last_order_at)
        raise RuntimeError(
            f"쿨다운 중: 다음 주문까지 약 {remaining.total_seconds()/60:.1f}분"
        )


def record_order(state: PilotState, now: datetime) -> PilotState:
    """Return state after one accepted order."""
    today = now.date().isoformat()
    orders_today = state.orders_today if state.date == today else 0
    return PilotState(
        date=today,
        orders_today=orders_today + 1,
        last_order_at=now.isoformat(timespec="seconds"),
    )


def parse_decimal(value: str) -> Decimal:
    """Parse a positive Decimal from CLI text."""
    parsed = Decimal(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 값이어야 합니다.")
    return parsed


def market_base_currency(market: str) -> str:
    """Return base currency from a KRW market string such as KRW-ETH."""
    parts = market.split("-")
    if len(parts) != 2 or parts[0] != "KRW":
        raise ValueError(f"KRW 마켓만 지원합니다: {market}")
    return parts[1]


def candles_to_ohlcv(candles: Iterable[Candle]) -> pd.DataFrame:
    """Convert Bithumb candles to the strategy OHLCV DataFrame shape."""
    rows = []
    for candle in candles:
        timestamp = candle.candle_datetime_kst or candle.candle_datetime_utc
        if timestamp is None:
            continue
        rows.append({
            "timestamp": pd.Timestamp(timestamp).tz_localize(None),
            "open": float(candle.opening_price),
            "high": float(candle.high_price),
            "low": float(candle.low_price),
            "close": float(candle.trade_price),
            "volume": float(candle.candle_acc_trade_volume),
        })
    if not rows:
        raise ValueError("캔들 데이터가 비어 있습니다.")

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    required = ["open", "high", "low", "close", "volume"]
    if df[required].isnull().any().any():
        raise ValueError("캔들 데이터에 NaN이 포함되어 있습니다.")
    return df[required]


def validate_live_permission(config: Config, live: bool, confirmed: bool) -> None:
    """Reject real order mode unless every live-trading gate is explicit."""
    if not live:
        return
    if not confirmed:
        raise ValueError("--live 사용 시 --yes-i-understand-risk 확인 플래그가 필요합니다.")
    if not config.is_live_trading:
        raise ValueError(".env TRADING_MODE=LIVE 상태에서만 --live 주문을 허용합니다.")


def quantize_krw(amount: Decimal) -> Decimal:
    """Round down KRW amounts to whole won."""
    return amount.quantize(KRW_QUANT, rounding=ROUND_DOWN)


def quantize_volume(volume: Decimal) -> Decimal:
    """Round down coin volume to a conservative precision."""
    return volume.quantize(VOLUME_QUANT, rounding=ROUND_DOWN)


def calculate_buy_amount(
    *,
    target_weight: float,
    portfolio_value_krw: Decimal,
    current_position_krw: Decimal,
    krw_balance: Decimal,
    max_order_krw: Decimal,
    max_position_krw: Decimal,
    min_order_krw: Decimal,
) -> Decimal:
    """Calculate the guarded KRW market-buy amount."""
    target_position_krw = quantize_krw(
        portfolio_value_krw * Decimal(str(target_weight))
    )
    desired_add = target_position_krw - current_position_krw
    remaining_position_cap = max_position_krw - current_position_krw
    amount = min(desired_add, remaining_position_cap, max_order_krw, krw_balance)
    amount = quantize_krw(max(amount, Decimal("0")))
    if amount < min_order_krw:
        return Decimal("0")
    return amount


def calculate_sell_volume(
    *,
    coin_balance: Decimal,
    trade_price: Decimal,
    min_order_krw: Decimal,
) -> Decimal:
    """Calculate the guarded market-sell volume."""
    estimated_value = coin_balance * trade_price
    if estimated_value < min_order_krw:
        return Decimal("0")
    return quantize_volume(coin_balance)


def build_strategy_context(
    *,
    market: str,
    ohlcv: pd.DataFrame,
    balances: Balances,
    trade_price: Decimal,
) -> StrategyContext:
    """Build a StrategyContext from live balances and recent candles."""
    base = market_base_currency(market)
    coin_balance = balances.get_total(base)
    krw_balance = balances.krw_balance
    position_value = coin_balance * trade_price
    portfolio_value = krw_balance + position_value
    account = balances.get(base)
    avg_buy_price = float(account.avg_buy_price) if account else 0.0

    return StrategyContext(
        timestamp=ohlcv.index[-1],
        ticker=market,
        ohlcv=ohlcv,
        current_position=float(coin_balance),
        avg_buy_price=avg_buy_price,
        cash=float(krw_balance),
        portfolio_value=float(portfolio_value),
    )


def calculate_portfolio_equity(
    *,
    market: str,
    balances: Balances,
    trade_price: Decimal,
) -> Decimal:
    """Return KRW cash plus live mark-to-market value for the traded coin."""
    base = market_base_currency(market)
    coin_balance = balances.get_total(base)
    return balances.krw_balance + (coin_balance * trade_price)


def build_pilot_plan(
    *,
    decision: TradeDecision,
    balances: Balances,
    trade_price: Decimal,
    market: str,
    max_order_krw: Decimal,
    max_position_krw: Decimal,
    min_order_krw: Decimal,
    dry_run: bool,
) -> PilotPlan:
    """Turn a strategy decision into one guarded pilot order plan."""
    base = market_base_currency(market)
    coin_balance = balances.get_total(base)
    krw_balance = balances.krw_balance
    current_position_krw = coin_balance * trade_price
    portfolio_value_krw = krw_balance + current_position_krw

    if decision.signal == Signal.BUY:
        krw_amount = calculate_buy_amount(
            target_weight=decision.target_weight,
            portfolio_value_krw=portfolio_value_krw,
            current_position_krw=current_position_krw,
            krw_balance=krw_balance,
            max_order_krw=max_order_krw,
            max_position_krw=max_position_krw,
            min_order_krw=min_order_krw,
        )
        if krw_amount <= 0:
            return PilotPlan(
                action="SKIP",
                reason=(
                    "매수 신호는 있지만 최소 주문금액/현금/포지션 한도 때문에 주문하지 않음 | "
                    f"{decision.reason}"
                ),
                dry_run=dry_run,
            )
        return PilotPlan(
            action="BUY",
            reason=decision.reason,
            krw_amount=krw_amount,
            dry_run=dry_run,
        )

    if decision.signal == Signal.SELL:
        volume = calculate_sell_volume(
            coin_balance=coin_balance,
            trade_price=trade_price,
            min_order_krw=min_order_krw,
        )
        if volume <= 0:
            return PilotPlan(
                action="SKIP",
                reason=f"매도 신호는 있지만 보유 ETH 평가액이 최소 주문금액 미만 | {decision.reason}",
                dry_run=dry_run,
            )
        return PilotPlan(
            action="SELL",
            reason=decision.reason,
            volume=volume,
            dry_run=dry_run,
        )

    return PilotPlan(action="HOLD", reason=decision.reason, dry_run=dry_run)


def assert_spread_ok(spread_pct: Optional[Decimal], max_spread_pct: Decimal) -> None:
    """Reject trading when the orderbook spread is too wide."""
    if spread_pct is None:
        raise ValueError("호가 스프레드를 계산할 수 없습니다.")
    if spread_pct > max_spread_pct:
        raise ValueError(
            f"호가 스프레드 과다: {spread_pct * Decimal('100'):.3f}% "
            f"> 허용 {max_spread_pct * Decimal('100'):.3f}%"
        )


def execute_plan(client: BithumbClient, market: str, plan: PilotPlan) -> Optional[Order]:
    """Execute one market order for the already-validated pilot plan."""
    if plan.action == "BUY":
        return client.market_buy(market, plan.krw_amount)
    if plan.action == "SELL":
        return client.market_sell(market, plan.volume)
    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run one guarded ETH/Cash micro-live cycle.")
    parser.add_argument("--market", default=MARKET, help="KRW market. Default: KRW-ETH")
    parser.add_argument("--timeframe", choices=["4h", "1d"], default=DEFAULT_TIMEFRAME)
    parser.add_argument("--profile", choices=["intraday4h", "daily"], default=DEFAULT_PROFILE)
    parser.add_argument("--candle-count", type=int, default=DEFAULT_CANDLE_COUNT)
    parser.add_argument("--max-order-krw", type=parse_decimal, default=Decimal("10000"))
    parser.add_argument("--max-position-krw", type=parse_decimal, default=Decimal("30000"))
    parser.add_argument("--min-order-krw", type=parse_decimal, default=Decimal("5000"))
    parser.add_argument("--max-spread-pct", type=parse_decimal, default=Decimal("0.005"))
    parser.add_argument("--loop", action="store_true", help="Keep running guarded decision cycles.")
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--cooldown-minutes", type=int, default=240)
    parser.add_argument("--max-orders-per-day", type=int, default=4)
    parser.add_argument(
        "--state-file",
        default="logs/eth_cash_micro_live_state.json",
        help="Persistent order counter/cooldown file.",
    )
    parser.add_argument(
        "--risk-state-file",
        default="logs/eth_cash_micro_live_risk_state.json",
        help="Persistent account risk baseline file.",
    )
    parser.add_argument(
        "--kill-switch-file",
        default="logs/KILL_SWITCH_ACTIVE",
        help="If this file exists, all BUY/SELL orders are blocked.",
    )
    parser.add_argument("--live", action="store_true", help="Send real orders. Requires TRADING_MODE=LIVE.")
    parser.add_argument("--yes-i-understand-risk", action="store_true")
    parser.add_argument("--no-file-log", action="store_true", help="Disable file logging for local permission-limited runs.")
    return parser.parse_args()


def validate_runner_args(args: argparse.Namespace) -> None:
    """Validate runtime argument combinations."""
    if args.market != MARKET:
        raise ValueError("현재 파일럿은 ETH 전용입니다. --market KRW-ETH만 허용합니다.")
    if args.profile == "intraday4h" and args.timeframe != "4h":
        raise ValueError("intraday4h profile은 --timeframe 4h와 함께 사용해야 합니다.")
    if args.profile == "daily" and args.timeframe != "1d":
        raise ValueError("daily profile은 --timeframe 1d와 함께 사용해야 합니다.")
    if args.interval_minutes <= 0:
        raise ValueError("--interval-minutes는 1 이상이어야 합니다.")
    if args.cooldown_minutes < 0:
        raise ValueError("--cooldown-minutes는 0 이상이어야 합니다.")
    if args.max_orders_per_day <= 0:
        raise ValueError("--max-orders-per-day는 1 이상이어야 합니다.")


def print_banner(args: argparse.Namespace, dry_run: bool) -> None:
    """Print runner settings at startup."""
    print("=" * 72)
    print("ETH/Cash Micro Live Runner")
    print("=" * 72)
    print(
        f"mode={'DRY-RUN' if dry_run else 'LIVE'} market={args.market} "
        f"timeframe={args.timeframe} profile={args.profile}"
    )
    print(
        f"max_order={args.max_order_krw:,} KRW "
        f"max_position={args.max_position_krw:,} KRW "
        f"min_order={args.min_order_krw:,} KRW"
    )
    print(
        f"loop={args.loop} interval={args.interval_minutes}m "
        f"cooldown={args.cooldown_minutes}m max_orders/day={args.max_orders_per_day}"
    )
    print(f"risk_state={args.risk_state_file} kill_switch={args.kill_switch_file}")


def run_cycle(
    *,
    client: BithumbClient,
    args: argparse.Namespace,
    dry_run: bool,
    state_path: Path,
    risk_state_path: Path,
    kill_switch_path: Path,
    risk_limits: RiskLimits,
) -> Optional[Order]:
    """Run one guarded decision cycle and optionally place one order."""
    now = datetime.now()
    state = load_pilot_state(state_path, now)

    strategy = ETHCashRegimeStrategy(params=strategy_params_for_profile(args.profile))
    if args.candle_count < strategy.warmup_period() + 5:
        raise ValueError("candle-count가 전략 warmup보다 작습니다.")

    if not client.health_check():
        raise RuntimeError("빗썸 public API 헬스체크 실패")

    open_orders = client.get_open_orders(args.market)
    if open_orders:
        raise RuntimeError(f"미체결 주문 {len(open_orders)}개 존재. 새 주문을 차단합니다.")

    candles = client.get_candles(args.market, unit=args.timeframe, count=args.candle_count)
    ohlcv = candles_to_ohlcv(candles)
    ticker = client.get_ticker(args.market)
    balances = client.get_balances()
    orderbook = client.get_orderbook(args.market)
    assert_spread_ok(orderbook.spread_pct, args.max_spread_pct)

    portfolio_equity = calculate_portfolio_equity(
        market=args.market,
        balances=balances,
        trade_price=ticker.trade_price,
    )
    risk_state = load_risk_state(risk_state_path, now, portfolio_equity)
    risk_report = evaluate_risk(
        state=risk_state,
        equity=portfolio_equity,
        limits=risk_limits,
        kill_switch_file_active=kill_switch_path.exists(),
    )

    ctx = build_strategy_context(
        market=args.market,
        ohlcv=ohlcv,
        balances=balances,
        trade_price=ticker.trade_price,
    )
    decision = strategy.generate_signal(ctx)
    plan = build_pilot_plan(
        decision=decision,
        balances=balances,
        trade_price=ticker.trade_price,
        market=args.market,
        max_order_krw=args.max_order_krw,
        max_position_krw=args.max_position_krw,
        min_order_krw=args.min_order_krw,
        dry_run=dry_run,
    )

    base = market_base_currency(args.market)
    coin_total = balances.get_total(base)
    coin_value = coin_total * ticker.trade_price
    print(f"\n[{now.isoformat(timespec='seconds')}]")
    print("[상태]")
    print(f"KRW 가용: {balances.krw_balance:,}")
    print(f"{base} 총량: {coin_total}")
    print(f"{base} 평가액: {quantize_krw(coin_value):,} KRW")
    print(f"현재가: {ticker.trade_price:,} KRW")
    print(f"스프레드: {orderbook.spread_pct * Decimal('100'):.4f}%")
    print(f"오늘 주문수: {state.orders_today}/{args.max_orders_per_day}")

    print("\n[위험 게이트]")
    print(f"계좌 평가액: {quantize_krw(risk_report.equity):,} KRW")
    print(
        f"일일 손실: {format_pct(risk_report.daily_loss)} "
        f"/ 한도 {format_pct(risk_limits.daily_loss_limit)}"
    )
    print(
        f"주간 손실: {format_pct(risk_report.weekly_loss)} "
        f"/ 한도 {format_pct(risk_limits.weekly_loss_limit)}"
    )
    print(
        f"MDD: {format_pct(risk_report.drawdown)} "
        f"/ 한도 {format_pct(risk_limits.max_drawdown_limit)}"
    )
    if risk_report.kill_switch_active:
        print("킬스위치: ACTIVE")
    else:
        print("킬스위치: inactive")

    print("\n[전략 판단]")
    print(f"signal={decision.signal.name} target_weight={decision.target_weight:.3f}")
    print(decision.reason)

    print("\n[파일럿 주문 계획]")
    print(plan)

    if not dry_run:
        save_risk_state(risk_state_path, risk_report.state)

    if plan.action in ("BUY", "SELL"):
        assert_risk_allows_action(plan.action, risk_report)
        if not dry_run:
            assert_order_quota(
                state=state,
                now=now,
                max_orders_per_day=args.max_orders_per_day,
                cooldown_minutes=args.cooldown_minutes,
            )

    order = execute_plan(client, args.market, plan)
    if order is None:
        print("\n주문 없음.")
        return None

    if not dry_run:
        save_pilot_state(state_path, record_order(state, now))
    print("\n주문 요청 완료")
    print(f"uuid={order.uuid} state={order.state.value} dry_run={dry_run}")
    return order


def main() -> None:
    """Run one or repeated guarded ETH/Cash pilot cycles."""
    args = parse_args()
    validate_runner_args(args)
    cfg = get_config()
    setup_logging(log_level=cfg.log_level, enable_file_log=not args.no_file_log)
    validate_live_permission(cfg, live=args.live, confirmed=args.yes_i_understand_risk)

    dry_run = not args.live
    state_path = Path(args.state_file)
    risk_state_path = Path(args.risk_state_file)
    kill_switch_path = Path(args.kill_switch_file)
    risk_limits = RiskLimits.from_config(cfg)
    client = BithumbClient.from_config(
        cfg,
        dry_run=dry_run,
        allow_live_orders=args.live and cfg.is_live_trading,
    )

    try:
        print_banner(args, dry_run)
        while True:
            try:
                run_cycle(
                    client=client,
                    args=args,
                    dry_run=dry_run,
                    state_path=state_path,
                    risk_state_path=risk_state_path,
                    kill_switch_path=kill_switch_path,
                    risk_limits=risk_limits,
                )
            except Exception as exc:
                print(f"\n[차단/오류] {type(exc).__name__}: {exc}")
                if not args.loop:
                    raise

            if not args.loop:
                return

            sleep_seconds = args.interval_minutes * 60
            print(f"\n다음 판단까지 {args.interval_minutes}분 대기...")
            time.sleep(sleep_seconds)
    finally:
        client.close()


if __name__ == "__main__":
    main()
