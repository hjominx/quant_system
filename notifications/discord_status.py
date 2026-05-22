"""Discord status embeds for live trading monitoring."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Optional

from execution.bithumb_client import BithumbClient
from execution.models import Balances, Order, Orderbook, Ticker
from notifications.discord import (
    COLOR_INFO,
    COLOR_SUCCESS,
    COLOR_WARNING,
    DiscordEmbed,
    DiscordField,
)


KRW_QUANT = Decimal("1")
PCT_100 = Decimal("100")


@dataclass(frozen=True)
class AccountSnapshot:
    """All account data needed to build Discord status messages."""

    market: str
    ticker: Ticker
    orderbook: Orderbook
    balances: Balances
    open_orders: tuple[Order, ...]
    filled_orders: tuple[Order, ...]
    canceled_orders: tuple[Order, ...]
    prices_by_currency: dict[str, Decimal]


def collect_account_snapshot(
    client: BithumbClient,
    *,
    market: str = "KRW-ETH",
    order_limit: int = 5,
) -> AccountSnapshot:
    """Fetch price, balances, and recent order history for Discord reporting."""
    ticker = client.get_ticker(market)
    orderbook = client.get_orderbook(market)
    balances = client.get_balances()
    prices_by_currency = fetch_krw_prices_for_balances(client, balances)
    open_orders = tuple(client.get_open_orders(market))
    filled_orders = tuple(client.get_orders(market=market, state="done", limit=order_limit))
    canceled_orders = tuple(client.get_orders(market=market, state="cancel", limit=order_limit))
    return AccountSnapshot(
        market=market,
        ticker=ticker,
        orderbook=orderbook,
        balances=balances,
        open_orders=open_orders,
        filled_orders=filled_orders,
        canceled_orders=canceled_orders,
        prices_by_currency=prices_by_currency,
    )


def fetch_krw_prices_for_balances(
    client: BithumbClient,
    balances: Balances,
) -> dict[str, Decimal]:
    """Return KRW prices for held coins that have a KRW market."""
    currencies = balances.coin_currencies
    if not currencies:
        return {}

    krw_markets = {market.market for market in client.get_krw_markets()}
    target_markets = [
        f"KRW-{currency}"
        for currency in currencies
        if f"KRW-{currency}" in krw_markets
    ]
    if not target_markets:
        return {}

    tickers = client.get_tickers(target_markets)
    return {
        ticker.market.split("-")[1]: ticker.trade_price
        for ticker in tickers
    }


def build_snapshot_embeds(
    snapshot: AccountSnapshot,
    *,
    title_prefix: str = "LIVE status",
    orders_today: Optional[int] = None,
    max_orders_per_day: Optional[int] = None,
    signal: Optional[str] = None,
    target_weight: Optional[float] = None,
    risk_summary: Optional[str] = None,
    kill_switch_active: Optional[bool] = None,
    timestamp: Optional[datetime] = None,
) -> tuple[DiscordEmbed, ...]:
    """Build compact Discord embeds for price, assets, and recent orders."""
    timestamp = timestamp or datetime.now()
    return (
        build_price_embed(
            snapshot=snapshot,
            title_prefix=title_prefix,
            signal=signal,
            target_weight=target_weight,
            orders_today=orders_today,
            max_orders_per_day=max_orders_per_day,
            risk_summary=risk_summary,
            kill_switch_active=kill_switch_active,
            timestamp=timestamp,
        ),
        build_assets_embed(snapshot),
        build_orders_embed(snapshot),
    )


def select_snapshot_embeds(
    embeds: tuple[DiscordEmbed, ...],
    report: str,
) -> tuple[DiscordEmbed, ...]:
    """Select one report section from build_snapshot_embeds output."""
    if report == "all":
        return embeds
    index = {"price": 0, "assets": 1, "orders": 2}[report]
    return (embeds[index],)


def build_price_embed(
    *,
    snapshot: AccountSnapshot,
    title_prefix: str,
    signal: Optional[str],
    target_weight: Optional[float],
    orders_today: Optional[int],
    max_orders_per_day: Optional[int],
    risk_summary: Optional[str],
    kill_switch_active: Optional[bool],
    timestamp: datetime,
) -> DiscordEmbed:
    """Build price and bot-state summary embed."""
    ticker = snapshot.ticker
    orderbook = snapshot.orderbook
    color = COLOR_WARNING if kill_switch_active else COLOR_INFO

    status_bits = []
    if signal:
        status_bits.append(f"signal={signal}")
    if target_weight is not None:
        status_bits.append(f"target={target_weight:.2f}")
    if orders_today is not None and max_orders_per_day is not None:
        status_bits.append(f"orders={orders_today}/{max_orders_per_day}")
    if kill_switch_active is not None:
        status_bits.append("kill=ON" if kill_switch_active else "kill=off")
    description = " | ".join(status_bits) if status_bits else "price/account heartbeat"

    fields = [
        DiscordField("Price", f"{format_krw(ticker.trade_price)} KRW"),
        DiscordField("24h", format_signed_pct(ticker.signed_change_rate)),
        DiscordField(
            "Range",
            f"H {format_krw(ticker.high_price)} / L {format_krw(ticker.low_price)}",
            inline=False,
        ),
        DiscordField("Best bid", f"{format_optional_krw(orderbook.best_bid)} KRW"),
        DiscordField("Best ask", f"{format_optional_krw(orderbook.best_ask)} KRW"),
        DiscordField("Spread", format_optional_pct(orderbook.spread_pct)),
        DiscordField("Open orders", str(len(snapshot.open_orders))),
        DiscordField("24h value", f"{format_krw(ticker.acc_trade_price_24h)} KRW"),
        DiscordField("Time", timestamp.isoformat(timespec="minutes")),
    ]
    if risk_summary:
        fields.append(DiscordField("Risk", risk_summary, inline=False))

    return DiscordEmbed(
        title=f"{title_prefix} - {snapshot.market}",
        description=description,
        color=color,
        fields=tuple(fields),
    )


def build_assets_embed(snapshot: AccountSnapshot) -> DiscordEmbed:
    """Build account asset summary embed."""
    total_equity = estimate_total_equity(
        snapshot.balances,
        snapshot.prices_by_currency,
    )
    asset_lines = format_asset_lines(
        snapshot.balances,
        snapshot.prices_by_currency,
    )
    return DiscordEmbed(
        title="Asset summary",
        description=f"Estimated equity: {format_krw(total_equity)} KRW",
        color=COLOR_SUCCESS,
        fields=(
            DiscordField("KRW available", f"{format_krw(snapshot.balances.krw_balance)} KRW"),
            DiscordField("Assets", asset_lines or "No coin balance", inline=False),
        ),
    )


def build_orders_embed(snapshot: AccountSnapshot) -> DiscordEmbed:
    """Build recent order summary embed."""
    filled_lines = format_order_lines(snapshot.filled_orders)
    canceled_lines = format_order_lines(snapshot.canceled_orders)
    open_lines = format_order_lines(snapshot.open_orders)
    return DiscordEmbed(
        title="Order history",
        description="Recent Bithumb order states",
        color=COLOR_INFO,
        fields=(
            DiscordField("Filled", filled_lines or "None", inline=False),
            DiscordField("Canceled", canceled_lines or "None", inline=False),
            DiscordField("Open", open_lines or "None", inline=False),
        ),
    )


def estimate_total_equity(
    balances: Balances,
    prices_by_currency: dict[str, Decimal],
) -> Decimal:
    """Estimate KRW equity for KRW plus held coins with known KRW prices."""
    total = balances.get_total("KRW")
    for account in balances.accounts:
        if account.currency == "KRW" or account.total <= 0:
            continue
        price = prices_by_currency.get(account.currency)
        if price is not None:
            total += account.total * price
    return total


def format_asset_lines(
    balances: Balances,
    prices_by_currency: dict[str, Decimal],
    *,
    limit: int = 8,
) -> str:
    """Format non-zero asset balances into short Discord-safe lines."""
    lines = []
    for account in balances.accounts:
        if account.total <= 0:
            continue
        if account.currency == "KRW":
            lines.append(
                f"KRW: available {format_krw(account.balance)}, locked {format_krw(account.locked)}"
            )
            continue
        price = prices_by_currency.get(account.currency)
        value = account.total * price if price is not None else None
        value_text = f" ~= {format_krw(value)} KRW" if value is not None else ""
        lines.append(
            f"{account.currency}: total {format_decimal(account.total)}"
            f", locked {format_decimal(account.locked)}{value_text}"
        )
    return "\n".join(lines[:limit])


def format_order_lines(orders: tuple[Order, ...], *, limit: int = 5) -> str:
    """Format order lines newest-first."""
    lines = []
    for order in orders[:limit]:
        created = order.created_at.isoformat(timespec="minutes") if order.created_at else "time?"
        value = infer_order_value(order)
        value_text = f" | {format_krw(value)} KRW" if value is not None else ""
        lines.append(
            f"{created} | {order.side.value}/{order.state.value}"
            f" | exec {format_decimal(order.executed_volume)}{value_text}"
        )
    return "\n".join(lines)


def infer_order_value(order: Order) -> Optional[Decimal]:
    """Best-effort KRW value from a Bithumb order payload."""
    raw_value = order.raw_data.get("executed_funds")
    if raw_value not in (None, ""):
        try:
            return Decimal(str(raw_value))
        except (InvalidOperation, ValueError):
            return None
    if order.price is not None and order.executed_volume > 0:
        return order.price * order.executed_volume
    if order.price is not None and order.volume is None:
        return order.price
    return None


def format_krw(value: Decimal) -> str:
    """Format KRW-like Decimal values without decimals."""
    return f"{value.quantize(KRW_QUANT, rounding=ROUND_DOWN):,}"


def format_optional_krw(value: Optional[Decimal]) -> str:
    if value is None:
        return "N/A"
    return format_krw(value)


def format_signed_pct(value: Decimal) -> str:
    pct = value * PCT_100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def format_optional_pct(value: Optional[Decimal]) -> str:
    if value is None:
        return "N/A"
    pct = value * PCT_100
    return f"{pct:.2f}%"


def format_decimal(value: Decimal) -> str:
    """Compact Decimal display without scientific notation."""
    normalized = value.normalize()
    text = format(normalized, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text
