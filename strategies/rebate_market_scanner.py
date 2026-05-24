"""Real-time market ranking for the Rebate Micro profile.

The scanner is intentionally separate from the stable trend models. It ranks
KRW markets by tradability first, then short-horizon opportunity:

1. 24h KRW liquidity
2. orderbook spread
3. recent 1m volatility
4. momentum/reversal signal quality
5. estimated edge after fee/rebate assumptions
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import log10
from statistics import pstdev
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RebateMarketCandidate:
    """Ranked Rebate Micro market candidate."""

    market: str
    score: float
    liquidity_krw: Decimal
    spread_pct: Decimal
    volatility_pct: Decimal
    momentum_pct: Decimal
    reversal_pct: Decimal
    expected_net_pct: Decimal
    reason: str


def scan_rebate_micro_candidates(
    client,
    *,
    min_liquidity_krw: Decimal = Decimal("1000000000"),
    max_spread_pct: Decimal = Decimal("0.0015"),
    min_volatility_pct: Decimal = Decimal("0.00025"),
    fee_round_trip_pct: Decimal = Decimal("0.0008"),
    rebate_round_trip_pct: Decimal = Decimal("0.0008"),
    slippage_buffer_pct: Decimal = Decimal("0.0002"),
    prefilter_limit: int = 30,
    result_limit: int = 5,
    candle_count: int = 30,
) -> list[RebateMarketCandidate]:
    """Return ranked live candidates for Rebate Micro trading.

    All percentages are fractions: 0.001 means 0.10%.
    """
    markets = [
        market.market for market in client.get_krw_markets()
        if not getattr(market, "market_warning", None)
    ]
    tickers = []
    for chunk in _chunked(markets, 60):
        tickers.extend(client.get_tickers(chunk))
    liquid = [
        ticker for ticker in tickers
        if ticker.acc_trade_price_24h >= min_liquidity_krw
    ]
    liquid.sort(key=lambda ticker: ticker.acc_trade_price_24h, reverse=True)

    candidates: list[RebateMarketCandidate] = []
    for ticker in liquid[:prefilter_limit]:
        try:
            orderbook = client.get_orderbook(ticker.market)
            if orderbook.spread_pct is None or orderbook.spread_pct > max_spread_pct:
                continue
            candles = sorted(
                client.get_candles(ticker.market, unit="1m", count=candle_count),
                key=_candle_sort_key,
            )
            closes = [float(candle.trade_price) for candle in candles]
            if len(closes) < 8 or min(closes) <= 0:
                continue
            candidate = score_rebate_market_candidate(
                market=ticker.market,
                liquidity_krw=ticker.acc_trade_price_24h,
                spread_pct=orderbook.spread_pct,
                closes=closes,
                min_liquidity_krw=min_liquidity_krw,
                max_spread_pct=max_spread_pct,
                min_volatility_pct=min_volatility_pct,
                fee_round_trip_pct=fee_round_trip_pct,
                rebate_round_trip_pct=rebate_round_trip_pct,
                slippage_buffer_pct=slippage_buffer_pct,
            )
        except Exception:
            continue
        if candidate.volatility_pct >= min_volatility_pct:
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            candidate.expected_net_pct > 0,
            candidate.score,
            candidate.expected_net_pct,
            candidate.liquidity_krw,
        ),
        reverse=True,
    )
    return candidates[:result_limit]


def score_rebate_market_candidate(
    *,
    market: str,
    liquidity_krw: Decimal,
    spread_pct: Decimal,
    closes: Sequence[float],
    min_liquidity_krw: Decimal,
    max_spread_pct: Decimal,
    min_volatility_pct: Decimal,
    fee_round_trip_pct: Decimal,
    rebate_round_trip_pct: Decimal,
    slippage_buffer_pct: Decimal,
) -> RebateMarketCandidate:
    """Score one market using only local ticker/orderbook/candle inputs."""
    returns = _returns(closes)
    vol = Decimal(str(pstdev(returns[-10:]))) if len(returns) >= 2 else Decimal("0")
    mom_3 = Decimal(str(closes[-1] / closes[-4] - 1)) if len(closes) >= 4 else Decimal("0")
    mom_10 = Decimal(str(closes[-1] / closes[-11] - 1)) if len(closes) >= 11 else Decimal("0")
    recent_low = min(closes[-12:])
    reversal = Decimal(str(max(closes[-1] / recent_low - 1, 0.0)))

    liquidity_score = _clamp(
        (log10(float(liquidity_krw)) - log10(float(min_liquidity_krw))) / 2.0
    )
    spread_score = _clamp(1.0 - float(spread_pct / max_spread_pct))
    volatility_score = _clamp(float(vol / (min_volatility_pct * Decimal("4"))))

    momentum_score = _clamp(float((mom_3 * Decimal("0.70") + mom_10 * Decimal("0.30")) / Decimal("0.003")))
    reversal_score = _clamp(float(reversal / Decimal("0.004")))
    signal_score = max(momentum_score, reversal_score)

    gross_edge = (
        max(mom_3, Decimal("0")) * Decimal("0.35") +
        max(mom_10, Decimal("0")) * Decimal("0.15") +
        reversal * Decimal("0.10") +
        vol * Decimal("0.45")
    )
    round_trip_cost = spread_pct + slippage_buffer_pct + fee_round_trip_pct
    expected_net = gross_edge + rebate_round_trip_pct - round_trip_cost

    score = (
        liquidity_score * 0.30 +
        spread_score * 0.25 +
        volatility_score * 0.20 +
        signal_score * 0.15 +
        _clamp(float(expected_net / Decimal("0.001"))) * 0.10
    )
    reason = _build_reason(expected_net, mom_3, mom_10, reversal)

    return RebateMarketCandidate(
        market=market,
        score=score,
        liquidity_krw=liquidity_krw,
        spread_pct=spread_pct,
        volatility_pct=vol,
        momentum_pct=mom_3,
        reversal_pct=reversal,
        expected_net_pct=expected_net,
        reason=reason,
    )


def _returns(values: Sequence[float]) -> list[float]:
    return [
        values[idx] / values[idx - 1] - 1
        for idx in range(1, len(values))
        if values[idx - 1] > 0
    ]


def _chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(values), size):
        yield values[idx:idx + size]


def _candle_sort_key(candle) -> float:
    dt = candle.candle_datetime_kst or candle.candle_datetime_utc
    return dt.timestamp() if dt is not None else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _build_reason(
    expected_net: Decimal,
    mom_3: Decimal,
    mom_10: Decimal,
    reversal: Decimal,
) -> str:
    signal = "positive EV" if expected_net > 0 else "thin edge"
    if mom_3 > 0 and mom_10 > 0:
        setup = "momentum"
    elif reversal > Decimal("0.001"):
        setup = "reversal"
    else:
        setup = "liquidity/spread"
    return f"{signal}, {setup}"


def format_candidate_lines(candidates: Iterable[RebateMarketCandidate]) -> str:
    """Compact text table for Discord embeds."""
    lines = []
    for idx, candidate in enumerate(candidates, 1):
        lines.append(
            f"{idx}. {candidate.market} | score {candidate.score:.2f} | "
            f"spread {candidate.spread_pct * Decimal('100'):.3f}% | "
            f"vol {candidate.volatility_pct * Decimal('100'):.3f}% | "
            f"EV {candidate.expected_net_pct * Decimal('100'):+.3f}%"
        )
    return "\n".join(lines) if lines else "추천 가능한 후보가 없습니다."
