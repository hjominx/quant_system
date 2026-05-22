"""
빗썸 API 응답 데이터 모델

설계 원칙:
1. 모든 가격·수량은 Decimal로 처리 (float 절대 금지)
2. 응답 dict를 dataclass로 변환 (오타·키 누락 자동 검출)
3. 시간은 모두 timezone-aware datetime
4. 빗썸 응답의 모든 필드를 보존 (raw_data로 접근 가능)

빗썸 API 2.0 응답 형식을 모델링.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


def _to_decimal(value: Any) -> Decimal:
    """안전한 Decimal 변환 (None, 빈 문자열 처리)"""
    if value is None or value == '':
        return Decimal('0')
    return Decimal(str(value))


def _parse_datetime(value: Any) -> Optional[datetime]:
    """ISO 8601 문자열 → timezone-aware datetime"""
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        # 빗썸 형식: '2024-01-15T09:00:00+09:00' 또는 '2024-01-15T09:00:00'
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


# ============================================================
# Enum 정의
# ============================================================

class OrderSide(str, Enum):
    """주문 방향"""
    BID = 'bid'  # 매수
    ASK = 'ask'  # 매도


class OrderType(str, Enum):
    """주문 타입 (빗썸 API 2.0 ord_type)
    
    중요: 빗썸 V2의 비대칭 설계
    - 시장가 매수는 'price' (금액 지정)
    - 시장가 매도는 'market' (수량 지정)
    - 지정가는 양쪽 다 'limit'
    """
    LIMIT = 'limit'         # 지정가 (가격 + 수량)
    MARKET_BUY = 'price'    # 시장가 매수 (금액만)
    MARKET_SELL = 'market'  # 시장가 매도 (수량만)


class OrderState(str, Enum):
    """주문 상태"""
    WAIT = 'wait'        # 체결 대기
    WATCH = 'watch'      # 예약주문 대기
    DONE = 'done'        # 전체 체결
    CANCEL = 'cancel'    # 주문 취소
    
    @classmethod
    def from_string(cls, value: str) -> 'OrderState':
        """알 수 없는 상태는 그대로 반환 (빗썸이 새 상태 추가할 가능성 대비)"""
        try:
            return cls(value)
        except ValueError:
            # 알 수 없는 상태는 일단 WAIT로 처리하되 로그 남김
            import logging
            logging.warning(f"알 수 없는 주문 상태: {value}")
            return cls.WAIT


# ============================================================
# 시장 정보 (Market)
# ============================================================

@dataclass(frozen=True)
class Market:
    """거래 가능 종목 정보"""
    market: str              # 'KRW-BTC' 같은 마켓 코드
    korean_name: str         # '비트코인'
    english_name: str        # 'Bitcoin'
    market_warning: Optional[str] = None  # 유의 종목 경고
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Market':
        return cls(
            market=data['market'],
            korean_name=data.get('korean_name', ''),
            english_name=data.get('english_name', ''),
            market_warning=data.get('market_warning'),
        )
    
    @property
    def base_currency(self) -> str:
        """기준 화폐 (예: 'BTC')"""
        return self.market.split('-')[1]
    
    @property
    def quote_currency(self) -> str:
        """결제 화폐 (예: 'KRW')"""
        return self.market.split('-')[0]


# ============================================================
# 현재가 (Ticker)
# ============================================================

@dataclass(frozen=True)
class Ticker:
    """종목 현재가 정보"""
    market: str
    trade_price: Decimal           # 현재가
    opening_price: Decimal         # 시가
    high_price: Decimal            # 고가
    low_price: Decimal             # 저가
    prev_closing_price: Decimal    # 전일 종가
    
    change: str                    # 'EVEN', 'RISE', 'FALL'
    change_price: Decimal          # 변동 금액 (절대값)
    change_rate: Decimal           # 변동률 (절대값, 0~1)
    signed_change_price: Decimal   # 부호 있는 변동 금액
    signed_change_rate: Decimal    # 부호 있는 변동률
    
    acc_trade_volume_24h: Decimal  # 24시간 거래량
    acc_trade_price_24h: Decimal   # 24시간 거래대금
    
    timestamp: int                 # 빗썸 응답 타임스탬프 (ms)
    
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ticker':
        return cls(
            market=data['market'],
            trade_price=_to_decimal(data.get('trade_price')),
            opening_price=_to_decimal(data.get('opening_price')),
            high_price=_to_decimal(data.get('high_price')),
            low_price=_to_decimal(data.get('low_price')),
            prev_closing_price=_to_decimal(data.get('prev_closing_price')),
            change=data.get('change', 'EVEN'),
            change_price=_to_decimal(data.get('change_price')),
            change_rate=_to_decimal(data.get('change_rate')),
            signed_change_price=_to_decimal(data.get('signed_change_price')),
            signed_change_rate=_to_decimal(data.get('signed_change_rate')),
            acc_trade_volume_24h=_to_decimal(data.get('acc_trade_volume_24h')),
            acc_trade_price_24h=_to_decimal(data.get('acc_trade_price_24h')),
            timestamp=int(data.get('timestamp', 0)),
            raw_data=data,
        )


# ============================================================
# 캔들 (OHLCV)
# ============================================================

@dataclass(frozen=True)
class Candle:
    """단일 캔들 (OHLCV)
    
    빗썸 응답은 봉별 정보를 풍부하게 주지만,
    백테스트에서 쓸 5개 핵심 필드 + 시간 정보만 정형화.
    """
    market: str
    candle_datetime_utc: datetime    # UTC 기준 봉 시작 시각
    candle_datetime_kst: datetime    # KST 기준 봉 시작 시각
    opening_price: Decimal
    high_price: Decimal
    low_price: Decimal
    trade_price: Decimal             # 종가
    candle_acc_trade_volume: Decimal # 누적 거래량
    candle_acc_trade_price: Decimal  # 누적 거래대금
    timestamp: int                   # 마지막 거래 timestamp (ms)
    
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candle':
        return cls(
            market=data['market'],
            candle_datetime_utc=_parse_datetime(data.get('candle_date_time_utc')),
            candle_datetime_kst=_parse_datetime(data.get('candle_date_time_kst')),
            opening_price=_to_decimal(data.get('opening_price')),
            high_price=_to_decimal(data.get('high_price')),
            low_price=_to_decimal(data.get('low_price')),
            trade_price=_to_decimal(data.get('trade_price')),
            candle_acc_trade_volume=_to_decimal(data.get('candle_acc_trade_volume')),
            candle_acc_trade_price=_to_decimal(data.get('candle_acc_trade_price')),
            timestamp=int(data.get('timestamp', 0)),
            raw_data=data,
        )


# ============================================================
# 잔고 (Account)
# ============================================================

@dataclass(frozen=True)
class Account:
    """단일 자산 잔고"""
    currency: str                    # 'KRW', 'BTC' 등
    balance: Decimal                 # 주문가능 잔고
    locked: Decimal                  # 주문 중 묶인 잔고
    avg_buy_price: Decimal           # 평균 매수가
    avg_buy_price_modified: bool     # 평단가 수정 여부
    unit_currency: str               # 평단가 기준 화폐 (보통 'KRW')
    
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    @property
    def total(self) -> Decimal:
        """총 잔고 (가용 + 묶인)"""
        return self.balance + self.locked
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Account':
        return cls(
            currency=data['currency'],
            balance=_to_decimal(data.get('balance')),
            locked=_to_decimal(data.get('locked')),
            avg_buy_price=_to_decimal(data.get('avg_buy_price')),
            avg_buy_price_modified=bool(data.get('avg_buy_price_modified', False)),
            unit_currency=data.get('unit_currency', 'KRW'),
            raw_data=data,
        )


@dataclass(frozen=True)
class Balances:
    """전체 잔고 (편의용 컨테이너)"""
    accounts: List[Account]
    
    def get(self, currency: str) -> Optional[Account]:
        """특정 화폐의 잔고 조회"""
        for acc in self.accounts:
            if acc.currency == currency:
                return acc
        return None
    
    def get_balance(self, currency: str) -> Decimal:
        """특정 화폐의 가용 잔고 (없으면 0)"""
        acc = self.get(currency)
        return acc.balance if acc else Decimal('0')
    
    def get_total(self, currency: str) -> Decimal:
        """특정 화폐의 총 잔고 (가용 + 묶인)"""
        acc = self.get(currency)
        return acc.total if acc else Decimal('0')
    
    @property
    def krw_balance(self) -> Decimal:
        """가용 원화 (가장 자주 쓰는 값)"""
        return self.get_balance('KRW')
    
    @property
    def coin_currencies(self) -> List[str]:
        """보유 중인 코인 종류 (KRW 제외, 잔고 > 0인 것만)"""
        return [
            acc.currency for acc in self.accounts
            if acc.currency != 'KRW' and acc.total > 0
        ]
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> 'Balances':
        return cls(accounts=[Account.from_dict(d) for d in data])


# ============================================================
# 주문 (Order)
# ============================================================

@dataclass(frozen=True)
class Order:
    """주문 정보 (place_order 응답, get_order 응답 공통)"""
    uuid: str                        # 주문 고유 ID
    market: str
    side: OrderSide
    ord_type: OrderType
    state: OrderState
    
    price: Optional[Decimal]         # 주문 가격 (시장가 매도면 None)
    volume: Optional[Decimal]        # 주문 수량 (시장가 매수면 None)
    
    remaining_volume: Decimal        # 남은 수량
    executed_volume: Decimal         # 체결된 수량
    
    paid_fee: Decimal                # 사용된 수수료
    locked: Decimal                  # 묶인 금액
    
    trades_count: int                # 체결 건수
    
    created_at: Optional[datetime]   # 주문 생성 시각
    
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    @property
    def is_filled(self) -> bool:
        """완전 체결 여부"""
        return self.state == OrderState.DONE
    
    @property
    def is_open(self) -> bool:
        """미체결 (대기 중) 여부"""
        return self.state in (OrderState.WAIT, OrderState.WATCH)
    
    @property
    def is_canceled(self) -> bool:
        """취소 여부"""
        return self.state == OrderState.CANCEL
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        return cls(
            uuid=data['uuid'],
            market=data['market'],
            side=OrderSide(data['side']),
            ord_type=OrderType(data['ord_type']),
            state=OrderState.from_string(data['state']),
            price=_to_decimal(data.get('price')) if data.get('price') else None,
            volume=_to_decimal(data.get('volume')) if data.get('volume') else None,
            remaining_volume=_to_decimal(data.get('remaining_volume')),
            executed_volume=_to_decimal(data.get('executed_volume')),
            paid_fee=_to_decimal(data.get('paid_fee')),
            locked=_to_decimal(data.get('locked')),
            trades_count=int(data.get('trades_count', 0)),
            created_at=_parse_datetime(data.get('created_at')),
            raw_data=data,
        )


# ============================================================
# 호가창 (Orderbook) - 슬리피지 예측용
# ============================================================

@dataclass(frozen=True)
class OrderbookUnit:
    """호가창의 한 단계 (가격 + 수량)"""
    ask_price: Decimal   # 매도 호가
    bid_price: Decimal   # 매수 호가
    ask_size: Decimal    # 매도 수량
    bid_size: Decimal    # 매수 수량
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderbookUnit':
        return cls(
            ask_price=_to_decimal(data.get('ask_price')),
            bid_price=_to_decimal(data.get('bid_price')),
            ask_size=_to_decimal(data.get('ask_size')),
            bid_size=_to_decimal(data.get('bid_size')),
        )


@dataclass(frozen=True)
class Orderbook:
    """호가창 전체"""
    market: str
    timestamp: int
    total_ask_size: Decimal     # 총 매도 잔량
    total_bid_size: Decimal     # 총 매수 잔량
    units: List[OrderbookUnit]
    
    @property
    def best_ask(self) -> Optional[Decimal]:
        """최우선 매도호가 (시장가 매수 시 체결될 가격)"""
        return self.units[0].ask_price if self.units else None
    
    @property
    def best_bid(self) -> Optional[Decimal]:
        """최우선 매수호가 (시장가 매도 시 체결될 가격)"""
        return self.units[0].bid_price if self.units else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """호가 스프레드 (best_ask - best_bid)"""
        if self.best_ask and self.best_bid:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_pct(self) -> Optional[Decimal]:
        """스프레드 비율 (스프레드 / 중간가)"""
        if self.best_ask and self.best_bid:
            mid = (self.best_ask + self.best_bid) / 2
            return self.spread / mid
        return None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Orderbook':
        return cls(
            market=data['market'],
            timestamp=int(data.get('timestamp', 0)),
            total_ask_size=_to_decimal(data.get('total_ask_size')),
            total_bid_size=_to_decimal(data.get('total_bid_size')),
            units=[OrderbookUnit.from_dict(u) for u in data.get('orderbook_units', [])],
        )
