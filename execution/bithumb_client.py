"""
빗썸 API 2.0 클라이언트

설계 원칙:
1. 모든 통신은 _request() 메서드를 거침 (단일 진입점)
2. 자동 재시도 (지수 백오프, 재시도 가능 에러만)
3. 모든 요청에 타임아웃 강제 (무한 대기 금지)
4. 모든 호출은 로깅 (감사 추적)
5. 응답은 dataclass로 변환 (타입 안전)
6. 시크릿 키는 외부 노출 금지 (repr 마스킹)

빗썸 API 2.0 문서: https://apidocs.bithumb.com/
인증 방식: JWT (Bearer 토큰)
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import JWTAuthenticator
from execution.errors import (
    BithumbError,
    NetworkError,
    RetryableError,
    classify_http_error,
)
from execution.models import (
    Account,
    Balances,
    Candle,
    Market,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    Orderbook,
    Ticker,
)

logger = logging.getLogger(__name__)


# ============================================================
# 상수 정의
# ============================================================

DEFAULT_BASE_URL = "https://api.bithumb.com"
DEFAULT_TIMEOUT = 10.0       # 초 (단일 요청 타임아웃)
DEFAULT_MAX_RETRIES = 3      # 재시도 횟수
DEFAULT_BACKOFF_BASE = 1.0   # 지수 백오프 베이스 (초)
DEFAULT_BACKOFF_MAX = 30.0   # 최대 대기 시간 (초)


# 빗썸 봉 단위 정의
CANDLE_UNITS = {
    '1m': ('minutes', 1),
    '3m': ('minutes', 3),
    '5m': ('minutes', 5),
    '10m': ('minutes', 10),
    '15m': ('minutes', 15),
    '30m': ('minutes', 30),
    '60m': ('minutes', 60),
    '1h': ('minutes', 60),     # alias
    '240m': ('minutes', 240),
    '4h': ('minutes', 240),    # alias
    '1d': ('days', None),
    '1w': ('weeks', None),
    '1M': ('months', None),
}


# ============================================================
# 클라이언트 본체
# ============================================================

class BithumbClient:
    """빗썸 API 2.0 클라이언트
    
    사용법:
        from utils.config import get_config
        cfg = get_config()
        client = BithumbClient(cfg.bithumb_api_key, cfg.bithumb_secret_key)
        
        # Public API (인증 불필요)
        ticker = client.get_ticker('KRW-BTC')
        candles = client.get_candles('KRW-BTC', unit='1d', count=200)
        
        # Private API (인증 필요)
        balances = client.get_balances()
        order = client.market_buy('KRW-BTC', krw_amount=Decimal('10000'))
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        dry_run: bool = False,
    ):
        """
        Args:
            api_key: 빗썸 API Access Key
            secret_key: 빗썸 API Secret Key
            base_url: API 기본 URL
            timeout: 단일 요청 타임아웃 (초)
            max_retries: 재시도 가능 에러 시 최대 재시도 횟수
            dry_run: True면 실제 주문을 보내지 않고 로그만 남김 (개발용)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = dry_run
        
        # JWT 인증 헬퍼
        self._auth = JWTAuthenticator(api_key, secret_key)
        
        # Session 재사용 (TCP 연결 유지 → 빠른 응답)
        self._session = requests.Session()
        self._session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'QuantSystem/1.0',
        })
        
        logger.info(
            f"BithumbClient 초기화: base_url={self.base_url}, "
            f"timeout={self.timeout}s, max_retries={self.max_retries}, "
            f"dry_run={self.dry_run}"
        )
    
    def __repr__(self) -> str:
        return (
            f"<BithumbClient base_url={self.base_url} "
            f"dry_run={self.dry_run} auth={self._auth!r}>"
        )
    
    def close(self) -> None:
        """세션 종료 (프로그램 종료 시 호출 권장)"""
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # ========================================================
    # 핵심: HTTP 요청 메서드 (모든 호출의 단일 진입점)
    # ========================================================
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        requires_auth: bool = False,
    ) -> Union[Dict, List]:
        """HTTP 요청 실행 (재시도 + 에러 분류 + 로깅 내장)
        
        Args:
            method: 'GET', 'POST', 'DELETE'
            endpoint: '/v1/ticker' 같은 경로
            params: 요청 파라미터
            requires_auth: True면 JWT 인증 헤더 추가
        
        Returns:
            파싱된 JSON 응답
        
        Raises:
            BithumbError 또는 그 하위 클래스
        """
        url = f"{self.base_url}{endpoint}"
        
        # 파라미터에서 None 값 제거 (빗썸은 null을 안 받는 경우 있음)
        clean_params = self._clean_params(params)
        
        # 인증 헤더 생성 (필요시)
        headers = {}
        if requires_auth:
            # 빗썸은 GET은 query string, POST/DELETE는 body에 해시 생성
            headers.update(self._auth.make_auth_header(clean_params))
        
        # 재시도 루프
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._execute_request(
                    method=method,
                    url=url,
                    params=clean_params,
                    headers=headers,
                    endpoint=endpoint,
                )
            
            except RetryableError as e:
                last_error = e
                if attempt >= self.max_retries:
                    logger.error(
                        f"재시도 한도 도달 ({attempt + 1}회): {endpoint} - {e}"
                    )
                    raise
                
                wait_time = self._calculate_backoff(attempt)
                logger.warning(
                    f"재시도 가능 에러 (시도 {attempt + 1}/{self.max_retries + 1}): "
                    f"{e} | {wait_time:.1f}초 후 재시도"
                )
                time.sleep(wait_time)
            
            except BithumbError:
                # 영구 에러는 즉시 raise (재시도 무의미)
                raise
        
        # 여기까지 오면 안 됨 (이론상)
        if last_error:
            raise last_error
        raise BithumbError(f"알 수 없는 실패: {endpoint}")
    
    def _execute_request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any],
        headers: Dict[str, str],
        endpoint: str,
    ) -> Union[Dict, List]:
        """실제 HTTP 요청 실행 (재시도 없음, _request에서 호출)"""
        # 빗썸 V2: GET은 query string, POST/DELETE는 JSON body
        request_kwargs = {
            'method': method,
            'url': url,
            'headers': headers,
            'timeout': self.timeout,
        }
        
        if method == 'GET':
            request_kwargs['params'] = params
        else:
            # POST, DELETE는 body에 JSON
            if params:
                request_kwargs['json'] = params
        
        logger.debug(f"{method} {endpoint} params={params}")
        
        try:
            response = self._session.request(**request_kwargs)
        except requests.exceptions.Timeout as e:
            raise NetworkError(f"요청 타임아웃: {endpoint}") from e
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"연결 실패: {endpoint} - {e}") from e
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"요청 실패: {endpoint} - {e}") from e
        
        # 응답 파싱
        try:
            data = response.json()
        except ValueError:
            data = None
        
        # HTTP 상태 확인
        if not response.ok:
            error = classify_http_error(
                status_code=response.status_code,
                response_data=data,
                response_text=response.text,
            )
            logger.error(f"{method} {endpoint} 실패: {error}")
            raise error
        
        # 일부 빗썸 API는 HTTP 200 이더라도 body에 error 객체를 담아 반환.
        # 이 경우도 명시적으로 에러 처리해서 "string indices must be integers" 같은
        # 후속 에러를 방지함.
        if isinstance(data, dict) and 'error' in data:
            error = classify_http_error(
                status_code=response.status_code,
                response_data=data,
                response_text=response.text,
            )
            logger.error(f"{method} {endpoint} 실패 (HTTP 200 + body error): {error}")
            raise error
        
        logger.debug(f"{method} {endpoint} 성공: {response.status_code}")
        return data
    
    @staticmethod
    def _clean_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """None 값 제거 + Decimal/float을 str로 변환
        
        빗썸 API는 일부 필드에 null을 받지 않고,
        Decimal은 JSON 직렬화 안 되므로 str 변환 필요.
        """
        if not params:
            return {}
        
        cleaned = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, Decimal):
                cleaned[k] = str(v)
            elif isinstance(v, float):
                # float은 정밀도 손실 위험 → Decimal 사용 강제
                raise ValueError(
                    f"파라미터 '{k}'에 float 사용 금지. Decimal 사용하세요. "
                    f"값: {v}"
                )
            else:
                cleaned[k] = v
        return cleaned
    
    @staticmethod
    def _calculate_backoff(attempt: int) -> float:
        """지수 백오프 시간 계산 (1초 → 2초 → 4초 → ...)"""
        wait = DEFAULT_BACKOFF_BASE * (2 ** attempt)
        return min(wait, DEFAULT_BACKOFF_MAX)
    
    # ========================================================
    # 1. Public API: 마켓 정보 (인증 불필요)
    # ========================================================
    
    def get_markets(self, is_details: bool = False) -> List[Market]:
        """거래 가능 종목 목록 조회
        
        Args:
            is_details: True면 유의종목 등 추가 정보 포함
        
        Returns:
            Market 리스트
        """
        data = self._request(
            'GET',
            '/v1/market/all',
            params={'isDetails': 'true' if is_details else 'false'},
            requires_auth=False,
        )
        return [Market.from_dict(d) for d in data]
    
    def get_krw_markets(self) -> List[Market]:
        """KRW 마켓만 필터링 (대부분의 사용 케이스)"""
        markets = self.get_markets()
        return [m for m in markets if m.market.startswith('KRW-')]
    
    # ========================================================
    # 2. Public API: 현재가 (인증 불필요)
    # ========================================================
    
    def get_ticker(self, market: str) -> Ticker:
        """단일 종목 현재가 조회"""
        data = self._request(
            'GET',
            '/v1/ticker',
            params={'markets': market},
            requires_auth=False,
        )
        if not data:
            raise BithumbError(f"종목 없음: {market}")
        return Ticker.from_dict(data[0])
    
    def get_tickers(self, markets: List[str]) -> List[Ticker]:
        """여러 종목 현재가 일괄 조회 (효율적)"""
        if not markets:
            return []
        data = self._request(
            'GET',
            '/v1/ticker',
            params={'markets': ','.join(markets)},
            requires_auth=False,
        )
        return [Ticker.from_dict(d) for d in data]
    
    # ========================================================
    # 3. Public API: 캔들 (OHLCV) - 백테스트 핵심
    # ========================================================
    
    def get_candles(
        self,
        market: str,
        unit: str = '1d',
        count: int = 200,
        to: Optional[str] = None,
    ) -> List[Candle]:
        """OHLCV 캔들 조회
        
        Args:
            market: 'KRW-BTC' 같은 마켓 코드
            unit: '1m', '3m', '5m', '10m', '15m', '30m', '60m', '240m', '1d', '1w', '1M'
            count: 가져올 봉 개수 (최대 200)
            to: 마지막 봉 시각 (ISO 8601, 페이지네이션용)
        
        Returns:
            Candle 리스트 (최신 봉이 앞쪽)
        """
        if unit not in CANDLE_UNITS:
            raise ValueError(
                f"잘못된 unit: '{unit}'. 가능값: {list(CANDLE_UNITS.keys())}"
            )
        
        unit_type, unit_value = CANDLE_UNITS[unit]
        
        # 엔드포인트 결정
        if unit_type == 'minutes':
            endpoint = f'/v1/candles/minutes/{unit_value}'
        elif unit_type == 'days':
            endpoint = '/v1/candles/days'
        elif unit_type == 'weeks':
            endpoint = '/v1/candles/weeks'
        elif unit_type == 'months':
            endpoint = '/v1/candles/months'
        else:
            raise ValueError(f"지원하지 않는 unit_type: {unit_type}")
        
        params = {
            'market': market,
            'count': min(count, 200),  # 빗썸 최대 200
        }
        if to:
            params['to'] = to
        
        data = self._request('GET', endpoint, params=params, requires_auth=False)
        
        # 응답 형식 검증 (빗썸이 가끔 에러를 200 OK로 보낼 경우 대비)
        if not isinstance(data, list):
            # dict가 오면 보통 에러 응답
            if isinstance(data, dict) and 'error' in data:
                error_info = data.get('error', {})
                error_name = error_info.get('name', 'unknown')
                error_msg = error_info.get('message', str(data))
                raise BithumbError(
                    f"빗썸 캔들 조회 에러: {error_name} - {error_msg} "
                    f"(market={market}, unit={unit}, to={params.get('to')})"
                )
            raise BithumbError(
                f"예상치 못한 응답 형식: list 기대, {type(data).__name__} 받음. "
                f"응답: {str(data)[:200]}"
            )
        
        # 빈 응답은 정상 (예: 그 기간에 데이터 없음)
        if not data:
            return []
        
        # 각 항목이 dict인지 확인 (이전 string indices 에러 방지)
        if not all(isinstance(item, dict) for item in data):
            raise BithumbError(
                f"응답 데이터 형식 오류: 각 항목이 dict가 아님. "
                f"첫 항목: {type(data[0]).__name__}, 값: {str(data[0])[:100]}"
            )
        
        return [Candle.from_dict(d) for d in data]
    
    # ========================================================
    # 4. Public API: 호가창
    # ========================================================
    
    def get_orderbook(self, market: str) -> Orderbook:
        """호가창 조회 (슬리피지 예측용)"""
        data = self._request(
            'GET',
            '/v1/orderbook',
            params={'markets': market},
            requires_auth=False,
        )
        if not data:
            raise BithumbError(f"호가창 조회 실패: {market}")
        return Orderbook.from_dict(data[0])
    
    # ========================================================
    # 5. Private API: 잔고 (인증 필요)
    # ========================================================
    
    def get_balances(self) -> Balances:
        """전체 잔고 조회"""
        data = self._request(
            'GET',
            '/v1/accounts',
            requires_auth=True,
        )
        return Balances.from_list(data)
    
    # ========================================================
    # 6. Private API: 주문 실행
    # ========================================================
    
    def place_order(
        self,
        market: str,
        side: OrderSide,
        ord_type: OrderType,
        volume: Optional[Decimal] = None,
        price: Optional[Decimal] = None,
    ) -> Order:
        """저수준 주문 실행 (직접 호출은 비추천 — market_buy/market_sell/limit_order 사용)
        
        빗썸 V2 주문 규칙:
        - 지정가(limit): volume + price 둘 다 필수
        - 시장가 매수(price): price만 (금액)
        - 시장가 매도(market): volume만 (수량)
        """
        # 입력 검증
        if not isinstance(volume, (Decimal, type(None))):
            raise ValueError(f"volume은 Decimal이어야 합니다: {type(volume)}")
        if not isinstance(price, (Decimal, type(None))):
            raise ValueError(f"price는 Decimal이어야 합니다: {type(price)}")
        
        # ord_type별 파라미터 검증
        if ord_type == OrderType.LIMIT:
            if volume is None or price is None:
                raise ValueError("지정가 주문은 volume과 price 모두 필요")
        elif ord_type == OrderType.MARKET_BUY:
            if price is None:
                raise ValueError("시장가 매수는 price(금액)가 필요")
            if side != OrderSide.BID:
                raise ValueError("시장가 매수(price)는 side='bid'여야 함")
        elif ord_type == OrderType.MARKET_SELL:
            if volume is None:
                raise ValueError("시장가 매도는 volume(수량)이 필요")
            if side != OrderSide.ASK:
                raise ValueError("시장가 매도(market)는 side='ask'여야 함")
        
        params = {
            'market': market,
            'side': side.value,
            'ord_type': ord_type.value,
        }
        if volume is not None:
            params['volume'] = str(volume)
        if price is not None:
            params['price'] = str(price)
        
        # Dry-run 모드: 실제 주문 보내지 않음
        if self.dry_run:
            logger.warning(
                f"[DRY RUN] 주문 시뮬레이션: {market} {side.value} {ord_type.value} "
                f"volume={volume} price={price}"
            )
            # 가짜 Order 객체 반환
            return Order(
                uuid='dry-run-' + str(time.time()),
                market=market,
                side=side,
                ord_type=ord_type,
                state=OrderState.WAIT,
                price=price,
                volume=volume,
                remaining_volume=volume or Decimal('0'),
                executed_volume=Decimal('0'),
                paid_fee=Decimal('0'),
                locked=Decimal('0'),
                trades_count=0,
                created_at=None,
                raw_data={'dry_run': True, **params},
            )
        
        # 거래 로그 (별도 로거에 영구 기록)
        trade_logger = logging.getLogger('trade')
        trade_logger.info(
            f"ORDER_REQUEST market={market} side={side.value} "
            f"ord_type={ord_type.value} volume={volume} price={price}"
        )
        
        data = self._request('POST', '/v1/orders', params=params, requires_auth=True)
        order = Order.from_dict(data)
        
        trade_logger.info(f"ORDER_PLACED uuid={order.uuid} state={order.state.value}")
        
        return order
    
    def market_buy(self, market: str, krw_amount: Decimal) -> Order:
        """시장가 매수 (금액 기준 — 빗썸 V2 규칙)
        
        Args:
            market: 'KRW-BTC' 같은 마켓
            krw_amount: 매수에 쓸 원화 금액 (Decimal)
        
        예: client.market_buy('KRW-BTC', Decimal('10000'))
            → 1만원어치 BTC 시장가 매수
        """
        if not isinstance(krw_amount, Decimal):
            raise ValueError(f"krw_amount는 Decimal이어야 합니다: {type(krw_amount)}")
        if krw_amount < Decimal('5000'):
            raise ValueError(f"최소 주문 금액 5,000원 미달: {krw_amount}")
        
        return self.place_order(
            market=market,
            side=OrderSide.BID,
            ord_type=OrderType.MARKET_BUY,
            price=krw_amount,
        )
    
    def market_sell(self, market: str, volume: Decimal) -> Order:
        """시장가 매도 (수량 기준 — 빗썸 V2 규칙)
        
        Args:
            market: 'KRW-BTC' 같은 마켓
            volume: 매도할 코인 수량 (Decimal)
        
        예: client.market_sell('KRW-BTC', Decimal('0.001'))
            → BTC 0.001개 시장가 매도
        """
        if not isinstance(volume, Decimal):
            raise ValueError(f"volume은 Decimal이어야 합니다: {type(volume)}")
        if volume <= Decimal('0'):
            raise ValueError(f"매도 수량은 0보다 커야 함: {volume}")
        
        return self.place_order(
            market=market,
            side=OrderSide.ASK,
            ord_type=OrderType.MARKET_SELL,
            volume=volume,
        )
    
    def limit_order(
        self,
        market: str,
        side: OrderSide,
        price: Decimal,
        volume: Decimal,
    ) -> Order:
        """지정가 주문"""
        if not isinstance(price, Decimal):
            raise ValueError("price는 Decimal이어야 합니다")
        if not isinstance(volume, Decimal):
            raise ValueError("volume은 Decimal이어야 합니다")
        if price <= Decimal('0') or volume <= Decimal('0'):
            raise ValueError("가격과 수량은 양수여야 함")
        
        return self.place_order(
            market=market,
            side=side,
            ord_type=OrderType.LIMIT,
            price=price,
            volume=volume,
        )
    
    # ========================================================
    # 7. Private API: 주문 조회/취소
    # ========================================================
    
    def get_order(self, uuid: str) -> Order:
        """단일 주문 상태 조회"""
        data = self._request(
            'GET',
            '/v1/order',
            params={'uuid': uuid},
            requires_auth=True,
        )
        return Order.from_dict(data)
    
    def get_open_orders(self, market: Optional[str] = None) -> List[Order]:
        """미체결 주문 목록 조회"""
        params = {'state': 'wait'}
        if market:
            params['market'] = market
        
        data = self._request(
            'GET',
            '/v1/orders',
            params=params,
            requires_auth=True,
        )
        return [Order.from_dict(d) for d in data]
    
    def cancel_order(self, uuid: str) -> Order:
        """주문 취소"""
        if self.dry_run:
            logger.warning(f"[DRY RUN] 주문 취소 시뮬레이션: uuid={uuid}")
            return Order(
                uuid=uuid,
                market='',
                side=OrderSide.BID,
                ord_type=OrderType.LIMIT,
                state=OrderState.CANCEL,
                price=None,
                volume=None,
                remaining_volume=Decimal('0'),
                executed_volume=Decimal('0'),
                paid_fee=Decimal('0'),
                locked=Decimal('0'),
                trades_count=0,
                created_at=None,
                raw_data={'dry_run': True},
            )
        
        trade_logger = logging.getLogger('trade')
        trade_logger.info(f"ORDER_CANCEL_REQUEST uuid={uuid}")
        
        data = self._request(
            'DELETE',
            '/v1/order',
            params={'uuid': uuid},
            requires_auth=True,
        )
        order = Order.from_dict(data)
        
        trade_logger.info(f"ORDER_CANCELED uuid={uuid} state={order.state.value}")
        return order
    
    # ========================================================
    # 8. 헬스체크
    # ========================================================
    
    def health_check(self) -> bool:
        """API 연결 상태 확인 (Public API로 가벼운 호출)
        
        Returns:
            True: 정상 응답
            False: 실패
        """
        try:
            self._request('GET', '/v1/market/all', requires_auth=False)
            return True
        except BithumbError as e:
            logger.warning(f"헬스체크 실패: {e}")
            return False
    
    def auth_check(self) -> bool:
        """인증 동작 확인 (잔고 조회로 검증)
        
        Returns:
            True: 인증 정상
            False: 인증 실패
        """
        try:
            self.get_balances()
            return True
        except BithumbError as e:
            logger.warning(f"인증 체크 실패: {e}")
            return False
