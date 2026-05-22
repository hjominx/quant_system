"""
빗썸 클라이언트 검증 테스트 (Mock 응답 기반)

실제 빗썸 서버에 연결하지 않고 클라이언트 로직 자체를 검증.
실제 연결 테스트는 test_bithumb_live.py 에서 별도로 수행 (사용자 환경에서).

검증 시나리오 16가지:
1. 클라이언트 초기화
2. Public API 응답 파싱 (ticker)
3. 캔들 데이터 파싱 (OHLCV)
4. 호가창 파싱
5. 잔고 응답 파싱
6. 시장가 매수 (금액 기반) 파라미터 검증
7. 시장가 매도 (수량 기반) 파라미터 검증
8. 지정가 주문 파라미터 검증
9. Decimal 강제 (float 거부)
10. Dry-run 모드 동작
11. 재시도 로직 (네트워크 에러)
12. 영구 에러는 즉시 raise (재시도 안 함)
13. 시크릿 노출 방지
14. 잘못된 ord_type 조합 거부
15. Config 거래 모드에 따른 dry-run 자동 설정
16. 실주문 이중 안전장치
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from decimal import Decimal
from unittest.mock import patch, MagicMock

from execution.bithumb_client import BithumbClient
from execution.models import (
    OrderSide, OrderType, OrderState,
    Ticker, Candle, Account, Balances, Order, Orderbook,
)
from execution.errors import (
    BithumbError,
    AuthenticationError,
    InvalidRequestError,
    NetworkError,
    ServerError,
    RetryableError,
)
from utils.config import Config, TradingMode


# 테스트용 더미 키
TEST_API_KEY = "test_api_key_abcdef1234567890ABCDEF"
TEST_SECRET_KEY = "test_secret_key_xyz9876543210ZYXWVU"


def make_config(trading_mode: TradingMode) -> Config:
    """테스트용 Config 생성"""
    return Config(
        bithumb_api_key=TEST_API_KEY,
        bithumb_secret_key=TEST_SECRET_KEY,
        trading_mode=trading_mode,
        daily_loss_limit=0.03,
        weekly_loss_limit=0.07,
        max_drawdown_limit=0.15,
        max_position_per_coin=0.25,
        max_total_crypto=0.80,
        initial_capital=10_000_000,
    )


# ============================================================
# Mock 응답 데이터 (빗썸 V2 실제 형식)
# ============================================================

MOCK_TICKER_RESPONSE = [{
    'market': 'KRW-BTC',
    'trade_date': '20240115',
    'trade_time': '090000',
    'trade_price': 60000000,
    'opening_price': 59000000,
    'high_price': 61000000,
    'low_price': 58500000,
    'prev_closing_price': 59500000,
    'change': 'RISE',
    'change_price': 500000,
    'change_rate': 0.0084,
    'signed_change_price': 500000,
    'signed_change_rate': 0.0084,
    'acc_trade_volume_24h': 1234.5678,
    'acc_trade_price_24h': 74000000000,
    'timestamp': 1705305600000,
}]

MOCK_CANDLE_RESPONSE = [
    {
        'market': 'KRW-BTC',
        'candle_date_time_utc': '2024-01-15T00:00:00',
        'candle_date_time_kst': '2024-01-15T09:00:00',
        'opening_price': 59000000,
        'high_price': 61000000,
        'low_price': 58500000,
        'trade_price': 60000000,
        'timestamp': 1705305600000,
        'candle_acc_trade_volume': 1234.5678,
        'candle_acc_trade_price': 74000000000,
        'unit': 1,
    },
    {
        'market': 'KRW-BTC',
        'candle_date_time_utc': '2024-01-14T00:00:00',
        'candle_date_time_kst': '2024-01-14T09:00:00',
        'opening_price': 58000000,
        'high_price': 59500000,
        'low_price': 57500000,
        'trade_price': 59000000,
        'timestamp': 1705219200000,
        'candle_acc_trade_volume': 987.654,
        'candle_acc_trade_price': 58000000000,
        'unit': 1,
    },
]

MOCK_ORDERBOOK_RESPONSE = [{
    'market': 'KRW-BTC',
    'timestamp': 1705305600000,
    'total_ask_size': 12.345,
    'total_bid_size': 23.456,
    'orderbook_units': [
        {'ask_price': 60000000, 'bid_price': 59950000, 'ask_size': 0.5, 'bid_size': 0.3},
        {'ask_price': 60050000, 'bid_price': 59900000, 'ask_size': 1.0, 'bid_size': 0.8},
    ],
}]

MOCK_BALANCES_RESPONSE = [
    {
        'currency': 'KRW',
        'balance': '5000000',
        'locked': '0',
        'avg_buy_price': '0',
        'avg_buy_price_modified': False,
        'unit_currency': 'KRW',
    },
    {
        'currency': 'BTC',
        'balance': '0.05',
        'locked': '0.01',
        'avg_buy_price': '55000000',
        'avg_buy_price_modified': False,
        'unit_currency': 'KRW',
    },
]

MOCK_ORDER_RESPONSE = {
    'uuid': 'abc-123-def-456',
    'market': 'KRW-BTC',
    'side': 'bid',
    'ord_type': 'limit',
    'state': 'wait',
    'price': '60000000',
    'volume': '0.001',
    'remaining_volume': '0.001',
    'executed_volume': '0',
    'paid_fee': '0',
    'locked': '60000',
    'trades_count': 0,
    'created_at': '2024-01-15T09:00:00+09:00',
}


# ============================================================
# Mock 응답 생성 헬퍼
# ============================================================

def make_mock_response(json_data, status_code=200):
    """requests.Response 모킹"""
    mock_resp = MagicMock()
    mock_resp.ok = (200 <= status_code < 300)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = str(json_data)
    return mock_resp


# ============================================================
# 테스트 함수들
# ============================================================

def test_client_initialization():
    print("\n[테스트 1] 클라이언트 초기화")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
    
    assert client.base_url == "https://api.bithumb.com"
    assert client.timeout == 10.0
    assert client.max_retries == 3
    assert client.dry_run == True
    assert client.allow_live_orders == False
    
    print(f"  ✓ 기본 설정 정상")
    print(f"  ✓ base_url: {client.base_url}")
    print(f"  ✓ timeout: {client.timeout}s")
    print(f"  ✓ max_retries: {client.max_retries}")
    print(f"  ✓ 기본 주문 모드: dry-run")


def test_ticker_parsing():
    print("\n[테스트 2] Ticker 응답 파싱")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
    
    with patch.object(client._session, 'request') as mock_req:
        mock_req.return_value = make_mock_response(MOCK_TICKER_RESPONSE)
        ticker = client.get_ticker('KRW-BTC')
    
    assert isinstance(ticker, Ticker)
    assert ticker.market == 'KRW-BTC'
    assert ticker.trade_price == Decimal('60000000')
    assert isinstance(ticker.trade_price, Decimal), "가격은 반드시 Decimal"
    assert ticker.change == 'RISE'
    
    print(f"  ✓ Ticker 객체로 변환됨")
    print(f"  ✓ market: {ticker.market}")
    print(f"  ✓ trade_price: {ticker.trade_price} (타입: {type(ticker.trade_price).__name__})")
    print(f"  ✓ change: {ticker.change}")


def test_candle_parsing():
    print("\n[테스트 3] Candle (OHLCV) 응답 파싱")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
    
    with patch.object(client._session, 'request') as mock_req:
        mock_req.return_value = make_mock_response(MOCK_CANDLE_RESPONSE)
        candles = client.get_candles('KRW-BTC', unit='1d', count=200)
    
    assert len(candles) == 2
    assert all(isinstance(c, Candle) for c in candles)
    assert candles[0].opening_price == Decimal('59000000')
    assert candles[0].trade_price == Decimal('60000000')
    assert candles[0].candle_datetime_utc is not None
    
    print(f"  ✓ {len(candles)}개 캔들 파싱됨")
    print(f"  ✓ 모든 가격이 Decimal 타입")
    print(f"  ✓ 첫 캔들 OHLC: {candles[0].opening_price} / "
          f"{candles[0].high_price} / {candles[0].low_price} / "
          f"{candles[0].trade_price}")


def test_orderbook_parsing():
    print("\n[테스트 4] 호가창 응답 파싱")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
    
    with patch.object(client._session, 'request') as mock_req:
        mock_req.return_value = make_mock_response(MOCK_ORDERBOOK_RESPONSE)
        orderbook = client.get_orderbook('KRW-BTC')
    
    assert isinstance(orderbook, Orderbook)
    assert orderbook.best_ask == Decimal('60000000')
    assert orderbook.best_bid == Decimal('59950000')
    assert orderbook.spread == Decimal('50000')
    
    print(f"  ✓ best_ask: {orderbook.best_ask:,}")
    print(f"  ✓ best_bid: {orderbook.best_bid:,}")
    print(f"  ✓ spread: {orderbook.spread:,} ({orderbook.spread_pct*100:.3f}%)")


def test_balances_parsing():
    print("\n[테스트 5] 잔고 응답 파싱")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
    
    with patch.object(client._session, 'request') as mock_req:
        mock_req.return_value = make_mock_response(MOCK_BALANCES_RESPONSE)
        balances = client.get_balances()
    
    assert isinstance(balances, Balances)
    assert balances.krw_balance == Decimal('5000000')
    assert balances.get_balance('BTC') == Decimal('0.05')
    assert balances.get_total('BTC') == Decimal('0.06')  # 가용 + 묶인
    
    print(f"  ✓ KRW 가용: {balances.krw_balance:,}원")
    print(f"  ✓ BTC 가용: {balances.get_balance('BTC')}")
    print(f"  ✓ BTC 총계: {balances.get_total('BTC')} (가용 + 묶인)")
    print(f"  ✓ 보유 코인: {balances.coin_currencies}")


def test_market_buy_amount_based():
    print("\n[테스트 6] 시장가 매수 — 금액 기반 (빗썸 V2 규칙)")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY, dry_run=False, allow_live_orders=True)
    
    captured_params = {}
    
    def capture_request(**kwargs):
        captured_params.update(kwargs)
        # 가짜 응답
        return make_mock_response({
            **MOCK_ORDER_RESPONSE,
            'ord_type': 'price',
            'side': 'bid',
            'price': '10000',
            'volume': None,
        })
    
    with patch.object(client._session, 'request', side_effect=capture_request):
        order = client.market_buy('KRW-BTC', krw_amount=Decimal('10000'))
    
    # 보낸 파라미터 확인
    sent_json = captured_params.get('json', {})
    assert sent_json['ord_type'] == 'price', "시장가 매수는 ord_type='price'"
    assert sent_json['side'] == 'bid'
    assert sent_json['price'] == '10000'
    assert 'volume' not in sent_json, "시장가 매수는 volume 보내지 않음"
    
    print(f"  ✓ ord_type='price' (시장가 매수)")
    print(f"  ✓ price='10000' (금액)")
    print(f"  ✓ volume 미전송 (빗썸 V2 규칙)")


def test_market_sell_volume_based():
    print("\n[테스트 7] 시장가 매도 — 수량 기반 (빗썸 V2 규칙)")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY, dry_run=False, allow_live_orders=True)
    
    captured_params = {}
    
    def capture_request(**kwargs):
        captured_params.update(kwargs)
        return make_mock_response({
            **MOCK_ORDER_RESPONSE,
            'ord_type': 'market',
            'side': 'ask',
            'volume': '0.001',
            'price': None,
        })
    
    with patch.object(client._session, 'request', side_effect=capture_request):
        order = client.market_sell('KRW-BTC', volume=Decimal('0.001'))
    
    sent_json = captured_params.get('json', {})
    assert sent_json['ord_type'] == 'market', "시장가 매도는 ord_type='market'"
    assert sent_json['side'] == 'ask'
    assert sent_json['volume'] == '0.001'
    assert 'price' not in sent_json, "시장가 매도는 price 보내지 않음"
    
    print(f"  ✓ ord_type='market' (시장가 매도)")
    print(f"  ✓ volume='0.001' (수량)")
    print(f"  ✓ price 미전송 (빗썸 V2 규칙)")


def test_limit_order():
    print("\n[테스트 8] 지정가 주문 — volume + price 모두 필요")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY, dry_run=False, allow_live_orders=True)
    
    captured_params = {}
    
    def capture_request(**kwargs):
        captured_params.update(kwargs)
        return make_mock_response(MOCK_ORDER_RESPONSE)
    
    with patch.object(client._session, 'request', side_effect=capture_request):
        order = client.limit_order(
            'KRW-BTC', OrderSide.BID,
            price=Decimal('60000000'), volume=Decimal('0.001')
        )
    
    sent_json = captured_params.get('json', {})
    assert sent_json['ord_type'] == 'limit'
    assert sent_json['side'] == 'bid'
    assert sent_json['price'] == '60000000'
    assert sent_json['volume'] == '0.001'
    
    print(f"  ✓ ord_type='limit'")
    print(f"  ✓ price + volume 모두 전송")


def test_decimal_enforcement():
    print("\n[테스트 9] Decimal 강제 — float 거부")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
    
    # float 사용 시 ValueError 발생해야 함
    try:
        client.market_buy('KRW-BTC', krw_amount=10000.0)  # float
        raise AssertionError("float이 통과됨!")
    except ValueError as e:
        print(f"  ✓ market_buy에 float 거부: {str(e)[:60]}")
    
    try:
        client.market_sell('KRW-BTC', volume=0.001)  # float
        raise AssertionError("float이 통과됨!")
    except ValueError as e:
        print(f"  ✓ market_sell에 float 거부: {str(e)[:60]}")


def test_dry_run_mode():
    print("\n[테스트 10] Dry-run 모드 — 실제 주문 안 보냄")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY, dry_run=True)
    
    # Mock이 아님에도 dry_run이라 네트워크 호출 안 일어남
    with patch.object(client._session, 'request') as mock_req:
        order = client.market_buy('KRW-BTC', krw_amount=Decimal('10000'))
    
    # 네트워크 호출 0번
    assert mock_req.call_count == 0, "dry_run인데 네트워크 호출이 발생함!"
    assert order.uuid.startswith('dry-run-')
    assert order.market == 'KRW-BTC'
    
    print(f"  ✓ 네트워크 호출 0번 (dry_run 작동)")
    print(f"  ✓ 가짜 주문 반환: uuid={order.uuid[:20]}...")


def test_retry_on_network_error():
    print("\n[테스트 11] 네트워크 에러 시 재시도 (지수 백오프)")
    print("-" * 60)
    
    import requests as req_module
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY, max_retries=2)
    
    call_count = [0]
    
    def flaky_request(**kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            # 처음 2번은 실패
            raise req_module.exceptions.ConnectionError("일시적 네트워크 오류")
        # 3번째 성공
        return make_mock_response(MOCK_TICKER_RESPONSE)
    
    with patch.object(client._session, 'request', side_effect=flaky_request):
        with patch('time.sleep'):  # 실제 대기 시간 스킵
            ticker = client.get_ticker('KRW-BTC')
    
    assert call_count[0] == 3, f"재시도 횟수 예상치 다름: {call_count[0]}"
    assert ticker.market == 'KRW-BTC'
    
    print(f"  ✓ 2번 실패 → 3번째 성공 (총 {call_count[0]}회 시도)")
    print(f"  ✓ 자동 재시도 정상 작동")


def test_no_retry_on_permanent_error():
    print("\n[테스트 12] 영구 에러는 즉시 raise (재시도 안 함)")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY, max_retries=3)
    
    call_count = [0]
    
    def auth_failure(**kwargs):
        call_count[0] += 1
        return make_mock_response(
            {'error': {'name': 'invalid_access_key', 'message': '인증 실패'}},
            status_code=401,
        )
    
    with patch.object(client._session, 'request', side_effect=auth_failure):
        with patch('time.sleep'):
            try:
                client.get_balances()
                raise AssertionError("인증 에러인데 통과됨!")
            except AuthenticationError as e:
                pass
    
    # 인증 에러는 재시도 안 해야 함 (1번만 호출)
    assert call_count[0] == 1, f"영구 에러에 재시도 발생: {call_count[0]}회"
    
    print(f"  ✓ 인증 에러 1번만 호출 (재시도 안 함)")
    print(f"  ✓ AuthenticationError 즉시 raise")


def test_secret_not_exposed():
    print("\n[테스트 13] 시크릿이 repr에 노출되지 않음")
    print("-" * 60)
    
    secret = "VERY_SECRET_DO_NOT_LEAK_1234567890XX"
    client = BithumbClient(TEST_API_KEY, secret)
    
    repr_out = repr(client)
    if secret in repr_out:
        raise AssertionError(f"시크릿 노출! {repr_out}")
    
    print(f"  ✓ repr 출력: {repr_out}")
    print(f"  ✓ 시크릿 노출 안 됨")


def test_invalid_ord_type_combinations():
    print("\n[테스트 14] 잘못된 ord_type 조합 거부")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
    
    # 시장가 매수에 side='ask' 잘못 줌
    try:
        client.place_order(
            'KRW-BTC',
            side=OrderSide.ASK,  # 잘못된 side
            ord_type=OrderType.MARKET_BUY,
            price=Decimal('10000'),
        )
        raise AssertionError("잘못된 조합이 통과됨!")
    except ValueError as e:
        print(f"  ✓ 시장가 매수 + side=ask 거부: {str(e)[:60]}")
    
    # 시장가 매도에 price 줌
    try:
        client.place_order(
            'KRW-BTC',
            side=OrderSide.ASK,
            ord_type=OrderType.MARKET_SELL,
            volume=None,  # 누락
        )
        raise AssertionError("volume 누락이 통과됨!")
    except ValueError as e:
        print(f"  ✓ 시장가 매도 + volume 누락 거부: {str(e)[:60]}")
    
    # 지정가에 price 누락
    try:
        client.place_order(
            'KRW-BTC',
            side=OrderSide.BID,
            ord_type=OrderType.LIMIT,
            volume=Decimal('0.001'),
            price=None,  # 누락
        )
        raise AssertionError("price 누락이 통과됨!")
    except ValueError as e:
        print(f"  ✓ 지정가 + price 누락 거부: {str(e)[:60]}")


def test_from_config_controls_dry_run():
    print("\n[테스트 15] Config 거래 모드에 따른 dry-run 자동 설정")
    print("-" * 60)
    
    dev_client = BithumbClient.from_config(make_config(TradingMode.DEVELOPMENT))
    paper_client = BithumbClient.from_config(make_config(TradingMode.PAPER))
    live_client = BithumbClient.from_config(make_config(TradingMode.LIVE))
    
    assert dev_client.dry_run is True
    assert dev_client.allow_live_orders is False
    assert paper_client.dry_run is True
    assert paper_client.allow_live_orders is False
    assert live_client.dry_run is False
    assert live_client.allow_live_orders is True
    
    print("  ✓ DEVELOPMENT: dry-run + 실주문 차단")
    print("  ✓ PAPER: dry-run + 실주문 차단")
    print("  ✓ LIVE: dry-run 해제 + 실주문 허용")


def test_live_order_requires_explicit_enable():
    print("\n[테스트 16] 실주문 이중 안전장치")
    print("-" * 60)
    
    client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY, dry_run=False)
    
    with patch.object(client._session, 'request') as mock_req:
        try:
            client.market_buy('KRW-BTC', krw_amount=Decimal('10000'))
            raise AssertionError("allow_live_orders 없이 실주문이 통과됨!")
        except BithumbError as e:
            assert "실주문 차단" in str(e)
    
    assert mock_req.call_count == 0, "차단된 주문이 네트워크로 전송됨!"
    
    print("  ✓ dry_run=False여도 allow_live_orders 없으면 차단")
    print("  ✓ 차단된 주문은 네트워크 호출 0번")


# ============================================================
# 메인 실행
# ============================================================

def main():
    print("=" * 60)
    print("빗썸 클라이언트 검증 테스트 (Mock 기반)")
    print("=" * 60)
    
    tests = [
        test_client_initialization,
        test_ticker_parsing,
        test_candle_parsing,
        test_orderbook_parsing,
        test_balances_parsing,
        test_market_buy_amount_based,
        test_market_sell_volume_based,
        test_limit_order,
        test_decimal_enforcement,
        test_dry_run_mode,
        test_retry_on_network_error,
        test_no_retry_on_permanent_error,
        test_secret_not_exposed,
        test_invalid_ord_type_combinations,
        test_from_config_controls_dry_run,
        test_live_order_requires_explicit_enable,
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


if __name__ == '__main__':
    main()
