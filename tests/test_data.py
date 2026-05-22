"""
데이터 캐시 및 수집기 검증 테스트

11가지 시나리오:
1. 빈 DataFrame 저장 거부
2. 필수 컬럼 누락 검출
3. NaN 값 검출
4. high < low 검출
5. 저장-로드 round-trip 정확성
6. 증분 업데이트 (병합)
7. 중복 인덱스 제거
8. DatetimeIndex 자동 처리
9. 캐시 목록 조회
10. 수집기 페이지네이션 (Mock)
11. 빈 응답 처리
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from data.cache import OHLCVCache, CacheError
from data.collector import OHLCVCollector
from execution.bithumb_client import BithumbClient
from execution.models import Candle


# 테스트용 더미 키
TEST_API_KEY = "test_api_key_abcdef1234567890ABCDEF"
TEST_SECRET_KEY = "test_secret_key_xyz9876543210ZYXWVU"


def make_test_ohlcv(start_date: str = '2024-01-01', n_days: int = 100) -> pd.DataFrame:
    """테스트용 합성 OHLCV 생성"""
    dates = pd.date_range(start=start_date, periods=n_days, freq='D', tz='UTC')
    
    # 단순 랜덤 워크
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_days)))
    
    df = pd.DataFrame({
        'open': close * (1 + np.random.normal(0, 0.001, n_days)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'close': close,
        'volume': np.random.uniform(100, 1000, n_days),
    }, index=dates)
    
    # high >= max(open, close), low <= min(open, close) 보장
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


# ============================================================
# 캐시 모듈 테스트
# ============================================================

def test_empty_dataframe_rejected():
    print("\n[테스트 1] 빈 DataFrame 저장 거부")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        empty_df = pd.DataFrame()
        
        try:
            cache.save('TEST', '1d', empty_df)
            raise AssertionError("빈 DataFrame이 통과됨!")
        except CacheError as e:
            print(f"  ✓ 빈 DataFrame 거부: {str(e)[:60]}")


def test_missing_columns():
    print("\n[테스트 2] 필수 컬럼 누락 검출")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        
        # volume 컬럼 누락
        df = make_test_ohlcv(n_days=10)
        df = df.drop(columns=['volume'])
        
        try:
            cache.save('TEST', '1d', df)
            raise AssertionError("volume 누락 통과됨!")
        except CacheError as e:
            print(f"  ✓ 컬럼 누락 검출: {str(e)[:60]}")


def test_nan_detection():
    print("\n[테스트 3] NaN 값 검출")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        
        df = make_test_ohlcv(n_days=10)
        df.iloc[5, 0] = np.nan  # open에 NaN 주입
        
        try:
            cache.save('TEST', '1d', df)
            raise AssertionError("NaN이 통과됨!")
        except CacheError as e:
            print(f"  ✓ NaN 검출: {str(e)[:60]}")


def test_high_low_consistency():
    print("\n[테스트 4] high < low 검출")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        
        df = make_test_ohlcv(n_days=10)
        # 한 봉의 high를 low보다 작게 만듦
        df.iloc[5, df.columns.get_loc('high')] = df.iloc[5, df.columns.get_loc('low')] - 1
        
        try:
            cache.save('TEST', '1d', df)
            raise AssertionError("high < low 통과됨!")
        except CacheError as e:
            print(f"  ✓ 가격 무결성 검출: {str(e)[:60]}")


def test_save_load_roundtrip():
    print("\n[테스트 5] 저장-로드 round-trip 정확성")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        
        original = make_test_ohlcv(n_days=100)
        cache.save('KRW-BTC', '1d', original, merge_with_existing=False)
        loaded = cache.load('KRW-BTC', '1d')
        
        # 길이 확인
        assert len(loaded) == len(original)
        
        # 값 일치 확인 (CSV round-trip이라 float 약간 오차 가능)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            max_diff = (loaded[col] - original[col]).abs().max()
            assert max_diff < 1e-6, f"{col} 값 불일치: max_diff={max_diff}"
        
        print(f"  ✓ {len(loaded)}봉 저장-로드 정확히 일치")
        print(f"  ✓ 최대 오차: {max_diff:.2e}")


def test_incremental_merge():
    print("\n[테스트 6] 증분 업데이트 (병합)")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        
        # 첫 번째: 1월 1일 ~ 1월 50일
        df1 = make_test_ohlcv(start_date='2024-01-01', n_days=50)
        cache.save('TEST', '1d', df1, merge_with_existing=False)
        
        # 두 번째: 1월 30일 ~ 2월 28일 (10일 겹침)
        df2 = make_test_ohlcv(start_date='2024-01-30', n_days=30)
        cache.save('TEST', '1d', df2, merge_with_existing=True)
        
        combined = cache.load('TEST', '1d')
        
        # 겹친 부분 없이 1월 1일 ~ 2월 28일 = 59일이어야 함
        # (1.1 ~ 1.29: 29일) + (1.30 ~ 2.28: 30일) = 59일
        expected_days = 59
        
        print(f"  첫 저장: {len(df1)}봉 (1/1 ~ 1/{len(df1)})")
        print(f"  두번째 저장: {len(df2)}봉 (1/30 ~ ...)")
        print(f"  병합 결과: {len(combined)}봉")
        
        assert len(combined) == expected_days, \
            f"병합 결과 오류: 기대 {expected_days}, 실제 {len(combined)}"
        
        # 인덱스 중복 없는지
        assert not combined.index.duplicated().any(), "중복 인덱스 존재!"
        
        # 시간 정렬
        assert combined.index.is_monotonic_increasing, "정렬 안 됨!"
        
        print(f"  ✓ 중복 없이 병합됨")
        print(f"  ✓ 시간순 정렬됨")


def test_get_last_timestamp():
    print("\n[테스트 7] 마지막 봉 시각 조회")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        
        df = make_test_ohlcv(start_date='2024-01-01', n_days=100)
        cache.save('TEST', '1d', df, merge_with_existing=False)
        
        last = cache.get_last_timestamp('TEST', '1d')
        expected = df.index[-1].to_pydatetime()
        
        # 시간대 통일해서 비교
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if expected.tzinfo is None:
            expected = expected.replace(tzinfo=timezone.utc)
        
        diff = abs((last - expected).total_seconds())
        assert diff < 1, f"시각 불일치: {last} vs {expected}"
        
        print(f"  ✓ 마지막 봉: {last}")
        
        # 없는 캐시
        none_result = cache.get_last_timestamp('NONEXISTENT', '1d')
        assert none_result is None
        print(f"  ✓ 없는 캐시는 None 반환")


def test_cache_listing():
    print("\n[테스트 8] 캐시 목록 조회")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        
        # 여러 종목 저장
        cache.save('KRW-BTC', '1d', make_test_ohlcv(n_days=50))
        cache.save('KRW-ETH', '1d', make_test_ohlcv(n_days=80))
        cache.save('KRW-BTC', '4h', make_test_ohlcv(n_days=200))
        
        cached = cache.list_cached()
        assert len(cached) == 3, f"3개여야 함: {len(cached)}"
        
        for entry in cached:
            print(f"  ✓ {entry['market']} {entry['timeframe']}: "
                  f"{entry['rows']}봉")


# ============================================================
# 수집기 모듈 테스트 (Mock 기반)
# ============================================================

def test_collector_pagination():
    print("\n[테스트 9] 수집기 페이지네이션 (Mock)")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
        collector = OHLCVCollector(client, cache, request_interval_sec=0.01)
        
        # Mock get_candles: 800봉 데이터 (최신부터 800일 전까지)
        # 최신 → 오래된 순으로 정렬 (빗썸 API 응답 방식)
        now = datetime.now(timezone.utc)
        all_candles = []
        for i in range(800):
            ts = now - timedelta(days=i)
            all_candles.append(Candle(
                market='KRW-BTC',
                candle_datetime_utc=ts,
                candle_datetime_kst=ts,
                opening_price=Decimal('100'),
                high_price=Decimal('105'),
                low_price=Decimal('95'),
                trade_price=Decimal('102'),
                candle_acc_trade_volume=Decimal('1000'),
                candle_acc_trade_price=Decimal('100000'),
                timestamp=int(ts.timestamp() * 1000),
            ))
        
        # 페이지네이션 시뮬레이션 (빗썸: 한 번에 최대 200봉, 최신순 반환)
        call_count = [0]
        def mock_get_candles(market, unit, count, to=None):
            call_count[0] += 1
            
            # all_candles는 최신 → 오래된 순으로 정렬되어 있음
            # 'to' 파라미터: 이 시각 이전(직전 포함 X)의 봉들을 최신순으로
            if to is None:
                # to 없으면 가장 최신부터 count개
                idx_start = 0
            else:
                # 'to' 형식: 'YYYY-MM-DD HH:MM:SS' (UTC, timezone 없음)
                # naive datetime을 만든 후 UTC timezone 부여
                to_dt = datetime.strptime(to, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                idx_start = next(
                    (i for i, c in enumerate(all_candles)
                     if c.candle_datetime_utc < to_dt),
                    len(all_candles)
                )
            
            # 빗썸 한계: 한 번에 최대 200봉
            actual_count = min(count, 200)
            return all_candles[idx_start:idx_start + actual_count]
        
        with patch.object(client, 'get_candles', side_effect=mock_get_candles):
            df = collector.collect_historical(
                market='KRW-BTC',
                timeframe='1d',
                years=1.5,  # 약 547일
                use_cache=False,
                verbose=False,
            )
        
        print(f"  ✓ {len(df)}봉 수집됨")
        print(f"  ✓ API 호출 횟수: {call_count[0]}회")
        # 1.5년치 = 약 547봉, 200봉씩 → 최소 3회 호출
        assert len(df) >= 500, f"1.5년치(약 547봉)는 받아야 함: {len(df)}"
        assert call_count[0] >= 3, f"페이지네이션 3회 이상 발생해야 함: {call_count[0]}"


def test_empty_response_handling():
    print("\n[테스트 10] 빈 응답 처리")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = OHLCVCache(cache_dir=Path(tmpdir))
        client = BithumbClient(TEST_API_KEY, TEST_SECRET_KEY)
        collector = OHLCVCollector(client, cache, request_interval_sec=0.01)
        
        # Mock: 항상 빈 리스트 반환
        with patch.object(client, 'get_candles', return_value=[]):
            df = collector.collect_historical(
                market='KRW-NONEXISTENT',
                timeframe='1d',
                years=1.0,
                use_cache=False,
                verbose=False,
            )
        
        assert df.empty
        print(f"  ✓ 빈 응답 시 빈 DataFrame 반환")


def test_candles_to_dataframe():
    print("\n[테스트 11] Candle → DataFrame 변환")
    print("-" * 60)
    
    candles = [
        Candle(
            market='KRW-BTC',
            candle_datetime_utc=datetime(2024, 1, 2, tzinfo=timezone.utc),
            candle_datetime_kst=datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
            opening_price=Decimal('100'),
            high_price=Decimal('110'),
            low_price=Decimal('95'),
            trade_price=Decimal('105'),
            candle_acc_trade_volume=Decimal('1000'),
            candle_acc_trade_price=Decimal('105000'),
            timestamp=1704153600000,
        ),
        Candle(
            market='KRW-BTC',
            candle_datetime_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
            candle_datetime_kst=datetime(2024, 1, 1, 9, tzinfo=timezone.utc),
            opening_price=Decimal('98'),
            high_price=Decimal('102'),
            low_price=Decimal('97'),
            trade_price=Decimal('100'),
            candle_acc_trade_volume=Decimal('800'),
            candle_acc_trade_price=Decimal('80000'),
            timestamp=1704067200000,
        ),
    ]
    
    df = OHLCVCollector._candles_to_dataframe(candles)
    
    # 시간 오름차순 정렬 확인
    assert df.index.is_monotonic_increasing, "시간순 정렬 안 됨!"
    
    # 값 확인 (오래된 것부터)
    assert df.iloc[0]['close'] == 100.0
    assert df.iloc[1]['close'] == 105.0
    
    print(f"  ✓ {len(df)}봉 변환")
    print(f"  ✓ 시간순 정렬됨")
    print(f"  ✓ Decimal → float 변환됨")


def main():
    print("=" * 60)
    print("데이터 캐시 & 수집기 검증 테스트")
    print("=" * 60)
    
    tests = [
        test_empty_dataframe_rejected,
        test_missing_columns,
        test_nan_detection,
        test_high_low_consistency,
        test_save_load_roundtrip,
        test_incremental_merge,
        test_get_last_timestamp,
        test_cache_listing,
        test_collector_pagination,
        test_empty_response_handling,
        test_candles_to_dataframe,
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
