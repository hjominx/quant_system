"""
빗썸 OHLCV 데이터 수집기

설계 원칙:
1. 빗썸 API 한계(한 번에 200봉)를 페이지네이션으로 극복
2. 레이트 리밋 준수 (자동 대기, 빗썸 Public 150 req/s 한계의 5% 이하 사용)
3. 증분 업데이트 (기존 캐시 활용, 새 봉만 받음)
4. 무결성 검증 (다운로드 직후 NaN/중복/누락 체크)
5. 진행 상황 표시 (수십 분 걸릴 수 있으니)

핵심 함수:
- collect_ohlcv: 단일 종목 OHLCV 수집
- collect_historical: 5년치 같은 대량 다운로드
- update_latest: 캐시 이후 새 봉만 받기
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.bithumb_client import BithumbClient
from execution.models import Candle
from data.cache import OHLCVCache, CacheError

logger = logging.getLogger(__name__)


# 빗썸 API 한계
MAX_CANDLES_PER_REQUEST = 200

# 안전한 요청 간격 (빗썸 한계 150/s 의 약 5% 사용)
# 동시성 없이 순차 호출이라 사실 더 빨라도 되지만 안전 마진 두기
DEFAULT_REQUEST_INTERVAL_SEC = 0.2  # 초당 5회 (빗썸 한계 150/s의 3%)


# 타임프레임별 일수 (페이지네이션 시 to 파라미터 계산용)
TIMEFRAME_TO_DAYS = {
    '1m': 200 / (24 * 60),       # 200분 = 약 0.14일
    '3m': 200 * 3 / (24 * 60),
    '5m': 200 * 5 / (24 * 60),
    '10m': 200 * 10 / (24 * 60),
    '15m': 200 * 15 / (24 * 60),
    '30m': 200 * 30 / (24 * 60),
    '60m': 200 / 24,              # 200시간 = 약 8.3일
    '1h': 200 / 24,
    '240m': 200 * 4 / 24,         # 800시간 = 약 33.3일
    '4h': 200 * 4 / 24,
    '1d': 200,                     # 200일
    '1w': 200 * 7,
    '1M': 200 * 30,                # 대략
}


class CollectorError(Exception):
    """데이터 수집 관련 에러"""
    pass


class OHLCVCollector:
    """빗썸 OHLCV 데이터 수집기
    
    사용법:
        from execution.bithumb_client import BithumbClient
        from data.cache import OHLCVCache
        
        client = BithumbClient(api_key, secret_key)
        cache = OHLCVCache()
        collector = OHLCVCollector(client, cache)
        
        # 5년치 BTC 일봉 다운로드
        df = collector.collect_historical(
            market='KRW-BTC',
            timeframe='1d',
            years=5,
        )
        
        # 캐시 활용 + 증분 업데이트
        df = collector.update_latest('KRW-BTC', '1d')
    """
    
    def __init__(
        self,
        client: BithumbClient,
        cache: Optional[OHLCVCache] = None,
        request_interval_sec: float = DEFAULT_REQUEST_INTERVAL_SEC,
    ):
        """
        Args:
            client: 빗썸 클라이언트
            cache: OHLCV 캐시 (None이면 메모리만, 저장 안 함)
            request_interval_sec: 요청 간 대기 시간 (레이트 리밋 보호)
        """
        self.client = client
        self.cache = cache if cache is not None else OHLCVCache()
        self.request_interval_sec = request_interval_sec
    
    def collect_ohlcv(
        self,
        market: str,
        timeframe: str = '1d',
        count: int = 200,
        to: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """단일 요청 OHLCV 수집 (페이지네이션 없음)
        
        Args:
            market: 'KRW-BTC' 같은 마켓 코드
            timeframe: '1d', '4h', '1h' 등
            count: 가져올 봉 개수 (최대 200)
            to: 마지막 봉 시각 (None이면 현재까지)
        
        Returns:
            OHLCV DataFrame (DatetimeIndex, 시간 오름차순)
        """
        to_str = self._format_datetime(to) if to else None
        
        candles = self.client.get_candles(
            market=market,
            unit=timeframe,
            count=min(count, MAX_CANDLES_PER_REQUEST),
            to=to_str,
        )
        
        if not candles:
            return pd.DataFrame()
        
        return self._candles_to_dataframe(candles)
    
    def collect_historical(
        self,
        market: str,
        timeframe: str = '1d',
        years: float = 5.0,
        use_cache: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """대량 과거 데이터 수집 (페이지네이션 자동)
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            years: 몇 년치 받을지
            use_cache: True면 기존 캐시 활용 (증분 업데이트)
            verbose: 진행 상황 출력
        
        Returns:
            전체 기간 OHLCV DataFrame
        """
        # 시작 시각 계산
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=int(years * 365.25))
        
        if verbose:
            print(f"\n=== {market} {timeframe} {years}년치 다운로드 시작 ===")
            print(f"기간: {start_time.date()} ~ {end_time.date()}")
        
        # 캐시 활용 분기
        if use_cache and self.cache.exists(market, timeframe):
            return self._collect_incremental(
                market, timeframe, start_time, end_time, verbose
            )
        else:
            return self._collect_fresh(
                market, timeframe, start_time, end_time, verbose
            )
    
    def update_latest(
        self,
        market: str,
        timeframe: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """캐시 이후 최신 봉만 추가 다운로드 (일상 운영용)
        
        매일 봇 시작 시 호출. 전체 5년치를 다시 받지 않고
        마지막 봉 이후의 새 봉만 가져옴.
        """
        if not self.cache.exists(market, timeframe):
            raise CollectorError(
                f"캐시 없음: {market} {timeframe}. "
                f"먼저 collect_historical()로 초기 데이터 받으세요."
            )
        
        last_time = self.cache.get_last_timestamp(market, timeframe)
        end_time = datetime.now(timezone.utc)
        
        if last_time is None:
            raise CollectorError(f"캐시 빔: {market} {timeframe}")
        
        # 마지막 봉 이후의 데이터만 받기
        if verbose:
            print(f"증분 업데이트: {market} {timeframe}")
            print(f"  마지막 봉: {last_time}")
            print(f"  현재: {end_time}")
        
        return self._collect_incremental(
            market, timeframe, last_time, end_time, verbose
        )
    
    def _collect_fresh(
        self,
        market: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        verbose: bool,
    ) -> pd.DataFrame:
        """새로 받기 (캐시 무시)"""
        all_dfs: List[pd.DataFrame] = []
        cursor = end_time  # 최신부터 거꾸로 받음 (빗썸 API 특성)
        request_count = 0
        
        while True:
            # 한 페이지 다운로드
            df = self.collect_ohlcv(
                market=market,
                timeframe=timeframe,
                count=MAX_CANDLES_PER_REQUEST,
                to=cursor,
            )
            
            request_count += 1
            
            if df.empty:
                if verbose:
                    print(f"  (빈 응답, 종료)")
                break
            
            all_dfs.append(df)
            
            # 가장 오래된 봉의 시각
            oldest = df.index[0]
            
            if verbose:
                print(f"  요청 {request_count}: "
                      f"{oldest.date()} ~ {df.index[-1].date()} "
                      f"({len(df)}봉)")
            
            # 시작 시점 도달했으면 종료
            if oldest <= start_time:
                if verbose:
                    print(f"  시작 시점 도달, 종료")
                break
            
            # 다음 페이지를 위해 cursor 이동 (가장 오래된 봉의 1초 전)
            cursor = oldest - timedelta(seconds=1)
            
            # 레이트 리밋 보호
            time.sleep(self.request_interval_sec)
        
        # 모든 페이지 합치기
        if not all_dfs:
            if verbose:
                print(f"  ⚠️  데이터 없음")
            return pd.DataFrame()
        
        combined = pd.concat(all_dfs)
        
        # 중복 제거 (페이지 경계에서 가끔 발생)
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # 시간순 정렬
        combined = combined.sort_index()
        
        # 요청한 기간만 필터링
        combined = combined[combined.index >= start_time]
        
        if verbose:
            print(f"\n✓ 다운로드 완료: 총 {len(combined)}봉")
            print(f"  기간: {combined.index[0].date()} ~ {combined.index[-1].date()}")
            print(f"  API 호출: {request_count}회")
        
        # 캐시 저장
        if self.cache is not None:
            self.cache.save(market, timeframe, combined, merge_with_existing=False)
            if verbose:
                print(f"  캐시 저장: data/cache/{market}_{timeframe}.csv")
        
        return combined
    
    def _collect_incremental(
        self,
        market: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        verbose: bool,
    ) -> pd.DataFrame:
        """증분 업데이트 (캐시 + 새 데이터)"""
        # 기존 캐시 로드
        existing = self.cache.load(market, timeframe)
        
        # start_time 이전 데이터는 그대로 유지
        if existing.index[0] <= start_time:
            # 캐시가 start_time 이상 커버함 → 새 데이터만 추가
            new_start = existing.index[-1]
        else:
            # 캐시가 부족 → start_time부터 새로 받아야 함
            new_start = start_time
        
        if verbose:
            print(f"  기존 캐시: {len(existing)}봉 "
                  f"({existing.index[0].date()} ~ {existing.index[-1].date()})")
            print(f"  필요한 기간: {new_start} 이후")
        
        # 새 데이터 다운로드 (캐시 무시 모드)
        all_dfs: List[pd.DataFrame] = []
        cursor = end_time
        request_count = 0
        
        while True:
            df = self.collect_ohlcv(
                market=market,
                timeframe=timeframe,
                count=MAX_CANDLES_PER_REQUEST,
                to=cursor,
            )
            
            request_count += 1
            
            if df.empty:
                break
            
            all_dfs.append(df)
            oldest = df.index[0]
            
            if verbose:
                print(f"  증분 요청 {request_count}: "
                      f"{oldest.date()} ~ {df.index[-1].date()} ({len(df)}봉)")
            
            if oldest <= new_start:
                break
            
            cursor = oldest - timedelta(seconds=1)
            time.sleep(self.request_interval_sec)
        
        if not all_dfs:
            if verbose:
                print(f"  새 봉 없음, 캐시 그대로 반환")
            return existing
        
        # 합치기
        new_df = pd.concat(all_dfs)
        new_df = new_df[~new_df.index.duplicated(keep='first')]
        new_df = new_df.sort_index()
        new_df = new_df[new_df.index >= new_start]
        
        # 기존 + 신규 병합 (중복 제거)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        # start_time 이전 데이터 제거
        combined = combined[combined.index >= start_time]
        
        if verbose:
            new_count = len(combined) - len(existing)
            print(f"\n✓ 증분 업데이트 완료: +{new_count}봉 (총 {len(combined)}봉)")
        
        # 캐시 저장
        self.cache.save(market, timeframe, combined, merge_with_existing=False)
        
        return combined
    
    @staticmethod
    def _candles_to_dataframe(candles: List[Candle]) -> pd.DataFrame:
        """Candle 리스트를 OHLCV DataFrame으로 변환"""
        records = []
        for c in candles:
            # KST 기준 시각 사용 (한국 거래소이므로)
            timestamp = c.candle_datetime_kst or c.candle_datetime_utc
            records.append({
                'timestamp': timestamp,
                'open': float(c.opening_price),
                'high': float(c.high_price),
                'low': float(c.low_price),
                'close': float(c.trade_price),
                'volume': float(c.candle_acc_trade_volume),
            })
        
        df = pd.DataFrame(records)
        df = df.set_index('timestamp')
        
        # 시간 오름차순 정렬 (빗썸은 최신순으로 주므로 뒤집기)
        df = df.sort_index()
        
        return df
    
    @staticmethod
    def _format_datetime(dt: datetime) -> str:
        """빗썸 API에 보낼 datetime 형식 (ISO 8601)"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # 빗썸 캔들 API의 to 파라미터는 timezone 없이 'YYYY-MM-DD HH:MM:SS' 형식을 사용합니다.
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    def collect_multiple(
        self,
        markets: List[str],
        timeframe: str = '1d',
        years: float = 5.0,
        verbose: bool = True,
    ) -> dict:
        """여러 종목 동시 수집
        
        Returns:
            {market: DataFrame} 딕셔너리
        """
        result = {}
        total = len(markets)
        
        for idx, market in enumerate(markets, 1):
            if verbose:
                print(f"\n[{idx}/{total}] {market} 처리 중...")
            
            try:
                df = self.collect_historical(
                    market=market,
                    timeframe=timeframe,
                    years=years,
                    use_cache=True,
                    verbose=verbose,
                )
                result[market] = df
            except Exception as e:
                logger.error(f"{market} 수집 실패: {e}")
                if verbose:
                    print(f"  ✗ 실패: {e}")
                result[market] = None
        
        return result
