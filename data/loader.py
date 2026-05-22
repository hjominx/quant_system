"""
백테스트용 데이터 로더

OHLCVCache에서 데이터를 가져와 백테스트 엔진에 맞는 형식으로 변환.

설계 원칙:
1. 캐시는 그대로 두고 읽기만 함 (불변성)
2. 백테스트 엔진 요구사항에 맞게 검증 (DatetimeIndex, 필수 컬럼 등)
3. 기간 필터링 지원 (Walk-Forward에서 사용)
4. timezone-naive로 변환 (백테스트 엔진이 그렇게 동작)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache import OHLCVCache


def load_for_backtest(
    market: str,
    timeframe: str = '1d',
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    cache: Optional[OHLCVCache] = None,
) -> pd.DataFrame:
    """캐시에서 백테스트용 OHLCV 로드
    
    Args:
        market: 'KRW-BTC' 같은 마켓 코드
        timeframe: '1d', '4h' 등
        start: 시작 시각 (None이면 전체)
        end: 끝 시각 (None이면 전체)
        cache: 캐시 인스턴스 (None이면 기본 위치 사용)
    
    Returns:
        백테스트 엔진에 바로 넣을 수 있는 DataFrame
        - DatetimeIndex (timezone-naive, KST 기준)
        - columns: open, high, low, close, volume
    """
    if cache is None:
        cache = OHLCVCache()
    
    if not cache.exists(market, timeframe):
        raise FileNotFoundError(
            f"캐시 없음: {market} {timeframe}. "
            f"먼저 scripts/download_historical.py 실행하세요."
        )
    
    df = cache.load(market, timeframe)
    
    # 기간 필터링 (있으면)
    if start is not None:
        if start.tzinfo is None and df.index.tz is not None:
            start = pd.Timestamp(start).tz_localize(df.index.tz)
        df = df[df.index >= start]
    
    if end is not None:
        if end.tzinfo is None and df.index.tz is not None:
            end = pd.Timestamp(end).tz_localize(df.index.tz)
        df = df[df.index <= end]
    
    # 백테스트 엔진 호환을 위해 timezone 제거 (naive datetime)
    # 시각은 그대로, timezone 정보만 떼어냄 (KST 시간이 KST 그대로 유지됨)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # 최종 검증
    if df.empty:
        raise ValueError(
            f"필터링 후 데이터 없음: {market} {timeframe} "
            f"(start={start}, end={end})"
        )
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    
    return df


def get_cache_summary() -> pd.DataFrame:
    """현재 캐시 상태 한눈에 보기"""
    cache = OHLCVCache()
    cached = cache.list_cached()
    
    if not cached:
        return pd.DataFrame()
    
    records = []
    for entry in cached:
        if entry['exists']:
            records.append({
                'market': entry['market'],
                'timeframe': entry['timeframe'],
                'rows': entry['rows'],
                'start': entry['start'][:10],  # YYYY-MM-DD만
                'end': entry['end'][:10],
            })
    
    return pd.DataFrame(records).sort_values(['market', 'timeframe'])
