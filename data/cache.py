"""
데이터 캐시 모듈

설계 원칙:
1. OHLCV는 CSV로 저장 (가독성 + 의존성 0)
2. 파일명에 종목·타임프레임 명시 (예: KRW-BTC_1d.csv)
3. DatetimeIndex 기반 (백테스트 엔진과 호환)
4. 증분 업데이트 지원 (기존 + 새 데이터 병합)
5. 무결성 검증 후 저장 (NaN, 중복, 정렬 확인)

저장 위치: data/cache/
파일 형식: {market}_{timeframe}.csv
예시: data/cache/KRW-BTC_1d.csv
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


# OHLCV에 필수로 있어야 하는 컬럼들
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']


class CacheError(Exception):
    """캐시 관련 에러"""
    pass


class OHLCVCache:
    """OHLCV 데이터 로컬 캐시
    
    사용법:
        cache = OHLCVCache()
        
        # 데이터 저장
        cache.save('KRW-BTC', '1d', df)
        
        # 로드
        df = cache.load('KRW-BTC', '1d')
        
        # 마지막 봉 시각 확인 (증분 업데이트 시 사용)
        last_time = cache.get_last_timestamp('KRW-BTC', '1d')
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: 캐시 저장 디렉토리 (기본: 프로젝트루트/data/cache)
        """
        if cache_dir is None:
            # 프로젝트 루트의 data/cache
            cache_dir = Path(__file__).parent.parent / "data" / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"캐시 디렉토리: {self.cache_dir}")
    
    def _get_filepath(self, market: str, timeframe: str) -> Path:
        """캐시 파일 경로 생성
        
        Args:
            market: 'KRW-BTC' 같은 마켓 코드
            timeframe: '1d', '4h', '1h' 등
        """
        # 파일명 안전화 (특수문자 제거)
        safe_market = market.replace('/', '_')
        return self.cache_dir / f"{safe_market}_{timeframe}.csv"
    
    def exists(self, market: str, timeframe: str) -> bool:
        """캐시 파일 존재 여부"""
        return self._get_filepath(market, timeframe).exists()
    
    def save(
        self,
        market: str,
        timeframe: str,
        df: pd.DataFrame,
        merge_with_existing: bool = True,
    ) -> int:
        """OHLCV 데이터 저장
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            df: OHLCV DataFrame (DatetimeIndex, columns: open/high/low/close/volume)
            merge_with_existing: True면 기존 데이터와 병합 (증분 업데이트)
        
        Returns:
            저장된 총 행 수
        """
        # 입력 검증
        self._validate_dataframe(df)
        
        filepath = self._get_filepath(market, timeframe)
        
        # 기존 데이터와 병합 (증분 업데이트)
        if merge_with_existing and filepath.exists():
            existing = self.load(market, timeframe)
            combined = self._merge_dataframes(existing, df)
            logger.info(
                f"증분 업데이트 {market} {timeframe}: "
                f"기존 {len(existing)}봉 + 신규 {len(df)}봉 = 총 {len(combined)}봉"
            )
            df_to_save = combined
        else:
            df_to_save = df
        
        # CSV 저장 (인덱스 포함)
        df_to_save.to_csv(filepath, index=True, date_format='%Y-%m-%dT%H:%M:%S%z')
        logger.debug(f"저장 완료: {filepath} ({len(df_to_save)} rows)")
        
        return len(df_to_save)
    
    def load(self, market: str, timeframe: str) -> pd.DataFrame:
        """OHLCV 데이터 로드
        
        Returns:
            OHLCV DataFrame (DatetimeIndex, 시간 오름차순 정렬)
        
        Raises:
            CacheError: 캐시 파일이 없거나 형식이 깨졌을 때
        """
        filepath = self._get_filepath(market, timeframe)
        
        if not filepath.exists():
            raise CacheError(f"캐시 없음: {market} {timeframe} ({filepath})")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # 인덱스가 timezone-aware가 아니면 UTC로 가정
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # 무결성 재검증
        self._validate_dataframe(df)
        
        # 시간 오름차순 정렬 보장
        df = df.sort_index()
        
        return df
    
    def get_last_timestamp(
        self,
        market: str,
        timeframe: str,
    ) -> Optional[datetime]:
        """캐시된 데이터의 마지막 봉 시각 반환
        
        증분 업데이트 시 "여기서부터 이어받자" 라는 기준점.
        
        Returns:
            마지막 봉 시각 (없으면 None)
        """
        if not self.exists(market, timeframe):
            return None
        
        df = self.load(market, timeframe)
        if len(df) == 0:
            return None
        
        return df.index[-1].to_pydatetime()
    
    def get_summary(self, market: str, timeframe: str) -> dict:
        """캐시된 데이터 요약 정보"""
        if not self.exists(market, timeframe):
            return {'exists': False, 'market': market, 'timeframe': timeframe}
        
        df = self.load(market, timeframe)
        return {
            'exists': True,
            'market': market,
            'timeframe': timeframe,
            'rows': len(df),
            'start': df.index[0].isoformat(),
            'end': df.index[-1].isoformat(),
            'filepath': str(self._get_filepath(market, timeframe)),
        }
    
    def list_cached(self) -> List[dict]:
        """캐시된 모든 데이터 목록"""
        result = []
        for filepath in self.cache_dir.glob('*.csv'):
            # 파일명에서 market_timeframe 분리
            stem = filepath.stem  # 'KRW-BTC_1d'
            parts = stem.rsplit('_', 1)  # ['KRW-BTC', '1d']
            if len(parts) != 2:
                continue
            
            market, timeframe = parts
            try:
                result.append(self.get_summary(market, timeframe))
            except Exception as e:
                logger.warning(f"캐시 파일 읽기 실패: {filepath} - {e}")
        
        return result
    
    def delete(self, market: str, timeframe: str) -> bool:
        """특정 캐시 삭제"""
        filepath = self._get_filepath(market, timeframe)
        if filepath.exists():
            filepath.unlink()
            logger.info(f"캐시 삭제: {filepath}")
            return True
        return False
    
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """DataFrame 무결성 검증"""
        if df.empty:
            raise CacheError("빈 DataFrame은 저장할 수 없습니다")
        
        # 필수 컬럼 확인
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise CacheError(f"필수 컬럼 누락: {missing}")
        
        # DatetimeIndex 확인
        if not isinstance(df.index, pd.DatetimeIndex):
            raise CacheError(
                f"인덱스가 DatetimeIndex여야 함: {type(df.index).__name__}"
            )
        
        # NaN 확인
        if df[REQUIRED_COLUMNS].isnull().any().any():
            null_cols = df[REQUIRED_COLUMNS].isnull().sum()
            null_info = {k: v for k, v in null_cols.items() if v > 0}
            raise CacheError(f"NaN 값 존재: {null_info}")
        
        # 가격 무결성 (high >= low)
        if not (df['high'] >= df['low']).all():
            bad_rows = df[df['high'] < df['low']]
            raise CacheError(f"high < low 인 봉 존재: {len(bad_rows)}개")
        
        # 가격이 양수인지
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            raise CacheError("0 이하의 가격 존재")
    
    @staticmethod
    def _merge_dataframes(
        existing: pd.DataFrame,
        new: pd.DataFrame,
    ) -> pd.DataFrame:
        """기존 + 새 데이터 병합 (중복 제거, 정렬)
        
        중복 시 새 데이터가 우선 (빗썸이 봉을 수정했을 가능성).
        """
        combined = pd.concat([existing, new])
        
        # 중복 인덱스 제거 (마지막 값 유지 = 새 데이터)
        combined = combined[~combined.index.duplicated(keep='last')]
        
        # 정렬
        combined = combined.sort_index()
        
        return combined
