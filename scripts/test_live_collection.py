"""
실제 빗썸 데이터 수집 미니 테스트

5년치 받기 전에 단 한 종목으로 빠르게 동작 확인.
- BTC 30일치 일봉 다운로드
- 데이터 무결성 검증
- 캐시 저장 확인
- 약 10초 이내 완료

실행:
    python scripts/test_live_collection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import setup_logging
from execution.bithumb_client import BithumbClient
from data.cache import OHLCVCache
from data.collector import OHLCVCollector


def main():
    setup_logging(log_level='INFO')
    
    print("=" * 60)
    print("실데이터 수집 미니 테스트 (BTC 30일치)")
    print("=" * 60)
    
    # 설정
    cfg = get_config()
    print(f"\n거래 모드: {cfg.trading_mode.value}")
    
    # 클라이언트
    client = BithumbClient.from_config(cfg)
    
    # 헬스체크
    print("\n빗썸 서버 헬스체크...")
    if not client.health_check():
        print("✗ 빗썸 서버 연결 실패")
        sys.exit(1)
    print("✓ 정상")
    
    # 캐시 + 수집기 (낮은 요청 간격으로 빠르게)
    cache = OHLCVCache()
    collector = OHLCVCollector(client, cache, request_interval_sec=0.2)
    
    # BTC 30일치 다운로드 (단일 페이지, 빠름)
    print("\nBTC 30일치 일봉 다운로드...")
    df = collector.collect_ohlcv(
        market='KRW-BTC',
        timeframe='1d',
        count=30,
    )
    
    if df.empty:
        print("✗ 빈 데이터 받음")
        sys.exit(1)
    
    # 결과 출력
    print(f"\n✓ {len(df)}봉 수신")
    print(f"  기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    
    print("\n최근 5봉:")
    recent = df.tail(5)
    for idx, row in recent.iterrows():
        print(f"  {idx.strftime('%Y-%m-%d')}: "
              f"O={row['open']:>13,.0f} "
              f"H={row['high']:>13,.0f} "
              f"L={row['low']:>13,.0f} "
              f"C={row['close']:>13,.0f} "
              f"V={row['volume']:>10,.2f}")
    
    # 무결성 검증
    print("\n무결성 검증:")
    assert df['high'].ge(df['low']).all(), "high < low 봉 존재!"
    assert not df.isnull().any().any(), "NaN 값 존재!"
    print("  ✓ high >= low 모두 정상")
    print("  ✓ NaN 없음")
    
    # 캐시 저장
    print("\n캐시 저장 중...")
    cache.save('KRW-BTC', '1d', df, merge_with_existing=False)
    print(f"  저장 위치: {cache._get_filepath('KRW-BTC', '1d')}")
    
    # 캐시 로드 검증 (round-trip)
    print("\n캐시 로드 검증 (round-trip):")
    loaded = cache.load('KRW-BTC', '1d')
    
    # 값 일치 확인
    max_diff = (loaded[['open', 'high', 'low', 'close', 'volume']] - 
                df[['open', 'high', 'low', 'close', 'volume']]).abs().max().max()
    
    if max_diff < 1e-6:
        print(f"  ✓ 저장-로드 정확히 일치 (오차 {max_diff:.2e})")
    else:
        print(f"  ⚠️  오차 발생: {max_diff}")
    
    # 최종
    print("\n" + "=" * 60)
    print("✓ 미니 테스트 통과")
    print("=" * 60)
    print("\n다음 단계:")
    print("  전체 5년치 다운로드를 원하면:")
    print("  $ python scripts/download_historical.py")
    
    client.close()


if __name__ == '__main__':
    main()
