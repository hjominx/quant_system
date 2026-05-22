"""
5년치 OHLCV 데이터 일괄 다운로드 스크립트

실행 방법:
    python scripts/download_historical.py

설명:
- 추세추종 전략용: BTC/ETH 일봉
- 평균회귀 전략용: 주요 알트 4시간봉
- 캐시 디렉토리: data/cache/

소요 시간 예측:
- BTC, ETH 일봉 5년: 약 30초
- 알트 8종 4시간봉 5년: 약 5분
- 총 약 6분 (네트워크 상황에 따라 변동)
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import setup_logging
from execution.bithumb_client import BithumbClient
from data.cache import OHLCVCache
from data.collector import OHLCVCollector


# ============================================================
# 다운로드 대상 정의
# ============================================================

# 코어 전략 (BTC/ETH 일봉)
CORE_TARGETS = [
    ('KRW-BTC', '1d'),
    ('KRW-ETH', '1d'),
]

# 평균회귀 전략용 알트 바스켓 (4시간봉)
# 시총 상위 + 거래량 많은 종목 우선
ALT_TARGETS_4H = [
    ('KRW-XRP', '4h'),
    ('KRW-SOL', '4h'),
    ('KRW-ADA', '4h'),
    ('KRW-DOGE', '4h'),
    ('KRW-AVAX', '4h'),
    ('KRW-LINK', '4h'),
    ('KRW-DOT', '4h'),
    ('KRW-MATIC', '4h'),
]

# 알트도 일봉으로 (장기 추세 확인용)
ALT_TARGETS_1D = [
    (market, '1d') for market, _ in ALT_TARGETS_4H
]

ALL_TARGETS = CORE_TARGETS + ALT_TARGETS_4H + ALT_TARGETS_1D

# 빗썸에서 상장 폐지되거나 티커가 변경된 종목 매핑
MARKET_ALIASES = {
    'KRW-MATIC': 'KRW-POL',
}


def main():
    # 로깅 설정 (INFO 레벨)
    setup_logging(log_level='INFO')
    
    print("=" * 70)
    print("빗썸 5년치 OHLCV 데이터 일괄 다운로드")
    print("=" * 70)
    
    # 설정 로드 (API 키 등)
    try:
        cfg = get_config()
    except Exception as e:
        print(f"\n✗ 설정 로드 실패: {e}")
        print("  .env 파일을 확인하세요.")
        sys.exit(1)
    
    print(f"\n거래 모드: {cfg.trading_mode.value}")
    print(f"대상 종목: {len(ALL_TARGETS)}개")
    
    # 클라이언트 초기화
    client = BithumbClient.from_config(cfg)
    available_markets = {m.market for m in client.get_krw_markets()}
    
    # 헬스체크
    print("\n빗썸 연결 확인 중...")
    if not client.health_check():
        print("✗ 빗썸 서버 연결 실패")
        sys.exit(1)
    print("✓ 빗썸 서버 정상")
    
    # 캐시 + 수집기
    cache = OHLCVCache()
    collector = OHLCVCollector(client, cache)
    
    # 다운로드 시작
    print("\n" + "=" * 70)
    print("다운로드 시작")
    print("=" * 70)
    
    results = {'success': [], 'failed': []}
    
    for idx, (market, timeframe) in enumerate(ALL_TARGETS, 1):
        print(f"\n[{idx}/{len(ALL_TARGETS)}] {market} {timeframe}")
        resolved_market = MARKET_ALIASES.get(market, market)
        if resolved_market != market:
            print(f"  ⚠️  {market}은(는) 현재 {resolved_market}으로 매핑됩니다.")
        if resolved_market not in available_markets:
            print(f"  ✗ 건너뜀: {resolved_market}은(는) 현재 지원되지 않는 시장입니다.")
            results['failed'].append(f"{market} {timeframe} - unsupported market")
            continue
        
        try:
            df = collector.collect_historical(
                market=resolved_market,
                timeframe=timeframe,
                years=5.0,
                use_cache=True,
                verbose=True,
            )
            
            if df.empty:
                print(f"  ⚠️  빈 데이터")
                results['failed'].append(f"{market} {timeframe}")
            else:
                results['success'].append(f"{market} {timeframe} ({len(df)}봉)")
        
        except Exception as e:
            print(f"  ✗ 실패: {e}")
            results['failed'].append(f"{market} {timeframe} - {e}")
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("다운로드 완료 요약")
    print("=" * 70)
    
    print(f"\n✓ 성공: {len(results['success'])}개")
    for item in results['success']:
        print(f"  - {item}")
    
    if results['failed']:
        print(f"\n✗ 실패: {len(results['failed'])}개")
        for item in results['failed']:
            print(f"  - {item}")
    
    # 캐시 디렉토리 안내
    print(f"\n저장 위치: {cache.cache_dir}")
    print(f"파일 확인: ls {cache.cache_dir}")
    
    # 종료
    client.close()
    
    if results['failed']:
        print("\n일부 종목 실패. 위 로그 확인 후 재실행 가능.")
        sys.exit(1)
    else:
        print("\n✓ 모든 종목 다운로드 완료!")


if __name__ == '__main__':
    main()
