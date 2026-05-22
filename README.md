# 빗썸 안정형 자동매매 시스템

## 진행 상황

### ✅ Phase 1: 백테스트 프레임워크
### ✅ Phase 2a: 안전한 설정 관리
### ✅ Phase 2b: 빗썸 API 2.0 클라이언트
### ✅ Phase 2c: 실제 빗썸 연결 검증
### ✅ Phase 2d: 데이터 수집 모듈 ← 최신

**검증 통과: 50개 시나리오 (9 + 9 + 16 + 5 + 11)**

## 다음 단계 (Phase 3)
- [ ] Walk-Forward 검증 엔진
- [ ] 위험관리 레이어
- [ ] 실전 전략 구현 (추세추종 + 평균회귀 + 변동성 타게팅)

## 빠른 시작

```bash
# 1. 환경변수 설정
cp .env.example .env  # API 키 입력

# 2. 가상환경
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy requests PyJWT

# 3. 검증 테스트
python tests/test_config.py
python tests/test_auth.py
python tests/test_bithumb_client.py
python tests/test_backtest_engine.py
python tests/test_data.py

# 4. 실데이터 수집 검증 (단일 종목)
python scripts/test_live_collection.py

# 5. 5년치 데이터 일괄 다운로드
python scripts/download_historical.py
```

## 디렉토리 구조

```
quant_system/
├── strategies/         # 전략 인터페이스
├── backtest/           # 백테스트 엔진, 성과 지표
├── execution/          # 빗썸 클라이언트, 데이터 모델
├── utils/              # 설정, 로거, JWT 인증
├── data/               # 캐시, 수집기 (NEW)
│   └── cache/          # OHLCV CSV 캐시
├── scripts/            # 실행 스크립트 (NEW)
├── tests/              # 47개 검증 시나리오
├── risk/               # (예정) 위험관리
├── configs/, results/, logs/
```

## 핵심 가치

- ✅ 모든 가격·수량 Decimal (부동소수점 오차 0)
- ✅ 빗썸 V2 비대칭 주문 규칙 (매수=금액, 매도=수량)
- ✅ LIVE 모드 외 실주문 이중 차단
- ✅ 자동 재시도 + 영구 에러 즉시 정지
- ✅ 시크릿 자동 마스킹
- ✅ Look-ahead bias 수학적 방지
- ✅ 거래비용 강제 반영
- ✅ Walk-Forward 검증 (예정)
- ✅ 증분 데이터 업데이트
