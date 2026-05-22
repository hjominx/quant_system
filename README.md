# 빗썸 안정형 자동매매 시스템

## 진행 상황

### ✅ Phase 1: 백테스트 프레임워크
### ✅ Phase 2a: 안전한 설정 관리
### ✅ Phase 2b: 빗썸 API 2.0 클라이언트
### ✅ Phase 2c: 실제 빗썸 연결 검증
### ✅ Phase 2d: 데이터 수집 모듈
### ✅ Phase 3a: Walk-Forward 검증 엔진
### ✅ Phase 3b: QPE 시장상태 적응형 복합 알파 엔진 ← 최신

**검증 통과: 60개 시나리오 (9 + 9 + 16 + 5 + 11 + 5 + 5)**

## 다음 단계
- [ ] 다종목 포트폴리오 QPE 검증
- [ ] EntanglementRiskManager: 종목 간 동조화/상관 리스크 관리
- [ ] 실거래 전 주문 집행 시뮬레이터 및 슬리피지 스트레스 테스트

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
python tests/test_walk_forward.py
python tests/test_quantum_probability_engine.py

# 4. 실데이터 수집 검증 (단일 종목)
python scripts/test_live_collection.py

# 5. 5년치 데이터 일괄 다운로드
python scripts/download_historical.py

# 6. Walk-Forward 검증
python scripts/walk_forward_sma.py
python scripts/walk_forward_qpe.py
```

## 디렉토리 구조

```
quant_system/
├── features/           # 시장 특징 추출기
├── regime/             # 시장상태 확률 엔진
├── risk/               # 확률 기반 포지션 사이징
├── strategies/         # 전략 인터페이스, SMA, QPE
├── backtest/           # 백테스트 엔진, 성과 지표, Walk-Forward
├── execution/          # 빗썸 클라이언트, 데이터 모델
├── utils/              # 설정, 로거, JWT 인증
├── data/               # 캐시, 수집기
│   └── cache/          # OHLCV CSV 캐시
├── scripts/            # 실행 스크립트
├── tests/              # 60개 검증 시나리오
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
- ✅ Walk-Forward 검증
- ✅ 증분 데이터 업데이트
- ✅ QPE: 시장상태 확률 + 복합 알파 + 확률 기반 포지션 사이징

## QPE 연구 엔진

QPE(Quantum Probability Engine)는 실제 양자컴퓨터가 아니라, 시장을 하나의 고정 상태로 단정하지 않고 여러 상태의 확률 분포로 해석하는 연구용 전략 엔진입니다.

- 시장상태: trend, mean-reversion, breakout, crash, neutral 확률을 계산
- 복합 알파: 추세, 모멘텀, 평균회귀, 돌파, 유동성/거래량 특징을 결합
- 포지션: edge, crash risk, uncertainty, volatility에 따라 연속적으로 축소/확대
- 검증: walk-forward 방식으로 out-of-sample 구간만 모아 평가
