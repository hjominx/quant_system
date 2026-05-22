# 빗썸 안정형 자동매매 시스템

## 진행 상황

### ✅ Phase 1: 백테스트 프레임워크
### ✅ Phase 2a: 안전한 설정 관리
### ✅ Phase 2b: 빗썸 API 2.0 클라이언트
### ✅ Phase 2c: 실제 빗썸 연결 검증
### ✅ Phase 2d: 데이터 수집 모듈
### ✅ Phase 3a: Walk-Forward 검증 엔진
### ✅ Phase 3b: QPE 시장상태 적응형 복합 알파 엔진
### ✅ Phase 3c: Sharpe/Sortino/Calmar 복합 Objective ← 최신
### 🧪 Phase 3d: QPE Trend Overlay 공격형 연구 후보
### 🧪 Phase 3e: ETH/Cash Regime Switch 최근 레짐 연구 후보
### 🧪 Phase 3f: ETH/Cash Micro Live 소액 파일럿 도구
### 🧪 Phase 3g: LIVE 전 계좌 위험관리 게이트

**검증 통과: 83개 시나리오 (9 + 9 + 16 + 5 + 11 + 6 + 5 + 4 + 4 + 14)**

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
python tests/test_qpe_trend_overlay.py
python tests/test_eth_cash_regime.py
python tests/test_eth_cash_micro_live.py

# 4. 실데이터 수집 검증 (단일 종목)
python scripts/test_live_collection.py

# 5. 5년치 데이터 일괄 다운로드
python scripts/download_historical.py

# 6. Walk-Forward 검증
python scripts/walk_forward_sma.py
python scripts/walk_forward_qpe.py
python scripts/walk_forward_qpe_trend_overlay.py
python scripts/walk_forward_eth_cash_regime.py

# 7. ETH/Cash 소액 파일럿 반복 실행 (기본은 실제 주문 없는 dry-run)
python scripts/run_eth_cash_micro_live.py --loop --max-order-krw 10000 --max-position-krw 30000
```

## 디렉토리 구조

```
quant_system/
├── features/           # 시장 특징 추출기
├── regime/             # 시장상태 확률 엔진
├── risk/               # 확률 기반 포지션 사이징, LIVE 위험 게이트
├── strategies/         # 전략 인터페이스, SMA, QPE
├── backtest/           # 백테스트 엔진, 성과 지표, 복합 Objective, Walk-Forward
├── execution/          # 빗썸 클라이언트, 데이터 모델
├── utils/              # 설정, 로거, JWT 인증
├── data/               # 캐시, 수집기
│   └── cache/          # OHLCV CSV 캐시
├── scripts/            # 실행 스크립트
├── tests/              # 83개 검증 시나리오
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
- ✅ 복합 Objective: Sharpe + Sortino + Calmar + 수익률 + MDD + Turnover
- ✅ 증분 데이터 업데이트
- ✅ QPE: 시장상태 확률 + 복합 알파 + 확률 기반 포지션 사이징

## QPE 연구 엔진

QPE(Quantum Probability Engine)는 실제 양자컴퓨터가 아니라, 시장을 하나의 고정 상태로 단정하지 않고 여러 상태의 확률 분포로 해석하는 연구용 전략 엔진입니다.

- 시장상태: trend, mean-reversion, breakout, crash, neutral 확률을 계산
- 복합 알파: 추세, 모멘텀, 평균회귀, 돌파, 유동성/거래량 특징을 결합
- 포지션: edge, crash risk, uncertainty, volatility에 따라 연속적으로 축소/확대
- 검증: walk-forward 방식으로 out-of-sample 구간만 모아 평가
- 최적화: `qpe_composite_score`로 Sharpe, Sortino, Calmar, 누적수익, MDD, 거래 빈도를 함께 평가

## QPE Trend Overlay 연구 후보

QPE Trend Overlay는 QPE v1의 방어성을 유지하되, SMA/모멘텀/돌파 기반 추세 포착력을 더 강하게 반영하는 공격형 후보입니다.

- BTC OOS: QPE v1보다 수익률은 개선됐지만 SMA Sharpe 기준선은 아직 미달
- ETH OOS: SMA 기준선보다 Sharpe가 개선된 구간이 확인됨
- 결론: 실전 전략이 아니라, 다음 포트폴리오/리밸런싱 연구의 중간 후보

## ETH/Cash Regime 연구 후보

ETH/Cash Regime은 2024년 말 이후 ETH 단순보유가 크게 흔들린 구간을 대상으로 만든 희소 진입형 연구 후보입니다.

- 구조: 장기 추세 + 30일 모멘텀 + 60일 돌파 조건이 동시에 맞을 때만 ETH 보유
- 위험관리: 60일 이탈, 모멘텀 둔화, RSI 약화, 단기 급락, 트레일링 스탑에서 현금 전환
- 최근 구간: 2024년 말 이후 고정 파라미터 기준 QPE Trend Overlay보다 높은 Sharpe 후보 확인
- 주의: 전체 5년 Walk-Forward 기준 최종 실전 전략은 아니며, 최근 레짐 필터/리스크 게이트 후보로 분류

## Micro Live 파일럿

`scripts/run_eth_cash_micro_live.py`는 ETH/Cash Regime을 소액으로 반복 실행하는 파일럿 도구입니다.

- 기본값은 dry-run이며 실제 주문을 보내지 않음
- 실제 주문은 `.env`의 `TRADING_MODE=LIVE`, `--live`, `--yes-i-understand-risk`가 모두 있을 때만 가능
- 최대 주문금액, 최대 ETH 평가금액, 최소 주문금액, 미체결 주문, 호가 스프레드 게이트를 강제
- 기본 프로필은 4시간봉 `intraday4h`이며 `--loop`, `--interval-minutes`, `--cooldown-minutes`, `--max-orders-per-day`로 반복 주기를 제어
- 계좌 위험 게이트가 일일 손실, 주간 손실, MDD, 킬스위치 파일(`logs/KILL_SWITCH_ACTIVE`)을 주문 직전에 강제
- 시장가 매수는 KRW 금액만, 시장가 매도는 ETH 수량만 전송하는 빗썸 V2 비대칭 규칙 준수
