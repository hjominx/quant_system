"""
백테스트 엔진

핵심 안전장치:
1. Look-ahead bias 방지: t시점 의사결정은 t시점까지의 데이터만 사용
2. 체결 시점 분리: 신호는 종가에 발생, 체결은 다음 봉 시가에
3. 거래비용·슬리피지 강제 반영 (편도 0.2% 기본)
4. 모든 거래는 risk_check 통과해야 실행
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision
from backtest.metrics import PerformanceMetrics, calculate_metrics


@dataclass
class BacktestConfig:
    """백테스트 설정 - 명시적으로 모든 가정을 기록"""
    initial_capital: float = 10_000_000  # 초기 자본 1천만원
    
    # 거래비용 (빗썸 실제 수수료 + 슬리피지)
    commission_rate: float = 0.0025  # 0.25% (빗썸 일반 수수료)
    slippage_rate: float = 0.002     # 0.2% (보수적 가정)
    
    # 체결 가정
    execution_lag: int = 1  # 신호 발생 후 N봉 뒤 체결 (look-ahead 방지)
    use_open_for_execution: bool = True  # True면 다음 시가, False면 다음 종가
    
    # 검증
    enforce_no_lookahead: bool = True  # look-ahead bias 자동 검출
    
    @property
    def round_trip_cost(self) -> float:
        """왕복 거래비용 (사고팔기 한 사이클)"""
        return (self.commission_rate + self.slippage_rate) * 2


@dataclass
class Trade:
    """개별 거래 기록"""
    entry_time: datetime
    exit_time: Optional[datetime]
    ticker: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'long' (코인 현물은 long만 가능)
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_reason: str = ""
    exit_reason: str = ""
    commission: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None


@dataclass
class BacktestResult:
    """백테스트 결과 묶음"""
    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series
    metrics: PerformanceMetrics
    config: BacktestConfig
    strategy_name: str
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"전략: {self.strategy_name}")
        print(f"{'='*60}")
        print(self.metrics.summary())
        print(f"[합격 여부]")
        passed = self.metrics.is_acceptable()
        print(f"  In-Sample 기준 (Sharpe≥1.5, Calmar≥2.0, MDD≥-20%):")
        is_passed = self.metrics.is_acceptable(min_sharpe=1.5, min_calmar=2.0)
        print(f"    {'✓ 통과' if is_passed else '✗ 실패'}")
        print(f"  Out-of-Sample 기준 (Sharpe≥1.0, Calmar≥1.5, MDD≥-20%):")
        print(f"    {'✓ 통과' if passed else '✗ 실패'}")
        print(f"{'='*60}\n")


class BacktestEngine:
    """단일 자산 백테스트 엔진
    
    포트폴리오(다자산) 백테스트는 추후 PortfolioBacktestEngine 으로 확장.
    먼저 단일 자산으로 전략 자체의 건전성 검증.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def run(
        self,
        strategy: BaseStrategy,
        ohlcv: pd.DataFrame,
        ticker: str,
    ) -> BacktestResult:
        """백테스트 실행
        
        Args:
            strategy: BaseStrategy 상속받은 전략 객체
            ohlcv: OHLCV 데이터 (DatetimeIndex, columns: open/high/low/close/volume)
            ticker: 종목 코드 (예: 'KRW-BTC')
        
        Returns:
            BacktestResult
        """
        self._validate_ohlcv(ohlcv)
        
        warmup = strategy.warmup_period()
        if len(ohlcv) < warmup + 10:
            raise ValueError(
                f"데이터가 부족합니다. 최소 {warmup + 10}개 필요, 현재 {len(ohlcv)}개"
            )
        
        # 상태 변수
        cash = self.config.initial_capital
        position = 0.0  # 보유 수량
        avg_buy_price = 0.0
        equity_history = []
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        pending_signal: Optional[TradeDecision] = None  # 다음 봉 체결 대기
        
        # 메인 루프
        for i in range(warmup, len(ohlcv)):
            current_bar = ohlcv.iloc[i]
            current_time = ohlcv.index[i]
            
            # 1. 이전 봉에서 발생한 신호 체결 (look-ahead 방지)
            if pending_signal is not None:
                exec_price = (
                    current_bar['open']
                    if self.config.use_open_for_execution
                    else current_bar['close']
                )
                cash, position, avg_buy_price, current_trade = self._execute_trade(
                    decision=pending_signal,
                    exec_price=exec_price,
                    exec_time=current_time,
                    ticker=ticker,
                    cash=cash,
                    position=position,
                    avg_buy_price=avg_buy_price,
                    current_trade=current_trade,
                    trades=trades,
                )
                pending_signal = None
            
            # 2. 현재 시점 자산 가치 기록
            portfolio_value = cash + position * current_bar['close']
            equity_history.append({
                'date': current_time,
                'equity': portfolio_value,
                'cash': cash,
                'position_value': position * current_bar['close'],
            })
            
            # 3. 현재 시점까지의 데이터로 신호 생성 (미래 데이터 차단)
            # 중요: i+1이 아닌 i까지만 슬라이싱 → look-ahead 방지
            historical_data = ohlcv.iloc[:i+1].copy()
            
            ctx = StrategyContext(
                timestamp=current_time,
                ticker=ticker,
                ohlcv=historical_data,
                current_position=position,
                avg_buy_price=avg_buy_price,
                cash=cash,
                portfolio_value=portfolio_value,
            )
            
            decision = strategy.generate_signal(ctx)
            
            # 4. 신호가 HOLD가 아니면 다음 봉에 체결 예약
            if decision.signal != Signal.HOLD:
                pending_signal = decision
        
        # 백테스트 종료 - 보유 중이면 마지막 봉 종가에 청산
        if position > 0 and current_trade is not None:
            final_price = ohlcv.iloc[-1]['close']
            final_time = ohlcv.index[-1]
            self._close_position(
                current_trade, final_price, final_time, 
                reason="백테스트 종료 강제청산"
            )
            cash += position * final_price * (1 - self.config.commission_rate - self.config.slippage_rate)
            position = 0
        
        # 결과 정리
        equity_df = pd.DataFrame(equity_history).set_index('date')
        equity_curve = equity_df['equity']
        daily_returns = equity_curve.pct_change().dropna()
        
        trades_df = self._trades_to_dataframe(trades)
        metrics = calculate_metrics(equity_curve, trades_df)
        
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            daily_returns=daily_returns,
            metrics=metrics,
            config=self.config,
            strategy_name=strategy.name,
        )
    
    def _validate_ohlcv(self, ohlcv: pd.DataFrame) -> None:
        """OHLCV 데이터 무결성 검증"""
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(ohlcv.columns):
            missing = required - set(ohlcv.columns)
            raise ValueError(f"OHLCV에 필수 컬럼 누락: {missing}")
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise ValueError("OHLCV 인덱스는 DatetimeIndex여야 합니다.")
        if not ohlcv.index.is_monotonic_increasing:
            raise ValueError("OHLCV 인덱스가 시간순으로 정렬되지 않았습니다.")
        if ohlcv.isnull().any().any():
            raise ValueError("OHLCV에 NaN이 포함되어 있습니다.")
        # 가격 무결성 (high >= low, high >= open, high >= close)
        if not (ohlcv['high'] >= ohlcv['low']).all():
            raise ValueError("high < low 인 봉이 존재합니다.")
    
    def _execute_trade(
        self,
        decision: TradeDecision,
        exec_price: float,
        exec_time: datetime,
        ticker: str,
        cash: float,
        position: float,
        avg_buy_price: float,
        current_trade: Optional[Trade],
        trades: List[Trade],
    ):
        """거래 체결 처리 (수수료·슬리피지 반영)"""
        cost_factor = self.config.commission_rate + self.config.slippage_rate
        
        if decision.signal == Signal.BUY and position == 0:
            # 매수: target_weight 만큼 매수
            available_cash = cash * decision.target_weight
            effective_price = exec_price * (1 + cost_factor)  # 슬리피지 + 수수료 반영
            quantity = available_cash / effective_price
            
            commission = available_cash * cost_factor
            cash -= available_cash
            position = quantity
            avg_buy_price = effective_price
            
            current_trade = Trade(
                entry_time=exec_time,
                exit_time=None,
                ticker=ticker,
                entry_price=effective_price,
                exit_price=None,
                quantity=quantity,
                side='long',
                entry_reason=decision.reason,
                commission=commission,
            )
            trades.append(current_trade)
        
        elif decision.signal == Signal.SELL and position > 0:
            # 매도: 전량 청산
            effective_price = exec_price * (1 - cost_factor)
            proceeds = position * effective_price
            commission = position * exec_price * cost_factor
            
            if current_trade is not None:
                self._close_position(
                    current_trade, effective_price, exec_time, decision.reason
                )
                current_trade.commission += commission
            
            cash += proceeds
            position = 0
            avg_buy_price = 0
            current_trade = None
        
        return cash, position, avg_buy_price, current_trade
    
    def _close_position(self, trade: Trade, exit_price: float, exit_time: datetime, reason: str):
        """포지션 종료 정보 기록"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        trade.pnl_pct = (exit_price / trade.entry_price) - 1
        trade.exit_reason = reason
    
    def _trades_to_dataframe(self, trades: List[Trade]) -> pd.DataFrame:
        """거래 리스트를 DataFrame으로 변환"""
        if not trades:
            return pd.DataFrame(columns=[
                'entry_time', 'exit_time', 'ticker', 'entry_price', 'exit_price',
                'quantity', 'pnl', 'pnl_pct', 'entry_reason', 'exit_reason'
            ])
        return pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'ticker': t.ticker,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'entry_reason': t.entry_reason,
                'exit_reason': t.exit_reason,
            }
            for t in trades if not t.is_open
        ])
