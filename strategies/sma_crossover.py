"""
단순 이동평균 교차 전략 (SMA Crossover)

목적: 백테스트 엔진이 제대로 동작하는지 검증하는 가장 단순한 전략.
실전용이 아님. 진짜 전략은 추후 별도 파일에서 작성.

규칙:
- 단기 이동평균이 장기 이동평균 위로 돌파 → 매수
- 단기 이동평균이 장기 이동평균 아래로 돌파 → 매도
"""
from typing import Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, Signal, StrategyContext, TradeDecision


class SMACrossoverStrategy(BaseStrategy):
    """단순 SMA 교차 전략 (검증용)"""
    
    def __init__(self, params: Dict = None):
        params = params or {}
        params.setdefault('short_window', 20)
        params.setdefault('long_window', 60)
        params.setdefault('target_weight', 0.95)  # 매수 시 자본의 95%만 사용
        params.setdefault('required_history_days', params['long_window'] + 5)
        super().__init__(name="SMA_Crossover", params=params)
    
    def _validate_params(self) -> None:
        assert self.params['short_window'] > 0
        assert self.params['long_window'] > self.params['short_window']
        assert 0 < self.params['target_weight'] <= 1.0
    
    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        close = ctx.ohlcv['close']
        
        # 데이터 부족 시 HOLD
        if len(close) < self.params['long_window'] + 1:
            return TradeDecision(
                signal=Signal.HOLD,
                target_weight=0.0,
                reason="warmup 기간",
            )
        
        # 이동평균 계산 (현재 시점까지의 데이터만 사용)
        short_ma = close.rolling(self.params['short_window']).mean()
        long_ma = close.rolling(self.params['long_window']).mean()
        
        # 현재 봉과 직전 봉의 위치 비교
        curr_short = short_ma.iloc[-1]
        curr_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        # 골든크로스: 매수
        if prev_short <= prev_long and curr_short > curr_long and ctx.current_position == 0:
            return TradeDecision(
                signal=Signal.BUY,
                target_weight=self.params['target_weight'],
                reason=f"골든크로스 (SMA{self.params['short_window']} > SMA{self.params['long_window']})",
            )
        
        # 데드크로스: 매도
        if prev_short >= prev_long and curr_short < curr_long and ctx.current_position > 0:
            return TradeDecision(
                signal=Signal.SELL,
                target_weight=0.0,
                reason=f"데드크로스 (SMA{self.params['short_window']} < SMA{self.params['long_window']})",
            )
        
        return TradeDecision(
            signal=Signal.HOLD,
            target_weight=0.0,
            reason="신호 없음",
        )
