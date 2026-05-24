"""
BaseStrategy: 모든 전략이 상속하는 추상 클래스

설계 원칙:
1. 시점 t의 의사결정은 시점 t 이전 데이터만 사용 (look-ahead 방지)
2. 신호 생성과 포지션 사이징 분리
3. 동일 인터페이스로 모든 전략 평가 가능
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
import pandas as pd


class Signal(Enum):
    """매매 신호"""
    BUY = 1
    HOLD = 0
    SELL = -1


@dataclass
class StrategyContext:
    """전략 실행 시점의 컨텍스트
    
    백테스트와 실전 모두에서 동일하게 사용됨.
    백테스트에서는 과거 시뮬레이션 시점, 실전에서는 현재.
    """
    timestamp: datetime
    ticker: str
    ohlcv: pd.DataFrame  # 현재 시점까지의 데이터만 포함 (look-ahead 방지)
    current_position: float = 0.0  # 현재 보유 수량 (0이면 미보유)
    avg_buy_price: float = 0.0  # 평균 매수가
    cash: float = 0.0  # 가용 현금
    portfolio_value: float = 0.0  # 전체 포트폴리오 가치
    metadata: Dict = field(default_factory=dict)  # 전략별 추가 정보


@dataclass
class TradeDecision:
    """전략이 내리는 거래 결정"""
    signal: Signal
    target_weight: float  # 목표 비중 (0.0 ~ 1.0)
    reason: str  # 결정 이유 (로깅·디버깅용 필수)
    confidence: float = 1.0  # 신호 강도 (0.0 ~ 1.0, 포지션 사이징에 사용)
    stop_loss_pct: Optional[float] = None  # 손절 라인 (예: -0.05)
    take_profit_pct: Optional[float] = None  # 익절 라인
    order_krw: Optional[float] = None  # 전략이 요청한 고정 KRW 주문액


class BaseStrategy(ABC):
    """모든 전략의 추상 베이스 클래스
    
    상속받는 클래스는 generate_signal() 만 구현하면 됨.
    나머지는 프레임워크가 처리 (백테스트, 검증, 실전 실행 모두).
    """
    
    def __init__(self, name: str, params: Dict):
        self.name = name
        self.params = params
        self.required_history_days = params.get('required_history_days', 90)
        self._validate_params()
    
    @abstractmethod
    def _validate_params(self) -> None:
        """파라미터 유효성 검증 (필수 구현)"""
        pass
    
    @abstractmethod
    def generate_signal(self, ctx: StrategyContext) -> TradeDecision:
        """주어진 컨텍스트에서 거래 결정 생성 (필수 구현)
        
        중요: ctx.ohlcv 는 현재 시점까지만 포함되어 있음.
        미래 데이터에 접근하려 하면 안 됨.
        """
        pass
    
    def warmup_period(self) -> int:
        """전략이 필요로 하는 최소 과거 데이터 일수
        
        백테스트 시작점에서 이만큼은 신호를 생성하지 않음.
        """
        return self.required_history_days
    
    def __repr__(self) -> str:
        return f"<Strategy {self.name} params={self.params}>"
