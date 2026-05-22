"""
성과 지표 계산 모듈

원칙: 수익률 외에도 위험조정수익률을 반드시 함께 본다.
샤프지수 1.5, 칼마지수 2.0이 우리 시스템의 합격선.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """백테스트 성과 종합 지표"""
    # 수익성
    total_return: float        # 누적 수익률
    annualized_return: float   # 연환산 수익률
    
    # 위험
    volatility: float          # 연환산 변동성
    max_drawdown: float        # MDD (음수)
    avg_drawdown: float        # 평균 낙폭
    
    # 위험조정수익
    sharpe_ratio: float        # 샤프지수 (변동성 대비)
    sortino_ratio: float       # 소르티노지수 (하방변동성 대비)
    calmar_ratio: float        # 칼마지수 (MDD 대비)
    
    # 매매 통계
    total_trades: int
    win_rate: float            # 승률
    profit_factor: float       # 손익비 (총수익/총손실)
    avg_win: float             # 평균 수익 거래 수익률
    avg_loss: float            # 평균 손실 거래 수익률 (음수)
    
    # 기간
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    days: int
    
    def is_acceptable(self, 
                      min_sharpe: float = 1.0,
                      min_calmar: float = 1.5,
                      max_mdd: float = -0.20) -> bool:
        """우리 시스템의 합격 기준 통과 여부
        
        기본값은 'Out-of-Sample 검증 최소 기준'.
        In-Sample은 더 엄격하게 (sharpe 1.5, calmar 2.0).
        """
        return (
            self.sharpe_ratio >= min_sharpe and
            self.calmar_ratio >= min_calmar and
            self.max_drawdown >= max_mdd
        )
    
    def summary(self) -> str:
        """사람이 읽기 좋은 요약"""
        return f"""
=== 성과 요약 ({self.start_date.date()} ~ {self.end_date.date()}, {self.days}일) ===

[수익성]
  누적 수익률:    {self.total_return * 100:>8.2f}%
  연환산 수익률:  {self.annualized_return * 100:>8.2f}%

[위험]
  연환산 변동성:  {self.volatility * 100:>8.2f}%
  최대 낙폭(MDD): {self.max_drawdown * 100:>8.2f}%
  평균 낙폭:      {self.avg_drawdown * 100:>8.2f}%

[위험조정수익] ← 가장 중요
  샤프지수:       {self.sharpe_ratio:>8.2f}  (1.5 이상 합격)
  소르티노지수:   {self.sortino_ratio:>8.2f}
  칼마지수:       {self.calmar_ratio:>8.2f}  (2.0 이상 합격)

[매매 통계]
  총 거래수:      {self.total_trades:>8d}
  승률:           {self.win_rate * 100:>8.2f}%
  손익비:         {self.profit_factor:>8.2f}
  평균 수익:      {self.avg_win * 100:>8.2f}%
  평균 손실:      {self.avg_loss * 100:>8.2f}%
"""


def calculate_metrics(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 365,
) -> PerformanceMetrics:
    """일별 자산곡선으로부터 모든 성과 지표 계산
    
    Args:
        equity_curve: 일별 포트폴리오 가치 시리즈 (DatetimeIndex)
        trades: 개별 거래 기록 (선택, 없으면 매매통계는 0으로 채움)
        risk_free_rate: 무위험 수익률 (연환산, 기본 3%)
        periods_per_year: 연환산 계수 (코인은 365, 주식은 252)
    
    Returns:
        PerformanceMetrics 객체
    """
    if len(equity_curve) < 2:
        raise ValueError("equity_curve는 최소 2개 데이터포인트가 필요합니다.")
    
    # 일별 수익률
    returns = equity_curve.pct_change().dropna()
    
    # 수익성 지표
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 위험 지표
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    # MDD 계산 (peak로부터의 낙폭)
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
    
    # 위험조정 수익률
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe_ratio = (
        excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
        if returns.std() > 0 else 0.0
    )
    
    # 소르티노: 하방 변동성만 사용
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = (
        excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
        if downside_std > 0 else 0.0
    )
    
    # 칼마: 연환산 수익률 / |MDD|
    calmar_ratio = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0 else 0.0
    )
    
    # 매매 통계
    if trades is not None and len(trades) > 0:
        wins = trades[trades['pnl_pct'] > 0]
        losses = trades[trades['pnl_pct'] <= 0]
        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_win = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
    else:
        total_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        max_drawdown=max_drawdown,
        avg_drawdown=avg_drawdown,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        start_date=equity_curve.index[0],
        end_date=equity_curve.index[-1],
        days=days,
    )
