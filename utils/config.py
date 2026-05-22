"""
중앙 설정 관리 모듈

설계 원칙:
1. 모든 설정값은 이 모듈을 통해서만 접근 (단일 진실 공급원)
2. 시스템 시작 시점에 모든 값을 검증 (런타임 에러 방지)
3. API 키는 로그·에러 메시지에 절대 노출되지 않음
4. TRADING_MODE 가 LIVE 가 아니면 실거래 차단 (안전장치)

사용법:
    from utils.config import get_config
    cfg = get_config()
    api_key = cfg.bithumb_api_key  # 자동 검증된 값
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import os
import sys


class TradingMode(str, Enum):
    """거래 모드 - LIVE 가 아니면 실거래는 무조건 차단"""
    DEVELOPMENT = "DEVELOPMENT"  # 개발용, 모든 주문 dry-run
    PAPER = "PAPER"              # 종이매매, 실거래 차단
    LIVE = "LIVE"                # 실거래 활성화 (충분히 검증 후)
    
    @property
    def is_live(self) -> bool:
        return self == TradingMode.LIVE


class ConfigError(Exception):
    """설정 관련 에러 - 시스템 시작을 막아야 하는 치명적 오류"""
    pass


def _load_env_file(env_path: Path) -> None:
    """단순 .env 파일 파서 (python-dotenv 의존성 없이 구현)
    
    외부 의존성을 최소화하기 위해 직접 구현.
    형식: KEY=VALUE (한 줄에 하나, # 으로 시작하는 줄은 주석)
    """
    if not env_path.exists():
        return
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 빈 줄 또는 주석 무시
            if not line or line.startswith('#'):
                continue
            
            # KEY=VALUE 형식 파싱
            if '=' not in line:
                raise ConfigError(
                    f".env 파일 {line_num}번째 줄 형식 오류: '{line}' "
                    f"(KEY=VALUE 형식이어야 함)"
                )
            
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            
            # 따옴표 제거 (있다면)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            
            # 환경변수가 이미 설정되어 있으면 덮어쓰지 않음
            # (시스템 환경변수가 .env 보다 우선)
            if key not in os.environ:
                os.environ[key] = value


def _get_env(
    key: str,
    default: Optional[str] = None,
    required: bool = False,
    secret: bool = False,
) -> Optional[str]:
    """환경변수 안전하게 읽기
    
    Args:
        key: 환경변수 이름
        default: 없을 때 기본값
        required: True면 없을 때 에러
        secret: True면 에러 메시지에 값을 노출하지 않음
    """
    value = os.environ.get(key, default)
    
    if required and (value is None or value == "" or value.startswith("your_")):
        raise ConfigError(
            f"필수 환경변수 '{key}' 가 설정되지 않았습니다. "
            f".env 파일을 확인하세요. (.env.example 참고)"
        )
    
    return value


def _get_env_float(key: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """환경변수를 float 로 읽고 범위 검증"""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"환경변수 '{key}' 값이 숫자가 아님: '{raw}'")
    
    if min_val is not None and value < min_val:
        raise ConfigError(f"'{key}' 값이 최소값 {min_val} 미만: {value}")
    if max_val is not None and value > max_val:
        raise ConfigError(f"'{key}' 값이 최대값 {max_val} 초과: {value}")
    
    return value


@dataclass(frozen=True)
class Config:
    """시스템 전체 설정값 (immutable)
    
    frozen=True 로 만들어 한 번 로드된 후에는 변경 불가.
    런타임에 누군가 실수로 위험 한도를 바꾸는 사고 방지.
    """
    # --- 빗썸 API ---
    bithumb_api_key: str
    bithumb_secret_key: str = field(repr=False)  # repr 시 노출 금지
    
    # --- 거래 모드 ---
    trading_mode: TradingMode
    
    # --- 위험 한도 ---
    daily_loss_limit: float
    weekly_loss_limit: float
    max_drawdown_limit: float
    max_position_per_coin: float
    max_total_crypto: float
    
    # --- 자본 ---
    initial_capital: float
    
    # --- 알림 ---
    telegram_bot_token: Optional[str] = field(default=None, repr=False)
    telegram_chat_id: Optional[str] = None
    
    # --- 로깅 ---
    log_level: str = "INFO"
    
    def __post_init__(self):
        """초기화 직후 무결성 검증"""
        # 위험 한도 일관성 체크
        if self.daily_loss_limit >= self.weekly_loss_limit:
            raise ConfigError(
                f"일일 손실 한도({self.daily_loss_limit})가 "
                f"주간 한도({self.weekly_loss_limit})보다 크거나 같음"
            )
        if self.weekly_loss_limit >= self.max_drawdown_limit:
            raise ConfigError(
                f"주간 손실 한도({self.weekly_loss_limit})가 "
                f"MDD 한도({self.max_drawdown_limit})보다 크거나 같음"
            )
        
        # 비중 한도 검증
        if not (0 < self.max_position_per_coin <= 1.0):
            raise ConfigError(
                f"max_position_per_coin 은 0~1 사이여야 함: {self.max_position_per_coin}"
            )
        if not (0 < self.max_total_crypto <= 1.0):
            raise ConfigError(
                f"max_total_crypto 은 0~1 사이여야 함: {self.max_total_crypto}"
            )
        
        # API 키 형식 기본 검증
        if len(self.bithumb_api_key) < 10:
            raise ConfigError("BITHUMB_API_KEY 형식이 잘못된 것 같습니다 (너무 짧음)")
        if len(self.bithumb_secret_key) < 10:
            raise ConfigError("BITHUMB_SECRET_KEY 형식이 잘못된 것 같습니다 (너무 짧음)")
        
        # 자본 검증
        if self.initial_capital <= 0:
            raise ConfigError(f"initial_capital 이 0 이하: {self.initial_capital}")
    
    @property
    def is_live_trading(self) -> bool:
        """실거래 모드 여부 (안전장치)"""
        return self.trading_mode.is_live

    @property
    def should_dry_run_orders(self) -> bool:
        """주문을 실제로 보내면 안 되는 모드인지 여부"""
        return not self.is_live_trading
    
    def mask_secret(self, value: str) -> str:
        """로그·디스플레이용 시크릿 마스킹
        예: 'abcd1234efgh5678' → 'abcd****5678'
        """
        if len(value) <= 8:
            return "****"
        return f"{value[:4]}****{value[-4:]}"
    
    def summary(self) -> str:
        """사람이 읽기 좋은 설정 요약 (시크릿은 마스킹)"""
        return f"""
=== 시스템 설정 요약 ===
거래 모드:         {self.trading_mode.value}
{'⚠️  실거래 활성화 ⚠️' if self.is_live_trading else '✓ 안전 모드 (실거래 차단)'}

API 키:           {self.mask_secret(self.bithumb_api_key)}
시크릿 키:        {self.mask_secret(self.bithumb_secret_key)}

[위험 한도]
일일 손실:        -{self.daily_loss_limit * 100:.1f}%
주간 손실:        -{self.weekly_loss_limit * 100:.1f}%
최대 낙폭(MDD):   -{self.max_drawdown_limit * 100:.1f}%

[포지션 한도]
단일 코인 최대:   {self.max_position_per_coin * 100:.0f}%
전체 코인 최대:   {self.max_total_crypto * 100:.0f}%

[자본]
초기 자본:        {self.initial_capital:,.0f}원

[알림]
텔레그램:         {'설정됨' if self.telegram_bot_token else '미설정'}

[로깅]
레벨:             {self.log_level}
======================
"""


# 모듈 레벨 캐시 (한 번만 로드)
_cached_config: Optional[Config] = None


def get_config(env_file: Optional[Path] = None, force_reload: bool = False) -> Config:
    """시스템 설정 가져오기 (싱글톤)
    
    Args:
        env_file: .env 파일 경로 (기본: 프로젝트 루트의 .env)
        force_reload: True면 캐시 무시하고 재로드 (테스트용)
    
    Returns:
        검증된 Config 객체
    
    Raises:
        ConfigError: 필수값 누락 또는 검증 실패 시
    """
    global _cached_config
    
    if _cached_config is not None and not force_reload:
        return _cached_config
    
    # .env 파일 로드
    if env_file is None:
        # 프로젝트 루트 찾기 (config.py 위치 기준)
        env_file = Path(__file__).parent.parent / ".env"
    
    _load_env_file(env_file)
    
    # 거래 모드 파싱
    mode_str = _get_env("TRADING_MODE", default="DEVELOPMENT")
    try:
        trading_mode = TradingMode(mode_str.upper())
    except ValueError:
        raise ConfigError(
            f"TRADING_MODE 값이 잘못됨: '{mode_str}' "
            f"(가능값: {[m.value for m in TradingMode]})"
        )
    
    # Config 구성 (각 필드를 명시적으로 검증)
    config = Config(
        bithumb_api_key=_get_env("BITHUMB_API_KEY", required=True, secret=True),
        bithumb_secret_key=_get_env("BITHUMB_SECRET_KEY", required=True, secret=True),
        trading_mode=trading_mode,
        daily_loss_limit=_get_env_float("DAILY_LOSS_LIMIT", default=0.03, min_val=0.001, max_val=0.10),
        weekly_loss_limit=_get_env_float("WEEKLY_LOSS_LIMIT", default=0.07, min_val=0.005, max_val=0.20),
        max_drawdown_limit=_get_env_float("MAX_DRAWDOWN_LIMIT", default=0.15, min_val=0.01, max_val=0.50),
        max_position_per_coin=_get_env_float("MAX_POSITION_PER_COIN", default=0.25, min_val=0.01, max_val=1.0),
        max_total_crypto=_get_env_float("MAX_TOTAL_CRYPTO", default=0.80, min_val=0.01, max_val=1.0),
        initial_capital=_get_env_float("INITIAL_CAPITAL", default=10_000_000, min_val=10_000),
        telegram_bot_token=_get_env("TELEGRAM_BOT_TOKEN", secret=True),
        telegram_chat_id=_get_env("TELEGRAM_CHAT_ID"),
        log_level=_get_env("LOG_LEVEL", default="INFO"),
    )
    
    _cached_config = config
    return config


def reset_config() -> None:
    """캐시 초기화 (테스트용)"""
    global _cached_config
    _cached_config = None
