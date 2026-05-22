"""
중앙 로깅 시스템

설계 원칙:
1. 모든 모듈은 동일한 로거 인터페이스 사용
2. 콘솔 + 파일 동시 출력 (파일은 운영 후 분석용)
3. API 키 등 민감정보는 자동 마스킹
4. 거래 관련 로그는 별도 파일로 분리 (감사 추적)
"""
import logging
import logging.handlers
import re
import sys
from pathlib import Path
from typing import Optional


# 민감정보 마스킹 패턴
SENSITIVE_PATTERNS = [
    # API 키 (32자 이상 영숫자)
    (re.compile(r'\b[a-zA-Z0-9]{32,}\b'), lambda m: f"{m.group()[:4]}****{m.group()[-4:]}"),
    # JWT 토큰
    (re.compile(r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+'), lambda m: 'JWT_TOKEN_MASKED'),
    # Bearer 토큰
    (re.compile(r'Bearer\s+[a-zA-Z0-9_-]+', re.IGNORECASE), lambda m: 'Bearer ****'),
]


class SensitiveFilter(logging.Filter):
    """로그 메시지에서 민감정보를 자동 마스킹하는 필터"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, replacer in SENSITIVE_PATTERNS:
                record.msg = pattern.sub(replacer, record.msg)
        
        # args 도 검사
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacer in SENSITIVE_PATTERNS:
                        arg = pattern.sub(replacer, arg)
                new_args.append(arg)
            record.args = tuple(new_args)
        
        return True


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_file_log: bool = True,
) -> None:
    """전역 로깅 시스템 초기화 (시스템 시작 시 한 번만 호출)
    
    Args:
        log_level: DEBUG / INFO / WARNING / ERROR
        log_dir: 로그 파일 저장 디렉토리 (기본: 프로젝트 루트/logs)
        enable_file_log: 파일 로깅 활성화 여부
    """
    global _initialized
    
    if _initialized:
        return
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 기존 핸들러 제거 (재초기화 안전성)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    sensitive_filter = SensitiveFilter()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sensitive_filter)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (회전 로그, 일별)
    if enable_file_log:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 일반 로그 파일
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "system.log",
            when="midnight",
            backupCount=30,  # 30일 보관
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sensitive_filter)
        root_logger.addHandler(file_handler)
        
        # 거래 전용 로그 파일 (별도 - 감사 추적용)
        trade_logger = logging.getLogger("trade")
        trade_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "trade.log",
            when="midnight",
            backupCount=365,  # 1년 보관
            encoding="utf-8",
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(formatter)
        trade_handler.addFilter(sensitive_filter)
        trade_logger.addHandler(trade_handler)
        trade_logger.propagate = False  # 루트 로거로 전파 방지
    
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 가져오기
    
    사용법:
        logger = get_logger(__name__)
        logger.info("메시지")
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)


def get_trade_logger() -> logging.Logger:
    """거래 전용 로거 (모든 매매 기록 영구 보관)
    
    사용법:
        trade_log = get_trade_logger()
        trade_log.info("BUY KRW-BTC qty=0.001 price=80000000 reason='골든크로스'")
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger("trade")
