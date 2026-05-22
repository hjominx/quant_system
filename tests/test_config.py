"""
설정 모듈 검증 테스트

핵심 검증:
1. 기본 로딩이 정상 동작하는가
2. 필수값 누락 시 에러가 나는가
3. 잘못된 값(범위 초과, 형식 오류) 검출하는가
4. 위험 한도 일관성 검증이 작동하는가
5. API 키가 로그·repr 에 노출되지 않는가  ← 가장 중요
6. .env 파일 파싱이 정확한가
"""
import sys
import os
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    Config, TradingMode, ConfigError,
    get_config, reset_config, _load_env_file
)


def _clean_env():
    """테스트 환경 깨끗하게 초기화"""
    keys_to_clean = [
        'BITHUMB_API_KEY', 'BITHUMB_SECRET_KEY', 'TRADING_MODE',
        'DAILY_LOSS_LIMIT', 'WEEKLY_LOSS_LIMIT', 'MAX_DRAWDOWN_LIMIT',
        'MAX_POSITION_PER_COIN', 'MAX_TOTAL_CRYPTO', 'INITIAL_CAPITAL',
        'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DISCORD_WEBHOOK_URL', 'LOG_LEVEL',
    ]
    for key in keys_to_clean:
        os.environ.pop(key, None)
    reset_config()


def _write_temp_env(content: str) -> Path:
    """임시 .env 파일 생성"""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False, encoding='utf-8')
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ============================================================
# 테스트 1: 기본 로딩
# ============================================================
def test_basic_loading():
    print("\n[테스트 1] 정상 환경에서 설정 로딩")
    print("-" * 60)
    _clean_env()
    
    env_content = """
BITHUMB_API_KEY=test_api_key_1234567890abcdef
BITHUMB_SECRET_KEY=test_secret_key_1234567890abcdef
TRADING_MODE=DEVELOPMENT
DAILY_LOSS_LIMIT=0.03
WEEKLY_LOSS_LIMIT=0.07
MAX_DRAWDOWN_LIMIT=0.15
MAX_POSITION_PER_COIN=0.25
MAX_TOTAL_CRYPTO=0.80
INITIAL_CAPITAL=10000000
LOG_LEVEL=INFO
"""
    env_path = _write_temp_env(env_content)
    
    try:
        cfg = get_config(env_file=env_path, force_reload=True)
        
        assert cfg.bithumb_api_key == "test_api_key_1234567890abcdef"
        assert cfg.trading_mode == TradingMode.DEVELOPMENT
        assert cfg.daily_loss_limit == 0.03
        assert cfg.is_live_trading is False
        
        print("  ✓ 모든 값 정상 로드")
        print(f"  ✓ 거래 모드: {cfg.trading_mode.value}")
        print(f"  ✓ 실거래 차단됨: {not cfg.is_live_trading}")
    finally:
        env_path.unlink()


# ============================================================
# 테스트 2: 필수값 누락 검출
# ============================================================
def test_missing_required():
    print("\n[테스트 2] API 키 누락 시 에러 발생")
    print("-" * 60)
    _clean_env()
    
    env_content = """
TRADING_MODE=DEVELOPMENT
"""
    env_path = _write_temp_env(env_content)
    
    try:
        try:
            get_config(env_file=env_path, force_reload=True)
            raise AssertionError("필수값 누락인데 에러가 안 났습니다!")
        except ConfigError as e:
            print(f"  ✓ 예상대로 ConfigError 발생: {e}")
    finally:
        env_path.unlink()


# ============================================================
# 테스트 3: 템플릿 값 검출 (your_api_key_here 같은 미수정 상태)
# ============================================================
def test_template_value_detection():
    print("\n[테스트 3] .env.example 그대로 둔 경우 검출")
    print("-" * 60)
    _clean_env()
    
    env_content = """
BITHUMB_API_KEY=your_api_key_here
BITHUMB_SECRET_KEY=your_secret_key_here
"""
    env_path = _write_temp_env(env_content)
    
    try:
        try:
            get_config(env_file=env_path, force_reload=True)
            raise AssertionError("템플릿 값인데 통과됨!")
        except ConfigError as e:
            print(f"  ✓ 템플릿 값 감지됨: {e}")
    finally:
        env_path.unlink()


# ============================================================
# 테스트 4: 위험 한도 일관성 검증
# ============================================================
def test_risk_limit_consistency():
    print("\n[테스트 4] 위험 한도 일관성 검증 (일일 < 주간 < MDD)")
    print("-" * 60)
    _clean_env()
    
    # 일일 한도가 주간 한도보다 크면 에러
    env_content = """
BITHUMB_API_KEY=test_api_key_1234567890abcdef
BITHUMB_SECRET_KEY=test_secret_key_1234567890abcdef
DAILY_LOSS_LIMIT=0.10
WEEKLY_LOSS_LIMIT=0.05
MAX_DRAWDOWN_LIMIT=0.15
"""
    env_path = _write_temp_env(env_content)
    
    try:
        try:
            get_config(env_file=env_path, force_reload=True)
            raise AssertionError("일관성 오류인데 통과됨!")
        except ConfigError as e:
            print(f"  ✓ 한도 일관성 오류 감지: {e}")
    finally:
        env_path.unlink()


# ============================================================
# 테스트 5: 범위 초과 검출
# ============================================================
def test_value_range():
    print("\n[테스트 5] 비정상 범위 값 검출")
    print("-" * 60)
    _clean_env()
    
    # 일일 손실 한도 50% — 비현실적으로 큰 값
    env_content = """
BITHUMB_API_KEY=test_api_key_1234567890abcdef
BITHUMB_SECRET_KEY=test_secret_key_1234567890abcdef
DAILY_LOSS_LIMIT=0.50
"""
    env_path = _write_temp_env(env_content)
    
    try:
        try:
            get_config(env_file=env_path, force_reload=True)
            raise AssertionError("범위 초과인데 통과됨!")
        except ConfigError as e:
            print(f"  ✓ 범위 초과 감지: {e}")
    finally:
        env_path.unlink()


# ============================================================
# 테스트 6: 잘못된 TRADING_MODE 검출
# ============================================================
def test_invalid_trading_mode():
    print("\n[테스트 6] 잘못된 거래 모드 검출")
    print("-" * 60)
    _clean_env()
    
    env_content = """
BITHUMB_API_KEY=test_api_key_1234567890abcdef
BITHUMB_SECRET_KEY=test_secret_key_1234567890abcdef
TRADING_MODE=YOLO
"""
    env_path = _write_temp_env(env_content)
    
    try:
        try:
            get_config(env_file=env_path, force_reload=True)
            raise AssertionError("잘못된 모드인데 통과됨!")
        except ConfigError as e:
            print(f"  ✓ 잘못된 모드 감지: {e}")
    finally:
        env_path.unlink()


# ============================================================
# 테스트 7: API 키가 repr 에 노출되지 않음 (보안 핵심)
# ============================================================
def test_secret_not_exposed():
    print("\n[테스트 7] API 키가 repr/문자열 표현에 노출되지 않음 ← 보안 핵심")
    print("-" * 60)
    _clean_env()
    
    secret_value = "VERY_SECRET_KEY_DO_NOT_LEAK_1234567890"
    env_content = f"""
BITHUMB_API_KEY=test_api_key_1234567890abcdef
BITHUMB_SECRET_KEY={secret_value}
TELEGRAM_BOT_TOKEN=ALSO_SECRET_TOKEN_1234567890
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1234567890/VERY_SECRET_DISCORD_WEBHOOK_TOKEN
"""
    env_path = _write_temp_env(env_content)
    
    try:
        cfg = get_config(env_file=env_path, force_reload=True)
        
        # repr() 호출 시 시크릿이 노출되면 안 됨
        cfg_repr = repr(cfg)
        if secret_value in cfg_repr:
            raise AssertionError(f"🚨 보안 사고: repr()에 시크릿 노출됨!\n{cfg_repr}")
        print(f"  ✓ repr() 에 시크릿 노출 안 됨")
        
        # summary() 도 마스킹되어야 함
        summary = cfg.summary()
        if secret_value in summary:
            raise AssertionError(f"🚨 보안 사고: summary() 에 시크릿 노출됨!\n{summary}")
        if "VERY_SECRET_DISCORD_WEBHOOK_TOKEN" in summary:
            raise AssertionError(f"🚨 보안 사고: summary() 에 Discord webhook 노출됨!\n{summary}")
        print(f"  ✓ summary() 에 시크릿 노출 안 됨 (마스킹됨)")
        
        # 마스킹 확인
        masked = cfg.mask_secret(secret_value)
        assert "****" in masked
        assert secret_value not in masked
        print(f"  ✓ 마스킹 동작 확인: '{secret_value[:8]}...' → '{masked}'")
        
    finally:
        env_path.unlink()


# ============================================================
# 테스트 8: .env 파서 강건성
# ============================================================
def test_env_parser_robustness():
    print("\n[테스트 8] .env 파일 파서 강건성")
    print("-" * 60)
    _clean_env()
    
    env_content = '''
# 이것은 주석입니다
BITHUMB_API_KEY=test_api_key_1234567890abcdef

# 빈 줄 위쪽 무시
BITHUMB_SECRET_KEY="test_secret_with_quotes_1234567890"
TRADING_MODE='DEVELOPMENT'
   # 들여쓰기된 주석
LOG_LEVEL=DEBUG
'''
    env_path = _write_temp_env(env_content)
    
    try:
        cfg = get_config(env_file=env_path, force_reload=True)
        
        # 따옴표 제거 확인
        assert cfg.bithumb_secret_key == "test_secret_with_quotes_1234567890"
        assert cfg.trading_mode == TradingMode.DEVELOPMENT
        assert cfg.log_level == "DEBUG"
        
        print("  ✓ 주석 무시")
        print("  ✓ 따옴표 제거 (큰따옴표, 작은따옴표)")
        print("  ✓ 빈 줄 처리")
    finally:
        env_path.unlink()


# ============================================================
# 테스트 9: LIVE 모드 시 경고 표시
# ============================================================
def test_live_mode_warning():
    print("\n[테스트 9] LIVE 모드 시 경고 표시 (실거래 활성화 경고)")
    print("-" * 60)
    _clean_env()
    
    env_content = """
BITHUMB_API_KEY=test_api_key_1234567890abcdef
BITHUMB_SECRET_KEY=test_secret_key_1234567890abcdef
TRADING_MODE=LIVE
"""
    env_path = _write_temp_env(env_content)
    
    try:
        cfg = get_config(env_file=env_path, force_reload=True)
        
        assert cfg.is_live_trading is True
        
        summary = cfg.summary()
        assert "⚠️" in summary or "경고" in summary or "활성화" in summary
        print("  ✓ LIVE 모드 감지")
        print("  ✓ summary() 에 경고 메시지 포함")
    finally:
        env_path.unlink()


# ============================================================
# 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print("설정 모듈 검증 테스트")
    print("=" * 60)
    
    tests = [
        test_basic_loading,
        test_missing_required,
        test_template_value_detection,
        test_risk_limit_consistency,
        test_value_range,
        test_invalid_trading_mode,
        test_secret_not_exposed,
        test_env_parser_robustness,
        test_live_mode_warning,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 실패: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 예외 발생: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과 / {failed}개 실패")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    
    # 정상 설정으로 최종 데모
    print("\n[데모] 정상 로딩 시 summary() 출력 예시")
    _clean_env()
    env_content = """
BITHUMB_API_KEY=demo_api_key_abcdef1234567890ZZZZ
BITHUMB_SECRET_KEY=demo_secret_key_xyz9876543210AAAA
TRADING_MODE=DEVELOPMENT
DAILY_LOSS_LIMIT=0.03
WEEKLY_LOSS_LIMIT=0.07
MAX_DRAWDOWN_LIMIT=0.15
MAX_POSITION_PER_COIN=0.25
MAX_TOTAL_CRYPTO=0.80
INITIAL_CAPITAL=10000000
"""
    env_path = _write_temp_env(env_content)
    try:
        cfg = get_config(env_file=env_path, force_reload=True)
        print(cfg.summary())
    finally:
        env_path.unlink()


if __name__ == '__main__':
    main()
