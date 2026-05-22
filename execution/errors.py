"""
빗썸 API 에러 분류 모듈

에러를 적절히 분류하지 않으면:
- 재시도해야 할 일시적 에러를 영구 실패로 처리 → 거래 기회 상실
- 영구 에러를 계속 재시도 → 빗썸 서버에 부담, 레이트 리밋 차단
- 인증 오류를 일반 에러로 처리 → 키 만료를 못 알아챔

설계 원칙:
1. 모든 에러는 BithumbError 를 상속 (한 곳에서 catch 가능)
2. is_retryable 속성으로 재시도 여부 명확히 표시
3. 에러 메시지에 빗썸 응답 원본 포함 (디버깅용)
4. 시크릿 등 민감정보는 에러에 포함 안 함
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class BithumbError(Exception):
    """모든 빗썸 관련 에러의 베이스 클래스"""
    is_retryable: bool = False  # 기본값: 재시도 안 함
    
    def __init__(
        self,
        message: str,
        response_data: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.response_data = response_data
        self.status_code = status_code
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"(HTTP {self.status_code})")
        if self.response_data:
            data_str = str(self.response_data)[:200]
            parts.append(f"응답: {data_str}")
        return " ".join(parts)


# ============================================================
# 재시도 가능한 에러 (일시적 문제)
# ============================================================

class RetryableError(BithumbError):
    """재시도하면 성공할 가능성이 있는 에러"""
    is_retryable = True


class NetworkError(RetryableError):
    """네트워크 연결 오류 (타임아웃, DNS 실패, 연결 끊김 등)"""
    pass


class ServerError(RetryableError):
    """빗썸 서버 측 오류 (5xx HTTP 상태 코드)"""
    pass


class RateLimitError(RetryableError):
    """레이트 리밋 초과 (429 HTTP 상태 코드)
    
    빗썸 사양: Public 150/sec, Private 140/sec, Order 10/sec
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after_seconds = retry_after_seconds


# ============================================================
# 영구 실패 에러 (재시도 무의미)
# ============================================================

class PermanentError(BithumbError):
    """재시도해도 같은 결과가 나오는 에러"""
    is_retryable = False


class AuthenticationError(PermanentError):
    """인증 실패 - 🚨 즉시 운영 중단하고 키 점검 필요"""
    pass


class InvalidRequestError(PermanentError):
    """잘못된 요청 (파라미터 오류, 형식 오류 등)"""
    pass


class InsufficientFundsError(InvalidRequestError):
    """잔고 부족"""
    pass


class OrderNotFoundError(InvalidRequestError):
    """존재하지 않는 주문 조회/취소 시도"""
    pass


class MarketClosedError(PermanentError):
    """거래 정지/점검 중인 종목"""
    pass


# ============================================================
# 빗썸 응답 파싱 및 에러 분류
# ============================================================

def classify_http_error(
    status_code: int,
    response_data: Optional[Dict[str, Any]] = None,
    response_text: Optional[str] = None,
) -> BithumbError:
    """HTTP 상태 코드와 응답을 보고 적절한 에러 객체 생성"""
    error_msg = _extract_error_message(response_data, response_text)
    error_name = _extract_error_name(response_data)
    
    # 1. 인증 오류 (최우선)
    if status_code in (401, 403):
        return AuthenticationError(
            f"인증 실패: {error_msg}",
            response_data=response_data,
            status_code=status_code,
        )
    
    # 빗썸은 인증 실패를 400으로 주는 경우도 있음
    if error_name and 'jwt' in error_name.lower():
        return AuthenticationError(
            f"JWT 인증 실패: {error_msg}",
            response_data=response_data,
            status_code=status_code,
        )
    
    # 2. 레이트 리밋
    if status_code == 429:
        return RateLimitError(
            f"요청 한도 초과: {error_msg}",
            response_data=response_data,
            status_code=status_code,
        )
    
    # 3. 서버 에러
    if 500 <= status_code < 600:
        return ServerError(
            f"빗썸 서버 오류: {error_msg}",
            response_data=response_data,
            status_code=status_code,
        )
    
    # 4. 클라이언트 에러 (400대)
    if 400 <= status_code < 500:
        if error_name:
            name_lower = error_name.lower()
            if 'insufficient' in name_lower or 'balance' in name_lower:
                return InsufficientFundsError(
                    f"잔고 부족: {error_msg}",
                    response_data=response_data,
                    status_code=status_code,
                )
            if 'not_found' in name_lower or 'order_not_found' in name_lower:
                return OrderNotFoundError(
                    f"주문 없음: {error_msg}",
                    response_data=response_data,
                    status_code=status_code,
                )
            if 'market' in name_lower and 'closed' in name_lower:
                return MarketClosedError(
                    f"거래 정지: {error_msg}",
                    response_data=response_data,
                    status_code=status_code,
                )
        
        return InvalidRequestError(
            f"잘못된 요청: {error_msg}",
            response_data=response_data,
            status_code=status_code,
        )
    
    # 5. 예상 못 한 상태 코드
    return BithumbError(
        f"예상치 못한 HTTP 응답: {error_msg}",
        response_data=response_data,
        status_code=status_code,
    )


def _extract_error_message(
    response_data: Optional[Dict],
    response_text: Optional[str],
) -> str:
    """빗썸 응답에서 에러 메시지 추출"""
    if response_data:
        # V2 형식: {"error": {"name": "...", "message": "..."}}
        if isinstance(response_data.get('error'), dict):
            msg = response_data['error'].get('message', '')
            if msg:
                return msg
        
        # V1 형식: {"status": "5500", "message": "..."}
        if 'message' in response_data:
            return str(response_data['message'])
        
        if 'error' in response_data:
            return str(response_data['error'])
    
    if response_text:
        return response_text[:200]
    
    return "알 수 없는 오류"


def _extract_error_name(response_data: Optional[Dict]) -> Optional[str]:
    """빗썸 응답에서 에러 코드/이름 추출"""
    if not response_data:
        return None
    
    if isinstance(response_data.get('error'), dict):
        name = response_data['error'].get('name')
        return str(name) if name is not None else None
    
    if 'status' in response_data and response_data['status'] != '0000':
        return f"status_{response_data['status']}"
    
    return None
