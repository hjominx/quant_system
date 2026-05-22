"""
빗썸 API 2.0 JWT 인증 헬퍼

빗썸 API 2.0의 인증 방식:
1. 매 요청마다 JWT 토큰 생성 (nonce는 매번 다른 UUID)
2. 인증 필요한 요청은 Authorization: Bearer {token} 헤더
3. 쿼리 파라미터가 있으면 SHA512 해시를 payload에 포함 (위변조 방지)

설계 원칙:
- 토큰 캐싱 절대 금지 (nonce 재사용 = replay attack 위험)
- 시크릿 키는 인스턴스 내부에만 저장, 외부 노출 차단
- 모든 함수는 순수 함수처럼 동작 (사이드 이펙트 없음)
- 입력 검증 엄격 (잘못된 키 형식이면 즉시 에러)
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Dict, Optional, Union
from urllib.parse import urlencode

import jwt


class AuthError(Exception):
    """인증 관련 에러 - 키가 잘못되었거나 서명 실패"""
    pass


class JWTAuthenticator:
    """빗썸 API 2.0 JWT 토큰 생성기
    
    사용법:
        auth = JWTAuthenticator(api_key, secret_key)
        
        # 파라미터 없는 요청 (잔고 조회 등)
        token = auth.make_token()
        
        # 파라미터 있는 요청 (주문 등)
        token = auth.make_token(params={'market': 'KRW-BTC', 'side': 'bid'})
        
        # 헤더로 바로 받기
        headers = auth.make_auth_header(params)
    """
    
    # JWT 서명 알고리즘 - 빗썸 공식 사양
    ALGORITHM = 'HS256'
    
    # 해시 알고리즘 - 빗썸 공식 사양 (파라미터 해시용)
    HASH_ALGORITHM = 'SHA512'
    
    def __init__(self, api_key: str, secret_key: str):
        """
        Args:
            api_key: 빗썸 API Access Key
            secret_key: 빗썸 API Secret Key
        
        Raises:
            AuthError: 키 형식이 명백히 잘못된 경우
        """
        self._validate_key(api_key, 'API Key')
        self._validate_key(secret_key, 'Secret Key')
        
        # 시크릿 키는 _ 로 시작하는 private 속성
        # repr 시 자동 노출 방지를 위해 별도 처리
        self._api_key = api_key
        self._secret_key = secret_key
    
    @staticmethod
    def _validate_key(key: str, name: str) -> None:
        """키 형식 기본 검증
        
        실제 키 유효성은 빗썸 서버만 판단 가능.
        여기서는 명백한 오류(빈 문자열, 너무 짧음, 템플릿 값)만 차단.
        """
        if not isinstance(key, str):
            raise AuthError(f"{name}는 문자열이어야 합니다 (받은 타입: {type(key).__name__})")
        if not key:
            raise AuthError(f"{name}가 비어있습니다")
        if len(key) < 10:
            raise AuthError(f"{name}가 너무 짧습니다 (길이 {len(key)}). 잘못 입력하셨을 가능성이 높습니다")
        if key.startswith('your_') or key == 'test':
            raise AuthError(f"{name}가 템플릿 값입니다. 실제 키로 교체하세요")
    
    def make_token(self, params: Optional[Dict[str, Union[str, int, float]]] = None) -> str:
        """JWT 토큰 생성 (raw token, Bearer prefix 없음)
        
        Args:
            params: 요청 파라미터 (있을 경우 SHA512 해시를 payload에 포함)
        
        Returns:
            JWT 토큰 문자열
        
        Note:
            - nonce는 매번 새로운 UUID 생성 (replay attack 방지)
            - timestamp는 밀리초 단위 (빗썸 사양)
        """
        payload = {
            'access_key': self._api_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': int(time.time() * 1000),
        }
        
        if params:
            # 파라미터를 URL 인코딩 후 SHA512 해시 생성
            # 이렇게 하면 중간에 누가 파라미터를 바꿔도 서버에서 감지 가능
            query_string = urlencode(params, doseq=True)
            query_hash = hashlib.sha512(query_string.encode('utf-8')).hexdigest()
            
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = self.HASH_ALGORITHM
        
        try:
            token = jwt.encode(payload, self._secret_key, algorithm=self.ALGORITHM)
        except Exception as e:
            # PyJWT 에러를 우리 에러 타입으로 변환
            raise AuthError(f"JWT 생성 실패: {e}") from e
        
        # PyJWT 1.x는 bytes를 반환했고 2.x부터는 str를 반환
        # 호환성을 위해 항상 str로 정규화
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        
        return token
    
    def make_auth_header(self, params: Optional[Dict] = None) -> Dict[str, str]:
        """Authorization 헤더 dict 생성 (HTTP 클라이언트에 바로 전달 가능)
        
        Args:
            params: 요청 파라미터
        
        Returns:
            {'Authorization': 'Bearer xxx'} 형식의 dict
        """
        token = self.make_token(params)
        return {'Authorization': f'Bearer {token}'}
    
    def __repr__(self) -> str:
        """repr 시 시크릿 노출 방지 - 마스킹된 형태로만 표시"""
        masked_key = f"{self._api_key[:4]}****{self._api_key[-4:]}" if len(self._api_key) > 8 else "****"
        return f"<JWTAuthenticator api_key={masked_key} secret_key=****>"
    
    def __str__(self) -> str:
        return self.__repr__()


def make_token(api_key: str, secret_key: str, params: Optional[Dict] = None) -> str:
    """일회성 토큰 생성 (간단한 사용을 위한 함수형 인터페이스)
    
    매번 객체를 만드는 게 부담스러우면 이 함수를 직접 호출.
    내부적으로는 JWTAuthenticator를 사용.
    
    Args:
        api_key: 빗썸 API Access Key
        secret_key: 빗썸 API Secret Key
        params: 요청 파라미터
    
    Returns:
        JWT 토큰 문자열
    """
    authenticator = JWTAuthenticator(api_key, secret_key)
    return authenticator.make_token(params)
