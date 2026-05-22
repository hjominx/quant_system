"""
JWT 인증 헬퍼 검증 테스트

핵심 검증 9가지:
1. 토큰이 정상 생성되는가
2. 매 요청마다 nonce가 다른가 (replay attack 방지)
3. 파라미터 해시가 SHA512로 정확히 생성되는가
4. JWT 서명이 secret_key로 검증 가능한가
5. 위변조 시 해시가 달라지는가
6. 잘못된 키 형식을 거부하는가
7. 시크릿이 repr/str에 노출되지 않는가  ← 보안 핵심
8. 헤더 포맷이 정확한가
9. timestamp가 밀리초 단위인가
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import time
from urllib.parse import urlencode

import jwt

from utils.auth import JWTAuthenticator, AuthError, make_token


# 테스트용 더미 키 (실제 빗썸 키 아님)
TEST_API_KEY = "test_api_key_abcdef1234567890ABCDEF"
TEST_SECRET_KEY = "test_secret_key_xyz9876543210ZYXWVU"


def test_basic_token_generation():
    print("\n[테스트 1] 파라미터 없는 토큰 생성")
    print("-" * 60)
    
    auth = JWTAuthenticator(TEST_API_KEY, TEST_SECRET_KEY)
    token = auth.make_token()
    
    parts = token.split('.')
    assert len(parts) == 3, f"JWT는 3부분이어야 함: {len(parts)}"
    print(f"  ✓ JWT 형식 정상 (3부분 구조)")
    
    decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=['HS256'])
    
    assert decoded['access_key'] == TEST_API_KEY
    assert 'nonce' in decoded
    assert 'timestamp' in decoded
    assert 'query_hash' not in decoded
    
    print(f"  ✓ access_key 정확히 포함됨")
    print(f"  ✓ nonce 생성됨: {decoded['nonce'][:8]}...")
    print(f"  ✓ timestamp 생성됨: {decoded['timestamp']}")
    print(f"  ✓ query_hash 없음 (파라미터 없으니 정상)")


def test_nonce_uniqueness():
    print("\n[테스트 2] Nonce 매번 새로 생성 (replay attack 방지)")
    print("-" * 60)
    
    auth = JWTAuthenticator(TEST_API_KEY, TEST_SECRET_KEY)
    
    nonces = set()
    for i in range(100):
        token = auth.make_token()
        decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=['HS256'])
        nonces.add(decoded['nonce'])
    
    assert len(nonces) == 100, f"nonce 중복 발생: {100 - len(nonces)}개"
    print(f"  ✓ 100개 토큰의 nonce 모두 다름")
    print(f"  ✓ Replay attack 방지 작동")


def test_query_hash_correctness():
    print("\n[테스트 3] 파라미터 SHA512 해시 정확성")
    print("-" * 60)
    
    auth = JWTAuthenticator(TEST_API_KEY, TEST_SECRET_KEY)
    
    params = {
        'market': 'KRW-BTC',
        'side': 'bid',
        'volume': '0.001',
        'price': '80000000',
        'ord_type': 'limit',
    }
    
    token = auth.make_token(params)
    decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=['HS256'])
    
    assert 'query_hash' in decoded, "query_hash 누락"
    assert 'query_hash_alg' in decoded, "query_hash_alg 누락"
    assert decoded['query_hash_alg'] == 'SHA512'
    
    expected_query = urlencode(params, doseq=True)
    expected_hash = hashlib.sha512(expected_query.encode('utf-8')).hexdigest()
    
    assert decoded['query_hash'] == expected_hash, "해시 불일치!"
    
    print(f"  ✓ query_hash 포함됨")
    print(f"  ✓ 알고리즘 SHA512 명시됨")
    print(f"  ✓ 해시값이 수동 계산과 정확히 일치")
    print(f"    예상: {expected_hash[:32]}...")
    print(f"    실제: {decoded['query_hash'][:32]}...")


def test_tampering_detection():
    print("\n[테스트 4] 파라미터 위변조 시 해시 변경 감지")
    print("-" * 60)
    
    auth = JWTAuthenticator(TEST_API_KEY, TEST_SECRET_KEY)
    
    original = {'market': 'KRW-BTC', 'volume': '0.001'}
    token1 = auth.make_token(original)
    hash1 = jwt.decode(token1, TEST_SECRET_KEY, algorithms=['HS256'])['query_hash']
    
    tampered = {'market': 'KRW-BTC', 'volume': '0.002'}
    token2 = auth.make_token(tampered)
    hash2 = jwt.decode(token2, TEST_SECRET_KEY, algorithms=['HS256'])['query_hash']
    
    assert hash1 != hash2, "위변조에도 해시가 같음!"
    
    print(f"  원본 (volume=0.001):   해시 {hash1[:20]}...")
    print(f"  변조 (volume=0.002):   해시 {hash2[:20]}...")
    print(f"  ✓ 미세한 변경도 해시가 완전히 달라짐")
    print(f"  ✓ 위변조 감지 메커니즘 정상 작동")


def test_invalid_key_rejection():
    print("\n[테스트 5] 잘못된 키 형식 거부")
    print("-" * 60)
    
    invalid_keys = [
        ("", "빈 문자열"),
        ("short", "너무 짧음"),
        ("your_api_key_here", "템플릿 값"),
        ("test", "테스트 값"),
        (None, "None 타입"),
        (12345, "숫자 타입"),
    ]
    
    for bad_key, description in invalid_keys:
        try:
            JWTAuthenticator(bad_key, TEST_SECRET_KEY)
            raise AssertionError(f"잘못된 키({description})가 통과됨!")
        except AuthError as e:
            print(f"  ✓ {description:15s} → 거부됨")


def test_secret_not_exposed():
    print("\n[테스트 6] 시크릿이 repr/str에 노출되지 않음 ← 보안 핵심")
    print("-" * 60)
    
    secret = "VERY_SECRET_KEY_DO_NOT_LEAK_1234567890"
    auth = JWTAuthenticator(TEST_API_KEY, secret)
    
    repr_output = repr(auth)
    if secret in repr_output:
        raise AssertionError(f"🚨 보안 사고: repr()에 시크릿 노출!\n{repr_output}")
    print(f"  ✓ repr() 출력: {repr_output}")
    print(f"  ✓ repr()에 시크릿 노출 안 됨")
    
    str_output = str(auth)
    if secret in str_output:
        raise AssertionError(f"🚨 보안 사고: str()에 시크릿 노출!\n{str_output}")
    print(f"  ✓ str()에 시크릿 노출 안 됨")


def test_auth_header_format():
    print("\n[테스트 7] Authorization 헤더 포맷")
    print("-" * 60)
    
    auth = JWTAuthenticator(TEST_API_KEY, TEST_SECRET_KEY)
    
    headers = auth.make_auth_header()
    assert 'Authorization' in headers
    assert headers['Authorization'].startswith('Bearer '), \
        f"Bearer prefix 누락"
    print(f"  ✓ Authorization 헤더 존재")
    print(f"  ✓ Bearer prefix 정확함")
    
    token = headers['Authorization'].replace('Bearer ', '')
    decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=['HS256'])
    assert decoded['access_key'] == TEST_API_KEY
    print(f"  ✓ 헤더에서 토큰 추출 및 디코드 가능")


def test_functional_interface():
    print("\n[테스트 8] 함수형 인터페이스")
    print("-" * 60)
    
    token = make_token(TEST_API_KEY, TEST_SECRET_KEY, {'market': 'KRW-BTC'})
    decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=['HS256'])
    
    assert decoded['access_key'] == TEST_API_KEY
    assert 'query_hash' in decoded
    print(f"  ✓ 함수형 호출도 정상 동작")


def test_timestamp_accuracy():
    print("\n[테스트 9] Timestamp 정확성 (밀리초 단위)")
    print("-" * 60)
    
    auth = JWTAuthenticator(TEST_API_KEY, TEST_SECRET_KEY)
    
    before = int(time.time() * 1000)
    token = auth.make_token()
    after = int(time.time() * 1000)
    
    decoded = jwt.decode(token, TEST_SECRET_KEY, algorithms=['HS256'])
    ts = decoded['timestamp']
    
    assert before <= ts <= after, f"timestamp 범위 오류"
    
    print(f"  요청 전 시각: {before}")
    print(f"  토큰 timestamp: {ts}")
    print(f"  요청 후 시각: {after}")
    print(f"  ✓ timestamp가 밀리초 단위로 정확히 생성됨")


def main():
    print("=" * 60)
    print("JWT 인증 헬퍼 검증 테스트")
    print("=" * 60)
    
    tests = [
        test_basic_token_generation,
        test_nonce_uniqueness,
        test_query_hash_correctness,
        test_tampering_detection,
        test_invalid_key_rejection,
        test_secret_not_exposed,
        test_auth_header_format,
        test_functional_interface,
        test_timestamp_accuracy,
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
            print(f"  ✗ 예외: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과 / {failed}개 실패 (총 {len(tests)}개)")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
