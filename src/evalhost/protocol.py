"""평가 서버와의 msgpack 인코딩 계약. 네트워크 바이트를 푸는 유일한 신뢰 경계다.

메시지는 전부 dict이고 numpy 배열은 msgpack_numpy 확장으로 dtype·shape를 유지한다.
"""

import msgpack
import msgpack_numpy
import numpy as np

# 배열 하나의 요소 수 상한. 액션 청크 (T, 22)·scan 960·이미지 (376, 672, 3)에 충분하다.
ARRAY_MAX_ELEMS = 1 << 20


def pack(msg: dict) -> bytes:
    """dict를 msgpack 바이트로 직렬화한다. numpy 배열은 msgpack_numpy 확장으로 나간다.

    Args:
        msg: 직렬화할 메시지 dict.

    Returns:
        msgpack 바이트열.
    """
    return msgpack.packb(msg, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(data: bytes) -> dict:
    """msgpack 바이트를 dict로 역직렬화한다. 배열은 decode_ndarray가 검증한다.

    Args:
        data: 수신한 msgpack 바이트열.

    Returns:
        역직렬화된 메시지 dict.
    """
    return msgpack.unpackb(data, object_hook=decode_ndarray, raw=False)


def decode_ndarray(obj: dict):
    """msgpack_numpy 와이어 형식(`{nd, type, shape, data}`)의 안전한 역변환이다.

    네트워크에서 온 바이트를 msgpack_numpy.decode로 풀면 `kind=b'O'` 맵이 pickle.loads를
    타서 송신자가 임의 코드를 실행시킬 수 있다. 숫자·불리언 dtype, ndim <= 3,
    요소 수 상한만 허용한다.

    Args:
        obj: msgpack 맵 하나. 배열이 아니면 그대로 통과한다.

    Returns:
        numpy 배열, numpy 스칼라, 또는 입력 dict 그대로.

    Raises:
        ValueError: pickle·structured 배열, 허용 밖 dtype, 상한 초과, 형식 오류.
    """
    if b"nd" not in obj:
        return obj
    try:
        if obj.get(b"kind"):  # b'V'(structured)·b'O'(pickle) 거부
            raise ValueError("structured/object ndarray not allowed")
        dt = np.dtype(obj[b"type"])
        if dt.kind not in "biuf":
            raise ValueError(f"ndarray dtype {dt} not allowed")
        data = obj[b"data"]
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("ndarray data must be bytes")
        a = np.frombuffer(data, dt)
        if a.size > ARRAY_MAX_ELEMS:
            raise ValueError(f"ndarray too large: {a.size} elements")
        if obj[b"nd"] is not True:
            return a[0]
        shape = tuple(obj[b"shape"])
        if len(shape) > 3 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"ndarray shape {shape} not allowed")
        return a.reshape(shape)
    except (KeyError, TypeError, IndexError) as e:
        raise ValueError(f"bad ndarray: {e}") from None
