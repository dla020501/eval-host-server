"""신뢰 경계 회귀 테스트. 여기가 깨지면 송신자가 임의 코드를 실행시킬 수 있다."""

import pickle

import msgpack
import msgpack_numpy
import numpy as np
import pytest

from evalhost.images import decode_depth
from evalhost.protocol import ARRAY_MAX_ELEMS, decode_ndarray, pack, unpack


def test_roundtrip_preserves_shape_and_dtype():
    msg = {"type": "action", "actions": np.arange(66, dtype=np.float32).reshape(3, 22),
           "scan": np.zeros(960, np.float32), "n": 3, "s": "ok"}
    out = unpack(pack(msg))
    assert out["type"] == "action"
    assert out["actions"].shape == (3, 22)
    assert out["actions"].dtype == np.float32
    assert np.array_equal(out["actions"], msg["actions"])
    assert out["scan"].shape == (960,)
    assert out["n"] == 3 and out["s"] == "ok"


def test_roundtrip_image_shaped_uint8():
    img = np.random.randint(0, 255, (16, 24, 3), np.uint8)
    out = unpack(pack({"images": {"head_l": img}}))
    assert np.array_equal(out["images"]["head_l"], img)


def test_pickle_object_array_rejected():
    """kind=b'O' 맵은 msgpack_numpy.decode에서 pickle.loads를 탄다. 반드시 거부한다."""
    wire = {b"nd": True, b"type": b"|O", b"kind": b"O", b"shape": (1,),
            b"data": pickle.dumps(["arbitrary"])}
    with pytest.raises(ValueError):
        decode_ndarray(wire)
    with pytest.raises(ValueError):
        unpack(msgpack.packb(wire, use_bin_type=True))


def test_structured_dtype_rejected():
    arr = np.zeros(2, dtype=[("a", "<i4"), ("b", "<f8")])
    with pytest.raises(ValueError):
        unpack(msgpack.packb(msgpack_numpy.encode(arr), use_bin_type=True))


def test_non_numeric_dtype_rejected():
    wire = {b"nd": True, b"type": b"<U4", b"shape": (1,), b"data": b"\x00" * 16}
    with pytest.raises(ValueError):
        decode_ndarray(wire)


def test_oversize_array_rejected():
    wire = {b"nd": True, b"type": b"|u1", b"shape": (ARRAY_MAX_ELEMS + 1,),
            b"data": b"\x00" * (ARRAY_MAX_ELEMS + 1)}
    with pytest.raises(ValueError):
        decode_ndarray(wire)


def test_bad_shape_rejected():
    wire = {b"nd": True, b"type": b"|u1", b"shape": (2, 2, 2, 2), b"data": b"\x00" * 16}
    with pytest.raises(ValueError):
        decode_ndarray(wire)


def test_non_array_map_passes_through():
    assert decode_ndarray({b"type": "reset"}) == {b"type": "reset"}


def test_decode_depth_mm_png_to_meters():
    from io import BytesIO

    from PIL import Image

    mm = np.array([[0, 1000], [2500, 65535]], np.uint16)
    buf = BytesIO()
    Image.fromarray(mm).save(buf, "PNG")
    depth = decode_depth(buf.getvalue())
    assert depth.dtype == np.float32
    assert depth[0, 0] == 0.0                      # 0은 무효값 그대로
    assert depth[0, 1] == pytest.approx(1.0)
    assert depth[1, 0] == pytest.approx(2.5)


def test_decode_depth_passes_arrays_through():
    a = np.zeros((2, 2), np.float32)
    assert decode_depth(a) is a
