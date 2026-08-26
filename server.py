"""참가자 정책 서버 골격.

BasePolicy를 상속해 infer()만 채우고 serve_policy()로 띄우면 된다.
평가 서버가 이 서버에 WebSocket 클라이언트로 접속한다 (아웃바운드 연결은 평가 서버 쪽에서만).

프로토콜 (msgpack, numpy 배열은 msgpack_numpy ext):
    수신 {"type":"reset", "episode_id","task_id","instruction","conf","server_info"}
                                     -> 송신 {"type":"ready"}
    수신 {"type":"observation", "sim_time","images","state","scan"(선택),"instruction"}
                                     -> 송신 {"type":"action", "actions": (T, action_dim) float32
                                              또는 {"joint_q","lift","mobile"} 구조화 dict}
    수신 {"type":"done", "reason": ..., "server_info": ...}  -> 응답 없이 연결 종료

**모르는 키·모르는 type은 전부 무시한다.** 평가 서버가 나중에 필드를 늘려도 이 서버가
죽지 않게 하는 규칙이다 (아는 것만 읽는다).

관측 이미지는 **JPEG 바이트**로 온다 (실기 기록 파이프라인이 CompressedImage 전용이라
평가 관측도 같은 형식이다). 이 골격이 **캠별 지연 디코드**로 넘기므로 정책이 실제로 꺼낸
카메라만 numpy (H, W, 3) uint8로 디코드된다 — 안 쓰는 캠은 비용이 0이다.
무압축 배열로 오면 그대로 통과한다.

의존성: pip install websockets msgpack msgpack-numpy numpy pillow
        (선택) pip install simplejpeg — 설치돼 있으면 JPEG 디코드에 자동으로 쓴다 (2–3× 빠름)
"""

import argparse
import secrets
import ssl
from io import BytesIO

import msgpack
import msgpack_numpy
import numpy as np
from websockets.sync.server import serve

# 통합 액션 벡터의 구간 이름 (평가 서버 스펙 §2). fill_action()으로 일부 구간만 채울 때 쓴다.
# 태스크별로 허용되지 않는 자유도는 **평가 서버가 0으로 마스킹**한다 (보내도 무시된다).
ACTION_SLICES = {
    "arm_r": slice(0, 7), "arm_l": slice(7, 14),
    "gripper_r": slice(14, 15), "gripper_l": slice(15, 16),
    "head": slice(16, 18), "lift": slice(18, 19), "base": slice(19, 22),
}

# 통합 IO v2 구조화 액션 그룹 (관측 state와 같은 이름·같은 순서). action_groups() 참조.
ACTION_GROUP_DIMS = {"joint_q": 18, "lift": 1, "mobile": 3}

try:  # simplejpeg이 있으면 자동으로 쓴다 (없으면 PIL 폴백 — 순수 선택 의존성)
    import simplejpeg

    def decode_jpeg(buf) -> np.ndarray:
        return simplejpeg.decode_jpeg(bytes(buf), colorspace="RGB")
except ImportError:
    def decode_jpeg(buf) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(BytesIO(buf)).convert("RGB"))


def pack(msg: dict) -> bytes:
    return msgpack.packb(msg, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=decode_ndarray, raw=False)


# 네트워크에서 온 바이트를 msgpack_numpy.decode로 풀면 kind=b'O' 맵이 pickle.loads를 타서
# 송신자가 임의 코드를 실행시킬 수 있다. 숫자·불리언 dtype·ndim<=2·요소 수 상한만 허용.
ARRAY_MAX_ELEMS = 1 << 20


def decode_ndarray(obj: dict):
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


class LazyImages(dict):
    """관측 이미지 dict. **꺼낸 카메라만** JPEG 디코드한다 (디코드 결과는 캐시된다).

    `obs["images"]["head_l"]` -> numpy (H, W, 3) uint8. 정책이 손목 캠을 안 쓰면 그 캠은
    디코드 비용이 아예 들지 않는다. bytes가 아니면(무압축 전송) 그대로 통과시킨다.
    """

    def __getitem__(self, cam):
        img = super().__getitem__(cam)
        if isinstance(img, (bytes, bytearray)):
            img = decode_jpeg(img)
            super().__setitem__(cam, img)
        return img

    def get(self, cam, default=None):
        return self[cam] if cam in self else default

    def values(self):
        return (self[cam] for cam in self)

    def items(self):
        return ((cam, self[cam]) for cam in self)


def action_groups(joint_q=None, lift=None, mobile=None) -> dict:
    """v2 구조화 액션 `{"joint_q": (T,18), "lift": (T,1), "mobile": (T,3)}`을 만든다.

    쓰는 그룹만 주면 되고 **누락 그룹은 평가 서버가 0으로 채운다**. joint_q 순서는 관측
    `state["joint_q"]`와 같다 (arm_r7, arm_l7, grip_r, grip_l, head2 — 그리퍼는 연속 각도).
    """
    out = {}
    for key, value in (("joint_q", joint_q), ("lift", lift), ("mobile", mobile)):
        if value is None:
            continue
        a = np.asarray(value, np.float32)
        out[key] = (a.reshape(-1, 1) if ACTION_GROUP_DIMS[key] == 1 and a.ndim < 2
                    else np.atleast_2d(a))
    return out


def log_server_info(msg: dict) -> None:
    """reset/done에 실려 오는 server_info를 한 줄로 찍는다 (type은 Start와 Done 둘뿐).

    info는 `"Task Z-N"` 또는 `"Task Z-N | 추가 설명"` — 1회 분리 후 양쪽 trim이 규격이라
    설명 안에 `|`가 또 나와도 안전하다. 추가 설명은 자연어이므로 내용에 기대면 안 된다.

    **모르는 키·모르는 type은 그냥 무시한다** — 평가 서버가 나중에 필드를 늘려도 이 서버가
    죽지 않게 하는 규칙이다.
    """
    info = msg.get("server_info")
    if not isinstance(info, dict) or info.get("type") not in ("Start", "Done"):
        return
    task, _, note = str(info.get("info") or "").partition("|")
    print(f"[{info['type']}] {task.strip()}" + (f" | {note.strip()}" if note.strip() else ""),
          flush=True)


def fill_action(action, action_dim: int | None = None, chunk_len: int = 1) -> np.ndarray:
    """부분 지정 액션을 (T, action_dim) float32 청크로 0 채워 완성한다.

    - v2 구조화 dict (`action_groups()`의 결과): 그대로 통과한다 — 22-dim 조립과 zero-fill은
      평가 서버 몫이다.
    - dict: 쓰고 싶은 구간만 준다 -> `{"arm_r": q7, "gripper_r": -1.0}`.
      키는 ACTION_SLICES 이름 또는 인덱스/슬라이스. 나머지 자유도는 전부 0이다.
    - 짧은 벡터/청크: 뒤를 0으로 패딩. 1차원이면 (1, action_dim)으로 본다.
    - 이미 (T, action_dim)이면 그대로 통과한다 (기존 방식 그대로 써도 된다).
    """
    if isinstance(action, dict):
        if action and not action.keys() - ACTION_GROUP_DIMS.keys():
            return action
        if action_dim is None:
            raise ValueError("dict 액션에는 action_dim이 필요하다 (reset의 conf에 실려 온다)")
        out = np.zeros((chunk_len, action_dim), np.float32)
        for key, value in action.items():
            out[:, ACTION_SLICES[key] if isinstance(key, str) else key] = value
        return out
    a = np.atleast_2d(np.asarray(action, dtype=np.float32))
    if action_dim and a.shape[1] < action_dim:
        a = np.pad(a, ((0, 0), (0, action_dim - a.shape[1])))
    return a


class BasePolicy:
    def reset(self, msg: dict) -> None:
        """에피소드 시작. msg["conf"]에 action_dim / control_hz / chunk_mode 등이 들어온다."""

    def infer(self, obs: dict) -> np.ndarray:
        """obs = {"sim_time": float,
                  "images": {"head_l": (376,672,3), "wrist_l"/"wrist_r": (240,424,3) uint8},
                  "state": {"joint_q": [18], "lift": [1], "mobile": [3]} float32,
                  "scan": float32[960],   # LiDAR가 있는 태스크만. 없으면 키 자체가 없다
                  "instruction": str}

        이미지는 꺼내는 순간 JPEG 디코드가 끝난 numpy 배열이다 (해상도는 실기 SDK 기록값
        고정 — head_l 672×376, 손목 424×240. 크롭/리사이즈 등 전처리는 정책 몫이다).

        반환: (T, action_dim) float32 액션 청크. T >= 1.
        평가 서버 conf의 max_chunk_len을 넘으면 앞에서부터 잘린다.
        일부 자유도만 쓰려면 dict나 짧은 벡터를 돌려줘도 된다 — fill_action()이 0을 채운다.
        v2 구조화 액션은 action_groups(joint_q=..., lift=..., mobile=...)로 만들면 된다.
        """
        raise NotImplementedError


def serve_policy(policy: BasePolicy, host: str = "0.0.0.0", port: int = 8000,
                 token: str | None = None, certfile: str | None = None,
                 keyfile: str | None = None) -> None:
    """token을 주면 평가 서버의 `Authorization: Bearer <token>` 헤더를 검증한다.

    홈페이지에서 발급받은 제출 토큰을 그대로 --token 으로 넘기면 된다.
    이게 없으면 주소를 아는 누구나 접속해 모델 출력을 뽑아 갈 수 있다.

    certfile/keyfile을 주면 wss:// (TLS)로 연다. self-signed 인증서를 쓸 경우
    그 지문(SHA-256)을 홈페이지에 등록해야 평가 서버가 접속한다 — README 참조.
    """

    def check_auth(conn, request):
        if token and not secrets.compare_digest(
                request.headers.get("Authorization") or "", f"Bearer {token}"):
            return conn.respond(401, "unauthorized\n")
        return None

    def handler(ws):
        action_dim = None
        for raw in ws:
            msg = unpack(raw)
            # 아는 키만 읽는다 — msg에 모르는 키가 더 있어도 그냥 놔둔다.
            kind = msg.get("type")
            if kind == "reset":
                action_dim = (msg.get("conf") or {}).get("action_dim")
                log_server_info(msg)          # [Start] Task Z-N
                policy.reset(msg)
                ws.send(pack({"type": "ready"}))
            elif kind == "observation":
                # JPEG -> numpy는 꺼낼 때. 정책은 항상 배열만 본다 (안 꺼낸 캠은 디코드 생략).
                msg["images"] = LazyImages(msg.get("images") or {})
                # 완전한 (T, action_dim) 배열은 그대로, 부분 지정(dict/짧은 벡터)은 0으로 채운다.
                actions = fill_action(policy.infer(msg), action_dim)
                ws.send(pack({"type": "action", "actions": actions}))
            elif kind == "done":
                log_server_info(msg)          # [Done] Task Z-N | 추가 설명
                return
            # 그 밖의 type은 **무시한다** — 평가 서버가 나중에 메시지를 늘려도 안 죽는다.

    ctx = None
    if certfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)

    # max_size=None: 이미지 관측이 websockets 기본 1MB 프레임 한도를 넘을 수 있음.
    with serve(handler, host, port, max_size=None, process_request=check_auth, ssl=ctx) as server:
        print(f"policy server listening on {'wss' if ctx else 'ws'}://{host}:{port}", flush=True)
        server.serve_forever()


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--chunk-len", type=int, default=10, help="한 번에 반환할 액션 청크 길이")
    p.add_argument("--token", default=None,
                   help="홈페이지에서 발급받은 제출 토큰. 주면 접속 인증을 요구한다 (권장).")
    p.add_argument("--certfile", default=None,
                   help="TLS 인증서(PEM). 주면 wss://로 연다. self-signed면 지문을 홈페이지에 등록.")
    p.add_argument("--keyfile", default=None, help="TLS 개인키(PEM). --certfile과 함께.")
    return p.parse_args()
