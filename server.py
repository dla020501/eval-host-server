"""참가자 정책 서버 골격.

BasePolicy를 상속해 infer()만 채우고 serve_policy()로 띄우면 된다.
평가 서버가 이 서버에 WebSocket 클라이언트로 접속한다 (아웃바운드 연결은 평가 서버 쪽에서만).

프로토콜 (msgpack, numpy 배열은 msgpack_numpy ext):
    수신 {"type":"reset", "episode_id","task_id","instruction","conf"} -> 송신 {"type":"ready"}
    수신 {"type":"observation", "sim_time","images","state","instruction"}
                                     -> 송신 {"type":"action", "actions": (T, action_dim) float32}
    수신 {"type":"done"}  -> 응답 없이 연결 종료

의존성: pip install websockets msgpack msgpack-numpy numpy
"""

import argparse
import secrets
import ssl

import msgpack
import msgpack_numpy
import numpy as np
from websockets.sync.server import serve


def pack(msg: dict) -> bytes:
    return msgpack.packb(msg, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


class BasePolicy:
    def reset(self, msg: dict) -> None:
        """에피소드 시작. msg["conf"]에 action_dim / control_hz / chunk_mode 등이 들어온다."""

    def infer(self, obs: dict) -> np.ndarray:
        """obs = {"sim_time": float, "images": {cam: (H,W,3) uint8},
                  "state": (D,) float32, "instruction": str}

        반환: (T, action_dim) float32 액션 청크. T >= 1.
        평가 서버 conf의 max_chunk_len을 넘으면 앞에서부터 잘린다.
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
        for raw in ws:
            msg = unpack(raw)
            kind = msg.get("type")
            if kind == "reset":
                policy.reset(msg)
                ws.send(pack({"type": "ready"}))
            elif kind == "observation":
                actions = np.asarray(policy.infer(msg), dtype=np.float32)
                ws.send(pack({"type": "action", "actions": actions}))
            elif kind == "done":
                return

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
