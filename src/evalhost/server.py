"""WebSocket 정책 서버. 인증(Bearer)·TLS·프레임 크기 등 서버 운영 관심사를 담는다.

평가 서버가 이 서버에 WebSocket 클라이언트로 접속한다 (아웃바운드 연결은 평가 서버만 한다).

프로토콜 (msgpack, numpy 배열은 msgpack_numpy ext):
    수신 {"type":"reset", "episode_id","task_id","instruction","conf","server_info"}
                                     -> 송신 {"type":"ready"}
    수신 {"type":"observation", "sim_time","images","head_l_depth"(선택),"state","scan"(선택),
          "instruction"}
                                     -> 송신 {"type":"action", "actions": (T, action_dim) float32
                                              또는 {"joint_q","lift","mobile"} 구조화 dict}
    수신 {"type":"done", "reason": ..., "server_info": ...}  -> 응답 없이 연결 종료

**모르는 키와 모르는 type은 전부 무시한다.** 평가 서버가 나중에 필드를 늘려도 이 서버의
연결이 끊기지 않게 하는 규칙이다 (아는 것만 읽는다).

관측 이미지는 **JPEG 바이트**로 전송된다 (실기 기록 파이프라인이 CompressedImage 전용이라
평가 관측도 같은 형식이다). 이 템플릿이 **카메라별 지연 디코드**로 넘기므로 정책이 실제로
꺼낸 카메라만 numpy (H, W, 3) uint8로 디코드된다 — 사용하지 않는 카메라는 비용이 0이다.
무압축 배열로 수신하면 그대로 통과한다.
"""

import secrets
import ssl

from websockets.sync.server import serve

from evalhost.actions import fill_action
from evalhost.debug import log_server_info
from evalhost.images import LazyImages, decode_depth
from evalhost.policy import BasePolicy
from evalhost.protocol import pack, unpack


def serve_policy(policy: BasePolicy, host: str = "0.0.0.0", port: int = 8000,
                 token: str | None = None, certfile: str | None = None,
                 keyfile: str | None = None) -> None:
    """정책 서버를 실행한다. 이 호출은 프로세스가 끝날 때까지 반환하지 않는다.

    token을 지정하면 평가 서버의 `Authorization: Bearer <token>` 헤더를 검증한다.
    홈페이지에서 발급받은 제출 토큰을 그대로 --token으로 전달하면 된다.
    이게 없으면 주소를 아는 누구나 접속해 모델 출력을 얻을 수 있다.

    certfile/keyfile을 지정하면 wss:// (TLS)로 연다. self-signed 인증서를 쓰는 경우
    그 지문(SHA-256)을 홈페이지에 등록해야 평가 서버가 접속한다 — README를 참조한다.

    Args:
        policy: BasePolicy 하위 클래스 인스턴스.
        host: 수신 대기 주소.
        port: 수신 대기 포트.
        token: 제출 토큰. 지정하면 Bearer 인증을 요구한다.
        certfile: TLS 인증서 PEM 경로.
        keyfile: TLS 개인키 PEM 경로.
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
            # 아는 키만 읽는다 — msg에 모르는 키가 더 있어도 그대로 둔다.
            kind = msg.get("type")
            if kind == "reset":
                action_dim = (msg.get("conf") or {}).get("action_dim")
                log_server_info(msg)          # [Start] Task Z-N
                policy.reset(msg)
                ws.send(pack({"type": "ready"}))
            elif kind == "observation":
                # JPEG -> numpy는 꺼낼 때 한다. 정책은 항상 배열만 본다.
                msg["images"] = LazyImages(msg.get("images") or {})
                if "head_l_depth" in msg:      # depth 없는 태스크에는 키 자체가 없다
                    msg["head_l_depth"] = decode_depth(msg["head_l_depth"])
                # 완전한 (T, action_dim) 배열은 그대로, 부분 지정은 0으로 채운다.
                actions = fill_action(policy.infer(msg), action_dim)
                ws.send(pack({"type": "action", "actions": actions}))
            elif kind == "done":
                log_server_info(msg)          # [Done] Task Z-N | 추가 설명
                return
            # 그 밖의 type은 **무시한다** — 평가 서버가 나중에 메시지를 늘려도 안전하다.

    ctx = None
    if certfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)

    # max_size=None: 이미지 관측이 websockets 기본 1MB 프레임 한도를 넘을 수 있다.
    with serve(handler, host, port, max_size=None, process_request=check_auth, ssl=ctx) as server:
        print(f"policy server listening on {'wss' if ctx else 'ws'}://{host}:{port}", flush=True)
        server.serve_forever()
