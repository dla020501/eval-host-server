# eval-host-server — 참가자 정책 서버

여러분은 **정책 서버**를 띄우기만 하면 됩니다. 평가 서버가 여러분의 서버에 WebSocket으로
접속해 관측(카메라 이미지 + 관절 상태 + instruction)을 보내고, 액션 청크를 받아
시뮬레이션을 굴려 채점합니다. 여러분의 코드가 우리 쪽에서 실행되는 일은 없습니다 —
연결은 항상 평가 서버 → 여러분 서버 방향입니다.

## 1. 코드 개요

| 파일 | 역할 |
|---|---|
| `server.py` | 정책 서버 골격. `BasePolicy`(인터페이스) + `serve_policy()`(WebSocket 서버) + `cli()`(공통 인자) |
| `random_policy.py` | 더미 정책 예시. 복사해서 `infer()`만 바꾸면 여러분의 정책 서버가 됩니다 |
| `figure/` | README 그림 (제출 페이지 캡처 등) |

통신 계약은 아래 프로토콜이 전부입니다. `server.py`는 편의용 골격일 뿐 필수가 아닙니다 —
GR00T, openpi(π0) 등 어떤 아키텍처든, 어떤 언어든 이 프로토콜대로 WebSocket 서버를 열면
평가받을 수 있습니다.

| 방향 | 메시지 (msgpack, numpy 배열은 `msgpack_numpy`) |
|---|---|
| 평가 → 정책 | `{"type":"reset", "episode_id", "task_id", "instruction", "conf"}` |
| 정책 → 평가 | `{"type":"ready"}` |
| 평가 → 정책 | `{"type":"observation", "sim_time", "images", "state", "instruction"}` |
| 정책 → 평가 | `{"type":"action", "actions": (T, action_dim) float32}` |
| 평가 → 정책 | `{"type":"done"}` (연결 종료) |

`conf` 키: `action_dim`(액션 차원), `control_hz`(액션 적용 주기),
`chunk_mode`(`exhaust` = 청크 소진 후 재추론 / `periodic` = 주기 재추론),
`inference_hz`, `max_chunk_len`(초과분은 잘림).

**시간은 시뮬 시간**입니다 — 추론이 오래 걸려도 시뮬은 멈춰 기다리므로 실시간 제약은
없습니다(총 평가 시간에는 반영). 액션은 `(T, action_dim)` 2차원 float32여야 하고
NaN/Inf가 있으면 에피소드가 실패합니다. 연결은 에피소드 내내 유지됩니다.

## 2. 활용 방법 — `server.py`를 임포트해 활용

```bash
pip install websockets msgpack msgpack-numpy numpy
```

`BasePolicy`를 상속해 `infer()`만 채우고 `serve_policy()`로 띄웁니다.

```python
from server import BasePolicy, cli, serve_policy

class MyPolicy(BasePolicy):
    def __init__(self):
        self.model = load_my_model()      # 무거운 로딩은 여기서 1회

    def reset(self, msg):
        # 에피소드 시작. msg = {"episode_id", "task_id", "instruction", "conf"}
        self.action_dim = msg["conf"]["action_dim"]

    def infer(self, obs):
        # obs["images"]: {cam: (H,W,3) uint8}   obs["state"]: (D,) float32
        # obs["instruction"]: str               obs["sim_time"]: float
        return self.model(obs)                # (T, action_dim) float32, T >= 1

if __name__ == "__main__":
    args = cli()
    serve_policy(MyPolicy(), args.host, args.port, args.token, args.certfile, args.keyfile)
```

동작 확인은 더미 정책으로:

```bash
python random_policy.py --port 8000
```

## 3. 활용 예시 — 기존 추론 스택 연결 (GR00T, openpi 등)

이미 추론 파이프라인이 있다면 `infer()`를 **브리지**로 씁니다. `infer()`는 평가 서버와의
왕복 한 번일 뿐, 내부에서 별도 추론 서비스를 호출하든 여러 프로세스로 쪼개든 자유입니다.

```python
class MyVLA(BasePolicy):
    def __init__(self):
        self.client = MyInferenceClient("localhost:5555")   # 별도로 띄운 추론 서비스

    def reset(self, msg):
        self.client.reset(task=msg["instruction"])

    def infer(self, obs):
        # 관측 -> 모델 입력 변환 -> 추론 -> (T, action_dim) 변환
        action = self.client.get_action({
            "video.head": obs["images"]["head_cam"][None],
            "state.joints": obs["state"][None],
            "annotation.task": [obs["instruction"]],
        })
        return to_chunk(action)   # (T, action_dim) float32
```

π0.5 정책을 실제로 붙여 본 경험에서 나온 어댑터 체크리스트:

- **관측/액션 규약을 표로 먼저 고정**하세요 (카메라 이름·해상도, state 구성, 액션 스케일·
  그리퍼 부호, 주기). 평가 환경과 학습 분포의 차이(해상도, 레터박스, 상대/절대 관절각)는
  조용한 성능 저하로 나타납니다.
- **`--selftest` 같은 어댑터 회귀 테스트**를 만들어 두세요 — GPU 없이 변환 로직만 검증하는
  경로가 있으면 규약 착오(부호 반전, 정규화 범위)를 서버를 띄우기 전에 잡습니다.
- 에피소드별 송수신 로그(수신 관측·송신 액션)를 남기면 원격 평가 결과를 디버깅할 수 있습니다.

## 4. 동작 예시 — 토큰 인증

**제출 토큰**은 로그인 후 평가 제출 페이지의 **"토큰 확인" 버튼**에서 확인합니다.
이 토큰을 서버에 걸어 두면, 평가 서버가 보내는
`Authorization: Bearer <토큰>` 헤더를 검증하고 그 외 접속은 HTTP 401로 거부합니다.
**꼭 켜세요** — 없으면 주소를 아는 누구나 여러분 모델의 출력을 뽑아 갈 수 있고, 다른
팀이 여러분의 서버 주소를 제출해 점수를 가져갈 수도 있습니다.

```bash
# 토큰은 파일에 두고 셸 치환으로 넘기면 명령 기록에 남지 않습니다 (.gitignore에 포함됨).
echo "tok_EXAMPLE_0000000000000000" > token.txt        # <- 발급받은 실제 토큰으로 교체
python random_policy.py --port 8000 --token "$(cat token.txt)"
```

```text
policy server listening on ws://0.0.0.0:8000
```

이 상태에서 제출하면: 평가 큐 등록 → 평가 서버가 접속(토큰 검증) → `reset` →
`observation`/`action` 반복 → `done` 순으로 에피소드가 진행되고, 점수는 리더보드에
반영됩니다.

### TLS (wss://) — 권장

`ws://`는 평문입니다. **공유 망(연구실 공용 LAN, 기숙사, 카페 Wi-Fi)에서 서버를 돌린다면
wss를 꼭 쓰세요** — ws는 같은 망의 타인이 제출 토큰을 볼 수 있고, 토큰이 새면 여러분의
모델이 무단 호출될 수 있습니다.

가장 쉬운 방법은 자동화 스크립트입니다 — 인증서 생성부터 제출값 출력까지 한 번에:

```bash
./wss_setup.sh <공인 IP 또는 도메인>
# → cert.pem/key.pem 생성 + 제출 페이지에 등록할 "서버 주소·인증서 지문"과 실행 명령 출력
```

수동으로 하려면: 도메인이 있으면 정식 인증서(Let's Encrypt 등)로 `wss://도메인:포트`를
등록하고, 없으면 self-signed 인증서를 만들어 **지문**을 제출 페이지에 함께 등록하세요
(평가 서버는 등록된 지문과 일치하는 인증서만 신뢰합니다).

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=<제출할 호스트명 또는 IP>" \
    -keyout key.pem -out cert.pem
openssl x509 -in cert.pem -noout -fingerprint -sha256    # -> 제출 폼의 "인증서 지문" 칸에
python random_policy.py --port 8000 --token "$(cat token.txt)" --certfile cert.pem --keyfile key.pem
```

인증서를 새로 만들면 지문이 바뀌므로 제출 페이지에서도 갱신해야 합니다. `key.pem`은
절대 공유하지 마세요.

## 5. 웹 페이지에 IP 등록

참가 흐름: **참가신청(개인정보 동의 필수) → 운영진 승인 → 로그인 → 평가 제출**

1. **참가신청**: 팀 정보와 연락처(담당자 이름·전화·이메일)를 입력하고 **개인정보
   수집·이용에 동의**해야 신청됩니다. 신청 후에는 운영진 승인 대기 상태이며,
   **승인 전에는 제출이 거부됩니다.**

   ![참가신청 폼](./figure/signup-form.png)

2. 승인되면 로그인한 뒤, 정책 서버를 **공인 IP(또는 도메인)의 열린 포트**로 띄웁니다.
   방화벽/공유기에서 해당 포트 인바운드를 허용해야 평가 서버가 접속할 수 있습니다.
3. 평가 제출 페이지에서 서버 주소를 `ws://<공인IP>:<포트>` 형식으로 등록합니다.
   `wss://`를 선택하면 인증서 지문 칸이 나타납니다 (self-signed인 경우 입력).

   ![제출 폼 — 서버 주소 등록](./figure/submit-form.png)

4. 같은 페이지의 **"토큰 확인" 버튼**으로 제출 토큰을 확인해 4번 섹션처럼 `--token`으로
   서버에 걸어 둡니다.

   ![토큰 확인](./figure/token-issue.png)

5. 제출하면 평가가 큐에 들어가고, 진행 상태와 점수는 제출 페이지에서 확인합니다.

   ![평가 진행 상태](./figure/job-status.png)
