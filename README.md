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
| 평가 → 정책 | `{"type":"reset", "episode_id", "task_id", "instruction", "conf", "server_info"}` |
| 정책 → 평가 | `{"type":"ready"}` |
| 평가 → 정책 | `{"type":"observation", "sim_time", "images"(JPEG bytes), "state", "scan"(선택), "instruction"}` |
| 정책 → 평가 | `{"type":"action", "actions": (T, action_dim) float32 또는 구조화 dict}` |
| 평가 → 정책 | `{"type":"done", "reason", "server_info"}` (연결 종료. `reason`은 종료 사유 — 로깅용 선택 필드로, 생략될 수 있음) |

**모르는 키·모르는 `type`은 무시하세요.** 평가 서버가 나중에 필드를 늘려도 여러분 서버가
죽지 않아야 합니다 (`server.py`는 이미 그렇게 동작합니다).

`conf` 키:

| 키 | 의미 |
|---|---|
| `action_dim` | 액션 차원. 이 값에 맞는 shape로 반환해야 함 |
| `control_hz` | 액션 1개가 적용되는 주기 |
| `chunk_mode` | `exhaust`(청크 전부 소비 후 재추론) / `periodic`(주기마다 재추론, 남은 청크 폐기) |
| `inference_hz` | `periodic`일 때 재추론 주기 |
| `max_chunk_len` | 청크 최대 길이. 초과분은 잘림 |

### server_info — 에피소드 경계 알림

`reset`과 `done`에만 실립니다 (**관측 틱에는 절대 없습니다**). type은 `Start`와 `Done`
둘뿐이고, 평가 중 진행·채점 정보는 어떤 형태로도 나가지 않습니다.

```
{"type": "Start", "info": "Task A-1"}
{"type": "Done",  "info": "Task B-3 | full marks (early stop)"}
```

`info`는 `"Task Z-N"`(태스크 문자, 시도 번호 1부터) 뒤에 추가 설명이 있으면 `" | "`로
잇습니다. 파싱은 **1회 분리 후 trim** (`info.split("|", 1)`) — 설명 안에 `|`가 또 나와도
안전합니다. 추가 설명은 자연어이므로 내용에 기대지 마세요.

### 주의

- **시간은 시뮬 시간**입니다 — 추론이 오래 걸려도 시뮬은 멈춰 기다리므로 실시간 제약은
  없습니다. 다만 총 평가 시간에는 반영되니 지나치게 느리면 타임아웃됩니다.
- 액션 청크는 `(T, action_dim)` **2차원 float32** 또는 구조화 dict여야 합니다.
  NaN/Inf가 있으면 에피소드가 실패합니다.
- 연결은 에피소드 내내 유지됩니다. 매 스텝 재연결하지 마세요.
- **연결 종료도 에피소드 끝으로 처리하세요.** 평가 쪽 강제 중단(벽시계 타임아웃·프로세스
  킬)에서는 `done` 메시지가 나가지 못하고 연결만 끊깁니다. `Done`만 기다리며 상태를
  붙잡아 두지 마세요.

**관절 각속도 가드**: 관절 공간 액션 모드(`action_mode`가 `joint_abs`인 경우, 그리고 통합
22차원 관절 벡터)에서는 평가 서버가 받은 청크의 관절 타깃 변화율 — 연속한 두 타깃의 차이 ×
`control_hz`, 그리고 직전 타깃에서 청크의 첫 타깃까지 — 을 관절별 한도와 비교합니다.
한 관절이라도 한도를 넘으면 **그 청크는 통째로 무시되고**(일부만 적용하거나 잘라서 쓰지
않습니다) **직전 타깃이 그대로 유지된 채** 시뮬 시간은 계속 흐릅니다. 한도는 실기
FFW-SG2의 관절별 SDK 속도 제한(팔 관절 기준 3~7 rad/s 급)에서 오고 운영진이 조정할 수
있습니다. 그리퍼(이진 열림/닫힘)와 모바일 베이스(속도 명령) 자유도는 검사 대상이 아니며,
`ee_delta` 모드는 수신 시점에 관절 타깃이 없으므로(EE 상대 델타를 시뮬 안 IK가 풉니다)
검사하지 않습니다. 실기가 낼 수 없는 급격한 관절 이동을 요구하지 않으면 걸리지 않습니다.

## 2. 활용 방법 — `server.py`를 임포트해 활용

```bash
pip install websockets msgpack msgpack-numpy numpy pillow
pip install simplejpeg        # 선택. 설치돼 있으면 JPEG 디코드에 자동으로 씁니다 (2–3× 빠름)
```

`BasePolicy`를 상속해 `infer()`만 채우고 `serve_policy()`로 띄웁니다.

```python
from server import BasePolicy, cli, serve_policy

class MyPolicy(BasePolicy):
    def __init__(self):
        self.model = load_my_model()      # 무거운 로딩은 여기서 1회

    def reset(self, msg):
        # 에피소드 시작. msg = {"episode_id", "task_id", "instruction", "conf", "server_info"}
        self.action_dim = msg["conf"]["action_dim"]

    def infer(self, obs):
        # obs["images"]: {"head_l", "wrist_l", "wrist_r"} -> (H,W,3) uint8
        # obs["state"]:  {"joint_q": [18], "lift": [1], "mobile": [3]} float32
        # obs["scan"]:   float32[960] — LiDAR가 있는 태스크에만 있는 키 (get으로 읽으세요)
        # obs["instruction"]: str      obs["sim_time"]: float (시뮬 시간, 벽시계 아님)
        return self.model(obs)            # (T, action_dim) float32, T >= 1

if __name__ == "__main__":
    args = cli()
    serve_policy(MyPolicy(), args.host, args.port, args.token, args.certfile, args.keyfile)
```

동작 확인은 더미 정책으로:

```bash
python random_policy.py --port 8000
```

## 3. 관측과 액션

### 이미지 — 실기 해상도 고정 + JPEG 전송

카메라 관측은 **실기 SDK가 기록하는 최고 해상도 그대로** 옵니다. 해상도 선택 기능은 없습니다.

| 카메라 | 해상도 | 실기 근거 |
|---|---|---|
| `head_l` | 672×376×3 uint8 | ZED Mini **좌안** VGA (실기 기록이 left 스트림뿐) |
| `wrist_l` | 424×240×3 uint8 | 왼손목 RealSense D405 기본 프로파일 |
| `wrist_r` | 424×240×3 uint8 | 오른손목 D405 기본 프로파일 |

전송은 **JPEG 압축**입니다. 실기 기록 파이프라인 자체가 `/compressed`(CompressedImage)
전용이라 학습 데이터와 같은 형식이고, 무압축이면 20Hz에서 220Mbps라 회선이 못 버팁니다.
`server.py`는 **캠별 지연 디코드**를 합니다 — `obs["images"]["head_l"]`처럼 꺼내는 순간
numpy `(H, W, 3) uint8`로 디코드되고, 정책이 안 쓰는 카메라는 디코드 비용이 아예 0입니다.
(그래서 `pillow`가 필요하고, `simplejpeg`이 설치돼 있으면 그쪽을 자동으로 씁니다.)

3캠 q85 기준 대역은 틱당 약 105KB — 20Hz에서 **약 17Mbps**입니다. **권장 회선 20Mbps↑**
(추론 주기를 낮추면 비례해서 줄어듭니다 — 10Hz면 약 9Mbps).

### state — 구조화 dict

`obs["state"]`는 평탄 벡터가 아니라 dict입니다.

| 키 | shape | 내용 |
|---|---|---|
| `joint_q` | float32[18] | 22차원 액션 레이아웃의 0–17: arm_r 7, arm_l 7, 그리퍼 2(**연속 각도**), 목 2 |
| `lift` | float32[1] | 리프트 위치 |
| `mobile` | float32[3] | 베이스 odom (vx, vy, ωz) |

속도(qdot)는 주지 않습니다 — 실기 기록에 없습니다. 필요하면 위치 차분으로 추정하세요.

### scan — LiDAR (있는 태스크만)

`obs["scan"]`은 병합 `/scan` float32[960] (0.375°, 20m 클램프 — 실기 인터페이스와 동일)
입니다. **LiDAR가 없는 태스크에는 키 자체가 없습니다** — `obs.get("scan")`으로 읽으세요.

**크롭·리사이즈 등 전처리는 전부 참가자 몫입니다.** 학습 분포와 맞추세요 (중앙 크롭 후
리사이즈, π0 관행의 레터박스 `resize_with_pad` 등). 해상도·종횡비가 학습 분포와 어긋나면
조용한 성능 저하로 나타납니다.

### 액션 — 청크 길이와 자유도

- `(T, action_dim)` 청크에서 **T=1이면 매 틱 추론**입니다 (제어 20Hz = 초당 20회 왕복).
  청크를 길게 주면 그만큼 추론 횟수가 줄어듭니다 (`chunk_mode` 참조).
- **태스크가 허용하지 않는 자유도는 평가 서버가 0으로 마스킹합니다.** 거부가 아니라 0 처리라
  에피소드는 그대로 진행됩니다 (예: 매장 진열 태스크의 이동·리프트).
- **22차원 통합 액션 벡터** (관측 `state`와 같은 이름·같은 순서):

  | 인덱스 | 구간 | 내용 |
  |---|---|---|
  | 0–6 | `arm_r` | 오른팔 관절 7 |
  | 7–13 | `arm_l` | 왼팔 관절 7 |
  | 14 / 15 | `gripper_r` / `gripper_l` | **연속 각도** (실기 마스터 조인트 각도와 같은 의미) |
  | 16–17 | `head` | 목 관절 2 |
  | 18 | `lift` | 리프트 |
  | 19–21 | `base` | 모바일 (vx, vy, ωz) |

- **구조화 액션(권장)**: `actions`를 아래 dict로 돌려줘도 됩니다. **누락 그룹은 평가 서버가
  0으로 채웁니다.**

  | 그룹 | shape | 22차원 구간 |
  |---|---|---|
  | `joint_q` | (T, 18) | 0–17 (`state["joint_q"]`와 같은 순서) |
  | `lift` | (T, 1) | 18 |
  | `mobile` | (T, 3) | 19–21 |

  `server.py`의 `action_groups(joint_q=..., lift=..., mobile=...)`이 이 dict를 만들어
  줍니다. **평탄 `(T, 22)` 배열도 계속 받습니다** — 기존 방식 그대로 써도 됩니다.
  일부 구간만 쓰고 싶으면 `{"arm_r": q7, "gripper_r": -1.0}` 같은 dict나 짧은 벡터를 돌려줘도
  됩니다 — `fill_action()`이 나머지 자유도를 0으로 채웁니다.

## 4. 활용 예시 — 기존 추론 스택 연결 (GR00T, openpi 등)

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
            "video.head": obs["images"]["head_l"][None],
            "state.joints": obs["state"]["joint_q"][None],
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

## 5. 동작 예시 — 토큰 인증

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
반영됩니다. 에피소드 경계에서는 `server_info`가 콘솔에 찍힙니다:

```text
[Start] Task A-1
[Done] Task A-1 | full marks (early stop)
```

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

## 6. 웹 페이지에 IP 등록

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

4. 같은 페이지의 **"토큰 확인" 버튼**으로 제출 토큰을 확인해 5번 섹션처럼 `--token`으로
   서버에 걸어 둡니다.

   ![토큰 확인](./figure/token-issue.png)

5. 제출하면 평가가 큐에 들어가고, 진행 상태와 점수는 제출 페이지에서 확인합니다.

   ![평가 진행 상태](./figure/job-status.png)
