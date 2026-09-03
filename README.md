# eval-host-server — 참가자 정책 서버

VLA(Vision-Language-Action) 시뮬레이션 평가에 제출할 **정책 서버**의 골격입니다.
`BasePolicy`를 상속해 `infer()` 하나만 채우면 통신·인코딩·인증은 골격이 처리합니다.

## 빠른 시작

```bash
# Python 3.10 이상 필요
pip install "websockets>=13" msgpack msgpack-numpy numpy pillow
pip install simplejpeg          # 선택 — 있으면 JPEG 디코드를 2~3× 빠르게

# 데모 서버를 띄우고 기다립니다. 평가 서버가 접속하면 첫 관측을 저장합니다.
python demo_server.py --port 8000 --save-dir ./received
```

`demo_server.py`는 액션 없이(로봇 정지) 관측만 저장하는 **배선 확인용**입니다. 여기서
받는 관측이 눈에 보이면 배선이 맞은 것이고, 그다음 §3처럼 여러분의 모델을 `infer()`에
연결하면 됩니다. 실제 제출에는 토큰·공인 IP가 필요합니다(§5).

| 파일 | 역할 |
|---|---|
| `server.py` | 정책 서버 골격 — `BasePolicy` + `serve_policy()` + `cli()`. 직접 실행하면 관측 확인용 `LogPolicy`로 뜹니다(§4.2) |
| `demo_server.py` | 배선 확인용 데모 — 액션 0, 에피소드 첫 관측을 `--save-dir`에 저장(§4.1) |
| `wss_setup.sh` | TLS 인증서 생성·제출값 출력 자동화(§5) |
| `figure/` | README용 관측 샘플 이미지(카메라·depth·LiDAR 미리보기) |

---

## 1. 개요

- 참가자가 이 **정책 서버**를 공인 IP에 띄우고, **평가 서버가 클라이언트로 접속**합니다.
  아웃바운드 연결은 평가 서버 쪽에서만 일어납니다 — 여러분 서버는 받기만 합니다.
- 평가 서버가 `이미지·instruction·상태`를 보내면 여러분 서버가 `액션 청크`를 돌려주고,
  그 액션으로 시뮬레이션을 굴려 채점하는 **closed-loop** 구조입니다.
- 통신은 **WebSocket + msgpack**. numpy 배열은 `msgpack_numpy`로 dtype/shape 그대로 왕복합니다.
- **시뮬 시간 기준 고정 주기**라 실시간이 아닙니다 — 추론이 느려도 시뮬은 멈춰 기다리므로
  점수에 불리하지 않습니다. 단, **추론 1회가 30초를 넘으면** 그 평가는 종료됩니다.

```
평가 서버                          여러분의 정책 서버 (이 레포)
   │  ── reset ─────────────────▶  에피소드 시작 (conf 전달)
   │  ◀── ready ──────────────────
   │  ── observation ───────────▶  이미지·depth·state·scan·instruction
   │  ◀── action (T, action_dim) ─  액션 청크
   │           ⋮  (청크를 control_hz 로 소진한 뒤 다음 observation)
   │  ── done ──────────────────▶  에피소드 끝 (응답 없이 연결 종료)
```

---

## 2. 통신 규약

메시지는 전부 dict이고 msgpack으로 직렬화됩니다. **모르는 키·모르는 type은 무시**하세요 —
평가 서버가 나중에 필드를 늘려도 여러분 서버가 죽지 않게 하는 규칙입니다.

```
수신 {"type":"reset", "episode_id","task_id","instruction","conf","server_info"}
                                  → 송신 {"type":"ready"}
수신 {"type":"observation", "sim_time","images","head_l_depth"(선택),"state","scan"(선택),"instruction"}
                                  → 송신 {"type":"action", "actions": (T, action_dim) float32
                                          또는 {"joint_q","lift","mobile"} 구조화 dict}
수신 {"type":"done", "reason","server_info"}   → 응답 없이 연결 종료
```

### conf — reset 에 실려 오는 설정

| 키 | 값 | 뜻 |
|---|---|---|
| `action_dim` | 22 | 액션 벡터 차원. `infer` 반환 shape가 이 값에 맞아야 합니다 |
| `control_hz` | 20 | 액션 1개가 적용되는 주기(이번 대회 고정). 50개 청크 ≈ 2.5초 |
| `max_chunk_len` | 정수 | 청크 상한. 넘으면 **앞에서 이만큼만 쓰고 나머지(뒤)는 버립니다** |

### 입력 — `infer(obs)` 가 받는 것

골격이 관측을 디코드해 `infer(obs)`에 넘기므로 정책 코드는 numpy 배열만 봅니다.
**이미지는 꺼낼 때 디코드**됩니다 — 안 꺼낸 카메라는 디코드 비용이 0입니다.

**① 이미지 `images`** — 실기 SDK 기록 해상도로 **고정**(선택 기능 없음). JPEG로 오고
꺼낼 때 `(H, W, 3) uint8`로 디코드됩니다. 크롭·리사이즈 등 전처리는 정책 몫입니다.

| 키 | 해상도 (H×W) | 카메라 | 샘플 |
|---|---|---|---|
| `head_l` | 376×672 | 머리 ZED Mini **좌안** | ![head_l](./figure/head_l.png) |
| `wrist_l` | 240×424 | **왼손목** RealSense D405 | ![wrist_l](./figure/wrist_l.png) |
| `wrist_r` | 240×424 | **오른손목** RealSense D405 | ![wrist_r](./figure/wrist_r.png) |

**② head depth `head_l_depth`** — 머리 카메라 depth. `(376, 672) float32` **미터**,
`0 = 무효`. `head_l`(좌안)과 짝이라 `depth[y,x]`가 `head_l[y,x]`와 같은 픽셀입니다.
depth가 없는 태스크에는 키 자체가 없습니다. **손목에는 depth가 없습니다 — head만.**

> ⚠️ **이 depth는 시뮬레이터의 이상화(idealized) 값입니다.** 실기 ZED Mini의 스테레오
> depth와 달리 노이즈·구멍(무텍스처/반사면)·시차 한계가 없는 완벽한 값입니다. 그대로
> 의존하는 정책은 실기에서 성능이 떨어지니 학습 시 노이즈/드롭아웃 증강을 권장합니다.

![head_l_depth](./figure/head_l_depth_view.png)

*(사람이 보도록 정규화한 미리보기 — 가까울수록 흰색, 무효는 검정. 실제 전달값은 미터 float32.)*

**③ 상태 `state`** — 구조화 dict `{"joint_q":[18], "lift":[1], "mobile":[3]}` float32.
`joint_q` 순서는 아래 액션 레이아웃의 0–17과 같습니다(`arm_r7, arm_l7, grip_r, grip_l, head2`,
그리퍼는 연속 각도). `lift`는 리프트, `mobile`은 베이스 odom(vx, vy, ωz). 속도(qdot)·
목표물 참값 포즈는 **주지 않습니다**(비전으로 풀도록 참값 상태는 감춥니다).

**④ LiDAR `scan`** — `float32[960]`, 병합 2D 스캔(거리, 미터). 로봇 몸통(base_link) 기준
**360°**를 960빈으로(0.375°/빈) 나눈 값이고, **빈 i의 각도 = -π + i·(2π/960)**
(`atan2(y, x)`, x=전방·y=좌). 값은 0.05~20 m, **20 m ≈ 무반사**. LiDAR가 있는 태스크에서만
실리고, 없으면 키 자체가 없습니다.

![scan](./figure/scan_view.png)

*(top-down 미리보기 — 중앙 주황 삼각형이 로봇 전방, 링은 1 m 간격, 초록 점이 반사.)*

**⑤ `instruction`** — 자연어 지시(str). `sim_time`(float)도 함께 옵니다.

### 출력 — `infer(obs)` 가 돌려주는 것

액션 청크를 세 가지 형식 중 하나로 반환합니다:

- **`(T, action_dim)` 2차원 float32** — 22개 자유도를 전부 채운 청크(레이아웃은 아래 표), 또는
- **구조화 dict** `{"joint_q":(T,18), "lift":(T,1), "mobile":(T,3)}` — 쓰는 그룹만, 또는
- **부분 dict** `{"arm_r": q7, "gripper_r": -1.0}` — 구간 이름으로 일부만.

구조화·부분 dict의 **누락 자유도는 골격이 0으로 채웁니다** — 여러분은 그냥 반환만 하면
됩니다. `T`는 1 이상, `max_chunk_len`을 넘으면 앞에서 그만큼만 씁니다. **NaN/Inf가 있으면
에피소드가 실패**합니다. 태스크가 허용하지 않는 자유도(예: Task B의 lift·base)는 보내도
**평가 서버가 0으로 마스킹**합니다(누락 채움과는 다른 개념).

**22-dim 레이아웃** — 관측 `state`와 같은 순서입니다.

| 인덱스 | 그룹(부분 dict 이름) | 내용 | 의미·단위 |
|---|---|---|---|
| 0–6 | `arm_r` | 오른팔 관절 7 | **절대 관절각** (rad) |
| 7–13 | `arm_l` | 왼팔 관절 7 | **절대 관절각** (rad) |
| 14 | `gripper_r` | 오른손 그리퍼 | 연속 각도 |
| 15 | `gripper_l` | 왼손 그리퍼 | 연속 각도 |
| 16–17 | `head` | 목 관절 2 | **절대 관절각** (rad) |
| 18 | `lift` | 리프트 | **절대 위치** (m) |
| 19–21 | `base` | 베이스 (= 구조화 dict의 `mobile`) | **속도 명령** (vx, vy, ωz) |

- 구조화 dict의 `joint_q`(18) = 위 인덱스 0–17, `lift` = 18, `mobile` = `base` = 19–21입니다.
- **관절·리프트는 절대 타깃(delta 아님), 베이스만 속도 명령**입니다. 그룹마다 의미가 다릅니다.

### server_info — 에피소드 경계 알림

`reset`과 `done`에만 실립니다(**관측 틱에는 없습니다**). type은 `Start`·`Done` 둘뿐이고,
평가 중 진행·채점 정보는 어떤 형태로도 나가지 않습니다.

```
{"type": "Start", "info": "Task B-1"}
{"type": "Done",  "info": "Task B-3 | full marks (early stop)"}
```

`info`는 `"Task Z-N"`(태스크 문자·시도 번호) 뒤에 설명이 있으면 `" | "`로 잇습니다.
파싱은 첫 `|`로 한 번만 분리하세요. 추가 설명은 자연어라 내용에 기대지 마세요.

### 조기 종료·각속도 가드

- **만점 조기 종료**(`reason:"full_marks"`): 채점이 전부 만점인 상태가 **시뮬 2.5초** 유지되면 종료.
- **정체 종료**(`reason:"stalled"`): 총점이 **시뮬 3분** 무진전이면 종료(시작부터 셈).
- **관절 각속도 가드**: 청크의 관절 타깃 변화율이 실기 FFW-SG2 속도 한도(팔 3~7 rad/s 급)를
  넘으면 **그 청크는 통째로 무시**되고 직전 타깃이 유지됩니다. 그리퍼·베이스는 검사 대상이
  아닙니다. 실기가 낼 수 있는 속도면 걸리지 않습니다.

---

## 3. 정책 구현

`BasePolicy`를 상속해 `infer()`만 채우고 `serve_policy()`로 띄웁니다. 기존 스택
(GR00T, openpi, 자체 아키텍처)은 `infer()` 안에서 브리지로 호출하면 됩니다.

```python
import numpy as np
from server import BasePolicy, action_groups, cli, serve_policy

class MyPolicy(BasePolicy):
    def __init__(self):
        self.model = load_my_model()        # 무거운 로딩은 여기서 한 번만

    def reset(self, msg: dict) -> None:
        self.action_dim = msg["conf"]["action_dim"]   # 에피소드별 상태 초기화만

    def infer(self, obs: dict) -> np.ndarray:
        head  = obs["images"]["head_l"]     # (376,672,3) uint8 — 꺼낼 때 디코드
        wrist = obs["images"]["wrist_r"]    # (240,424,3) uint8
        depth = obs.get("head_l_depth")     # (376,672) float32 미터, 0=무효 (없을 수 있음)
        scan  = obs.get("scan")             # float32[960] LiDAR (없을 수 있음)
        state = obs["state"]                # {"joint_q":[18], "lift":[1], "mobile":[3]}
        text  = obs["instruction"]

        # ── 여기서 모델을 호출해 액션 청크를 만듭니다 ──
        return self.model(head, wrist, depth, state, text)   # (T, 22) float32

        # 일부 그룹만 쓰려면 구조화 dict로 반환해도 됩니다(나머지는 0으로 채워짐):
        # return action_groups(joint_q=my_joint_targets)     # (T, 18)

if __name__ == "__main__":
    args = cli()
    serve_policy(MyPolicy(), args.host, args.port, args.token, args.certfile, args.keyfile)
```

- **모델 로딩은 `__init__`에서 한 번.** `reset`은 에피소드마다 호출되므로 여기서 모델을
  다시 올리면 매 에피소드 수십 초를 낭비하고 30초 추론 한도에 걸릴 수 있습니다.
- `reset` 오버라이드는 선택입니다(필요 없으면 생략).
- 부분 dict `{"arm_r": q7}`는 **단일 타임스텝**만 만듭니다. 여러 스텝 청크는 `(T,22)`
  배열이나 `action_groups(...)`를 쓰세요.

---

## 4. 배선 테스트

제출 전에 평가 서버가 보내는 관측이 제대로 도착하는지 확인하는 두 도구입니다. 둘 다
**평가 서버가 접속해 와야** 동작하므로, 띄운 뒤 제출(§6)해서 평가가 붙을 때 로그가 찍힙니다.

### 4.1 `demo_server.py` — 관측을 파일로 저장

액션 0(로봇 정지)을 돌려주고 매 에피소드 **첫 관측**을 `--save-dir`에 저장합니다.

```bash
python demo_server.py --port 8000 --token "$(cat token.txt)" --save-dir ./received
```

저장물(에피소드별 하위 폴더): `head_l.png`·`wrist_l.png`·`wrist_r.png`(RGB),
`head_l_depth.png`(16-bit mm) + `head_l_depth_view.png`(미리보기), `scan.npy` + `scan_view.png`,
그리고 `obs.json`:

```json
{ "sim_time": 0.0, "instruction": "Sort the objects in the basket: ...",
  "images": { "head_l": [376,672,3], "wrist_l": [240,424,3], "wrist_r": [240,424,3] },
  "state":  { "joint_q": [18], "lift": [1], "mobile": [3] },
  "has_head_l_depth": true, "has_scan": true, "scan_len": 960 }
```

### 4.2 `server.py` 직접 실행 — 매 틱 로그

`server.py`를 직접 실행하면 정책 대신 확인용 `LogPolicy`가 떠서 매 틱 관측 구조를 찍습니다
(액션은 현재 자세 유지 → 로봇 정지). `--save-obs DIR`을 주면 카메라 관측이 `DIR/*.jpg`로
매 틱 덮어써져, 이미지 뷰어로 열어 두면 실시간으로 볼 수 있습니다.

```bash
python server.py --port 8000 --token "$(cat token.txt)" --save-obs obs_dump
```

```text
policy server listening on ws://0.0.0.0:8000
[Start] Task A-1
[obs0] images: {head_l: ndarray(376, 672, 3) uint8, wrist_l: (240, 424, 3), wrist_r: (240, 424, 3)}
[obs0] state: {joint_q: (18,) float32, lift: (1,) float32, mobile: (3,) float32}  scan: (960,)
[obs]  t=0.000 head_l(376,672,3) ... joint_q[18] scan[960] instr='pick up the cup'
[Done] Task A-1 | full marks (early stop)
```

`[obs0]`(첫 틱)은 받은 관측의 **전체 구조**(키·shape·dtype)라, 여러분이 가정한 규격과
대조하기 좋습니다. 관측 경로가 확인되면 §3의 `BasePolicy`로 바꿔 띄우세요.

---

## 5. 배포 필수 조건

- **공인 IP + 포트포워딩.** 평가 서버가 여러분 서버로 접속하므로, 공인 IP(또는 도메인)의
  열린 포트에 있어야 합니다. 방화벽·공유기에서 인바운드를 허용하세요. 사설 IP(192.168.x,
  10.x)·NAT 뒤 주소는 도달할 수 없어 제출이 실패합니다.
- **토큰을 거세요.** 제출 페이지 **"토큰 확인"**에서 토큰을 받아 `--token`으로 겁니다. 평가
  서버는 `Authorization: Bearer <토큰>`을 검증하고 그 외 접속은 401로 거부합니다. 없으면
  주소를 아는 누구나 여러분 모델의 출력을 뽑아 가거나, 다른 팀이 여러분 주소를 제출해
  점수를 가져갈 수 있습니다.
  ```bash
  echo "tok_EXAMPLE_0000000000000000" > token.txt   # 발급받은 실제 토큰으로 교체
  # token.txt 는 .gitignore 에 넣어 커밋하지 마세요.
  ```
- **평가 내내 서버를 켜 두세요.** 제출 → 큐 대기 → 평가(에피소드 여러 회, 수십 분) 동안
  서버가 살아 있어야 합니다. 중간에 꺼지면 그 평가는 실패합니다.
- **연결 처리.** 연결은 에피소드 내내 유지됩니다(매 스텝 재연결 금지). **`done` 수신과 연결
  종료를 둘 다 에피소드 끝으로** 처리하세요 — 강제 중단 시 `done`이 못 나가고 연결만 끊깁니다.
- **TLS(wss://) 권장.** `ws://`는 평문이라 공유 망(연구실 LAN·기숙사·카페)에서 같은 망의
  타인이 토큰을 볼 수 있습니다. 인증서 생성부터 제출값 출력까지 한 번에:
  ```bash
  ./wss_setup.sh <공인 IP 또는 도메인>
  # → cert.pem/key.pem 생성 + 제출 페이지에 등록할 "서버 주소·인증서 지문"·실행 명령 출력
  ```
  도메인이 있으면 정식 인증서(Let's Encrypt 등), 없으면 self-signed 인증서의 **지문**을
  제출 페이지에 등록하세요(평가 서버는 등록된 지문과 일치하는 인증서만 신뢰). 인증서를
  새로 만들면 지문이 바뀌니 제출 페이지도 갱신하세요. `key.pem`은 절대 공유 금지.

---

## 6. 제출

**참가신청(개인정보 동의) → 운영진 승인 → 로그인 → 정책 서버 기동 → 제출**

1. 참가신청: 팀 정보·연락처 입력 + 개인정보 수집·이용 동의. **승인 전에는 제출이 거부**됩니다.
2. 승인 후 로그인 → 정책 서버를 공인 IP의 열린 포트로 기동(§5).
3. 제출 페이지에서 서버 주소를 `ws://<공인IP>:<포트>`(또는 `wss://`)로 등록. wss면 인증서 지문 입력.
4. **"토큰 확인"**으로 토큰을 받아 `--token`으로 서버에 겁니다.
5. 제출하면 평가가 큐에 들어가고, 진행 상태·점수는 제출 페이지에서 확인합니다.

---

## 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| 제출했는데 서버에 **아무 로그도 안 찍힌다** | 토큰 불일치(401) — 거부는 handler 전에 일어나 서버엔 안 보임 | 제출 페이지의 토큰과 `--token`이 정확히 같은지 확인 |
| 〃 | 방화벽·NAT·사설 IP로 평가 서버가 도달 못 함 | 공인 IP·인바운드 포트 개방 확인(§5) |
| 〃 (wss) | 제출한 인증서 지문 ≠ 실제 인증서 | 인증서 재생성 후 지문 갱신 제출 |
| 연결은 되는데 **평가만 실패** | `infer` 예외, 또는 액션에 NaN/Inf | 반환 shape·dtype·값 확인. `demo_server.py`로 배선부터 격리 |
| 〃 (`policy_timeout`) | 추론 1회가 30초 초과(첫 추론 모델 로딩 포함) | 모델 로딩을 `__init__`으로, 추론 최적화 |
| 로봇이 **엉뚱하게 움직임** | 액션 의미 착오 — 관절=절대각인데 delta로 채움, base를 위치로 착각 | §2 22-dim 레이아웃의 의미·단위 확인 |
| 청크가 무시된 듯 정지 | 관절 각속도 가드에 걸림(실기 한도 초과) | 연속 타깃 간 변화율을 실기 속도 이내로 |

---

## 요구사항

- **Python 3.10 이상** (골격이 `X | None` 애노테이션을 써서 3.9 이하는 import 시 실패)
- `pip install "websockets>=13" msgpack msgpack-numpy numpy pillow`
- (선택) `pip install simplejpeg` — 있으면 JPEG 디코드에 자동으로 씁니다(2~3× 빠름)
</content>
