# eval-host-server

> **MILAB 전용 포크** ([dla020501/eval-host-server](https://github.com/dla020501/eval-host-server)). 원본은
> [kairobahq/eval-host-server](https://github.com/kairobahq/eval-host-server)이며 이 저장소는 MILAB 서버에서 검증한 설정 절차를 [MILAB 서버 설정 가이드](#milab-서버-설정-가이드) 절로
> 추가한 것이다. 프로토콜과 API는 원본과 같다.

VLA 시뮬레이션 평가에 제출하는 참가자 정책 서버의 최소 구현 템플릿이다.

## 개요

이 저장소는 참가자가 실행하는 WebSocket 정책 서버를 제공한다. 참가자는 `BasePolicy`를
상속해 `infer()`를 구현한다. 통신, 이미지 디코드, 인증은 `evalhost` 패키지가 처리하며 정책
로직과 학습 코드는 이 저장소에 없다. 정책 서버는 참가자가 공인 IP의 열린 포트에서 실행한다. 평가
서버는 WebSocket 클라이언트이며 그 주소로 아웃바운드 접속한다. 정책 서버는 접속을 받기만 한다.

평가 서버는 관측을 전송하고 정책 서버는 액션 청크를 응답한다. 평가 서버는 그 액션으로
시뮬레이션을 진행하고 채점한다. 진행은 시뮬레이션 시간 기준 고정 주기라 실시간이 아니며,
추론이 느린 동안 시뮬레이션은 정지한 채 기다린다. 메시지는 msgpack으로 직렬화하여 송수신하고
numpy 배열은 `msgpack_numpy` 확장으로 dtype과 shape를 유지한다.

## 요구사항

- Python 3.10 이상
- 정책 서버는 공인 IP 또는 도메인의 열린 포트에서 실행.
- 필수 의존성은 `websockets`, `msgpack`, `msgpack-numpy`, `numpy`, `Pillow`이다.
- `simplejpeg`은 선택 의존성이다. `pip install "eval-host-server[fast]"`로 설치하면 JPEG
  디코드에 자동으로 사용된다.

## 설치

```bash
pip install git+https://github.com/kairobahq/eval-host-server@v0.1.1
```

`@` 뒤의 태그가 설치되는 버전이다. 최신 태그와 변경 내역은
[Releases](https://github.com/kairobahq/eval-host-server/releases) 페이지에 있다. MILAB에서는 아래처럼 포크를 복제하고 editable로 설치한다.

```bash
git clone https://github.com/dla020501/eval-host-server.git
cd eval-host-server
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

의존성은 `pyproject.toml`이 정의한다. 모델 실행에 필요한 패키지는 같은 가상환경에 추가로
설치한다. conda를 쓰는 경우 `python -m venv` 대신 `conda create -n eval-host python=3.10 &&
conda activate eval-host`로 환경을 만들고 같은 `pip install` 명령을 실행한다.

## 빠른 시작

`evalhost-demo`는 액션을 전부 0으로 응답하고 에피소드마다 첫 관측을 파일로 저장하는 데모
서버다. 평가 서버와의 연결은 이 명령으로 먼저 테스트한다. MILAB 서버에서 실제로 통과한 순서와
방화벽 설정은 [MILAB 서버 설정 가이드](#milab-서버-설정-가이드)에 있다.

```bash
evalhost-demo --port 8000 --save-dir ./received
```

정책은 `BasePolicy`를 상속해 `infer()`를 구현하고 `serve_policy()`로 실행한다. 아래
내용을 `my_policy.py`로 저장한다. 같은 코드가 [examples/my_policy.py](examples/my_policy.py)에
있다. `load_my_model()`은 참가자가 구현한다.

```python
import numpy as np
from evalhost import BasePolicy, cli, serve_policy


class MyPolicy(BasePolicy):
    def __init__(self):
        self.model = load_my_model()

    # msg 구조는 reset 절 확인.
    def reset(self, msg: dict) -> None:
        self.action_dim = msg["conf"]["action_dim"]

    # obs 구조는 observation 절 확인.
    def infer(self, obs: dict) -> np.ndarray:
        head = obs["images"]["head_l"]
        state = obs["state"]
        
        # return type은 action절 확인
        return self.model(head, state, obs["instruction"])


if __name__ == "__main__":
    args = cli()
    serve_policy(MyPolicy(), args.host, args.port, args.token, args.certfile, args.keyfile)
```

```bash
export EVALHOST_TOKEN=<발급받은 토큰>
python my_policy.py --port 8000
```

모델 로드는 `__init__`에서 한 번만 한다. `reset()`은 에피소드마다 호출되므로 여기서 다시
로드하면 추론 응답 타임아웃을 초과한다. `reset()` 재정의는 선택이다.

### 연결 테스트

`evalhost-demo`는 에피소드마다 `--save-dir` 아래 `ep<번호>` 폴더를 만들고 첫 관측을
저장한다. 저장 파일은 `head_l.png`, `wrist_l.png`, `wrist_r.png`, `head_l_depth.png`,
`head_l_depth_view.png`, `scan.npy`, `scan_view.png`, `obs.json` 여덟 개다.

`evalhost-log`는 관측 확인용 `LogPolicy`를 실행한다. 이 정책은 관측 구조를 출력하고 액션을
전부 0으로 응답하므로 로봇은 리셋 포즈를 유지한다. `--save-obs`를 지정하면 관측 이미지를 매
틱 같은 파일에 덮어쓴다.

```bash
export EVALHOST_TOKEN=<발급받은 토큰>
evalhost-log --port 8000 --save-obs obs_dump
```

실행 결과는 다음과 같다.

```text
policy server listening on ws://0.0.0.0:8000
[Start] Task B-1
[obs0] type: 'observation'
[obs0] sim_time: 0.0
[obs0] images: {head_l: ndarray(376, 672, 3) uint8, wrist_l: ndarray(240, 424, 3) uint8, wrist_r: ndarray(240, 424, 3) uint8}
[obs0] state: {joint_q: ndarray(18,) float32, lift: ndarray(1,) float32, mobile: ndarray(3,) float32}
[obs0] instruction: 'pick up the can and place it on the shelf'
[obs] t=0.000 head_l(376, 672, 3) joint_q[18] lift[1] mobile[3] instr='pick up the can and place it on the shelf'
[Done] Task B-1 | full marks (early stop)
```

`[obs0]` 줄은 첫 틱 관측의 전체 구조다.

## 프로토콜 요약

메시지 흐름은 접속 → `reset`/`ready` → (`observation`/`action`) 반복 → `done`이며 연결 하나가
에피소드 1회에 대응한다. 메시지 필드, 관측·액션 형식, 22차원 레이아웃, 종료 사유, 동작 규칙은
[docs/protocol.md](docs/protocol.md)에 있다. 아래 두 표는 그 문서의 요약이다.

### 과제별 규약

| `task_id` | 과제 | 에피소드 | 틱 상한 | 에피소드 만점 | 잠기는 슬롯 |
| --- | --- | --- | --- | --- | --- |
| `ConvStore-Task-A` | 바구니 운반 | 3 | 12,000틱 | 0 | 없음. 22차원 전부 자유 |
| `ConvStore-Task-B` | 상자에서 진열대로 진열 | 3 | 12,000틱 | 0 | 없음. 22차원 전부 자유 |
| `ConvStore-Task-C` | 계산대 QR 스캔 | 3 | 12,000틱 | 0 | 14, 18, 19–21 |

세 과제 모두 `action_dim`은 22이고 `control_hz`는 20이며, 12,000틱은 시뮬레이션 10분이다.
점수는 에피소드 합산이다. 에피소드 만점은 채점 기준 확정 전이라 0으로 두며 확정 시 갱신한다.
Task C는 스캐너를 쥔 오른손이 고정되므로 오른손 그리퍼와 리프트, 베이스 슬롯이 잠긴다.

### 한도와 타임아웃

| 항목 | 값 | 초과 시 동작 |
| --- | --- | --- |
| 추론 응답 타임아웃 | 30초 | 해당 에피소드를 실패로 기록한다 |
| 액션 청크 길이 | `max_chunk_len` 스텝 | 뒤쪽 초과분을 버린다 |
| 메시지 크기 | 32 MiB | 평가 서버가 연결을 종료한다 |
| 만점 상태 유지 | 시뮬레이션 2.5초 | 에피소드를 조기 종료한다 |
| 총점 무진전 지속 | 시뮬레이션 180초 | 에피소드를 종료한다 |
| keepalive ping 간격 | 30초 | 응답이 120초 없으면 연결을 종료한다 |

## API 레퍼런스

공개 API는 `from evalhost import ...` 한 줄로 임포트한다. 내부 모듈 경로는 참고용이며 하위
호환을 보장하는 이름은 아래 표의 이름이다.

| 이름 | 모듈 | 시그니처 | 설명 |
| --- | --- | --- | --- |
| `BasePolicy.reset` | `evalhost.policy` | `reset(self, msg: dict) -> None` | 에피소드 시작 처리. 재정의는 선택 |
| `BasePolicy.infer` | `evalhost.policy` | `infer(self, obs: dict) -> np.ndarray` | 관측을 받아 액션 청크 반환. 필수 |
| `serve_policy` | `evalhost.server` | `serve_policy(policy, host="0.0.0.0", port=8000, token=None, certfile=None, keyfile=None) -> None` | 정책 서버 실행 |
| `action_groups` | `evalhost.actions` | `action_groups(joint_q=None, lift=None, mobile=None) -> dict` | 구조화 액션 dict 생성 |
| `fill_action` | `evalhost.actions` | `fill_action(action, action_dim=None, chunk_len=1) -> np.ndarray` | 부분 지정 액션을 0으로 채움 |
| `decode_depth` | `evalhost.images` | `decode_depth(data) -> np.ndarray` | 16-bit mm PNG를 float32 미터 배열로 변환 |
| `cli` | `evalhost.cli` | `cli(argv=None) -> argparse.Namespace` | 공통 명령행 인자 파싱 |

## 설정

`reset`의 `conf`는 평가 서버가 정하고 정책 서버는 값을 읽기만 한다.

| 키 | 타입 | 값 | 설명 |
| --- | --- | --- | --- |
| `action_dim` | `int` | 22 | 통합 액션 벡터 차원 |
| `control_hz` | `int` | 20 | 액션 1개가 적용되는 주기. 청크 50개가 시뮬레이션 2.5초 |
| `max_chunk_len` | `int` | 50 | 청크 길이 상한 |
| `image_size` | `str` | `real` | 관측 해상도. `real` 고정 |

명령행 인자는 `evalhost-log`와 `evalhost-demo`가 공유한다. `--token`은 환경 변수
`EVALHOST_TOKEN`으로도 전달하며 명령행 인자가 환경 변수보다 우선한다.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--host` | `0.0.0.0` | 수신 대기 주소 |
| `--port` | `8000` | 대기 포트 |
| `--token` | `EVALHOST_TOKEN` | 제출 토큰. 지정하거나 환경 변수가 있으면 접속 인증을 요구한다 |
| `--certfile` | - | TLS 인증서 PEM. 지정하면 `wss://`로 수신 대기한다 |
| `--keyfile` | - | TLS 개인키 PEM. `--certfile`과 함께 쓴다 |
| `--save-obs` | - | `evalhost-log` 전용. 관측 이미지를 매 틱 덮어쓸 디렉터리 |
| `--save-dir` | `./received` | `evalhost-demo` 전용. 첫 관측을 저장할 폴더 |

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `src/evalhost/policy.py` | 참가자가 상속하는 `BasePolicy` 정의 |
| `src/evalhost/actions.py` | 액션 레이아웃과 `fill_action`, `action_groups` |
| `src/evalhost/server.py` | `serve_policy`. 수신 대기, Bearer 인증, TLS |
| `src/evalhost/protocol.py` | msgpack 직렬화와 수신 배열 검증 |
| `src/evalhost/images.py` | JPEG 지연 디코드와 depth 변환 |
| `src/evalhost/demo.py` | `evalhost-demo`의 데모 정책과 관측 저장 |
| `src/evalhost/debug.py` | `evalhost-log`의 `LogPolicy`와 관측 출력 |
| `src/evalhost/cli.py` | 명령행 인자 파싱과 진입점 |
| `examples/my_policy.py` | 제출용 정책 템플릿 |
| `examples/demo_server.py` | `BasePolicy` 상속 최소 예제 |
| `scripts/wss_setup.sh` | self-signed 인증서 생성과 지문 출력 |
| `server.py`, `demo_server.py` | 구버전 호환 shim. v1.0.0에서 삭제 |
| `docs/protocol.md` | 프로토콜 레퍼런스. 메시지 필드, 관측·액션 형식, 22차원 레이아웃, 종료 사유, 동작 규칙 |
| `docs/faq.md` | Q&A와 트러블슈팅 |
| `tests/` | 프로토콜 검증, 액션 변환, CLI, 서버 스모크 테스트 |

## MILAB 서버 설정 가이드

MILAB 서버에서 정책 서버를 띄우고 제출 페이지 접속 테스트까지 통과한 절차다. 2026-09-07에
아래 환경에서 확인했다. 다른 서버에서는 [포트 번호 바꾸기](#다른-서버에서-포트-번호-바꾸기)
절을 따라 같은 순서로 진행한다.

| 항목 | 값 |
| --- | --- |
| 호스트 | `milab-3090` (Ubuntu, Linux 5.15) |
| 공인 IP | `203.~~~.~~~.~~~` (NAT 없이 머신이 직접 보유) |
| 포트 | `8000` |
| 방화벽 | `ufw` (초기 상태 `inactive`) |
| Python | 3.10, `.venv` 가상환경 |
| 제출 주소 | `ws://203.~~~.~~~.~~~:8000` |

### 검증된 절차

1. 포크를 복제하고 가상환경에 editable로 설치한다.

   ```bash
   git clone https://github.com/dla020501/eval-host-server.git
   cd eval-host-server
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. 제출 페이지에서 발급받은 토큰을 환경 변수로 넣는다. 셸을 새로 열면 다시 넣어야 한다.

   ```bash
   export EVALHOST_TOKEN=<발급받은 토큰>
   ```

3. 데모 서버를 띄운다. 아래 한 줄이 뜨면 프로세스는 정상이다. **그 뒤로는 평가 서버가
   접속해 올 때까지 아무 출력도 없다.** 정책 서버는 접속을 받기만 하므로 멈춘 것이 아니다.

   ```bash
   evalhost-demo --port 8000 --save-dir ./received
   ```

   ```text
   policy server listening on ws://0.0.0.0:8000
   ```

4. 다른 터미널에서 로컬 도달을 확인한다. 토큰이 걸려 있으므로 `401 Unauthorized`가 돌아오면
   정상이다. 401은 핸들러 전에 처리되어 서버 터미널에는 로그가 남지 않는다.

   ```bash
   curl -i http://127.0.0.1:8000
   ```

   ```text
   HTTP/1.1 401 Unauthorized
   Server: Python/3.10 websockets/16.1.1

   unauthorized
   ```

5. OS 방화벽을 확인하고 포트를 연다. MILAB 서버는 `ufw`가 `inactive`라 실제로 막고 있는
   규칙은 없었고, `allow`는 나중에 `ufw enable`을 하더라도 8000이 열려 있도록 미리 등록한
   것이다.

   ```bash
   sudo ufw status            # Status: inactive
   sudo ufw allow 8000/tcp
   ```

6. 제출 페이지에 `ws://203.~~~.~~~.~~~:8000`과 토큰을 등록하고 접속 테스트를 실행한다.
   성공하면 서버 터미널에 `[Start]` 로그가 찍히고 `received/ep0/`에 관측 파일 여덟 개가
   생긴다.

7. 접속 테스트가 통과하면 데모 서버를 내리고 같은 포트에서 `my_policy.py`를 띄운다.
   평가가 끝날 때까지 프로세스가 살아 있어야 하므로 `tmux`나 `nohup`으로 실행한다.

   ```bash
   tmux new -s policy
   source .venv/bin/activate
   export EVALHOST_TOKEN=<발급받은 토큰>
   python my_policy.py --port 8000
   # Ctrl+b, d 로 분리. 다시 붙을 때는 tmux attach -t policy
   ```

### 다른 서버에서 포트 번호 바꾸기

포트는 명령행 `--port`로 정하고 제출 페이지에는 같은 번호를 등록한다. 코드 수정은 없다.
아래는 포트를 `PORT` 변수로 두고 새 서버에서 처음부터 진행하는 순서다.

1. 포트를 정한다. 1024 이상이고 아직 아무 프로세스도 쓰지 않는 번호를 고른다. 출력이 비어
   있으면 사용 가능하다.

   ```bash
   PORT=8000
   ss -ltn | grep ":$PORT "
   ```

2. 공인 IP를 확인한다. 두 값이 같으면 머신이 공인 IP를 직접 갖고 있는 것이고, 다르면 NAT
   뒤에 있어 라우터에서 포트포워딩이 필요하다. 사설 IP(`10.`, `172.16–31.`, `192.168.`)만
   나오면 평가 서버가 도달하지 못한다.

   ```bash
   hostname -I          # 머신이 가진 IP
   curl -s ifconfig.me  # 외부에서 보이는 IP
   ```

3. OS 방화벽을 연다. `ufw`가 `inactive`면 막는 것이 없지만 규칙은 등록해 둔다. `active`면
   `allow`가 실제로 필요하다. **`ufw enable`을 새로 하려면 먼저 `sudo ufw allow ssh`를 실행한다.**
   빼먹으면 SSH 접속이 끊긴다.

   ```bash
   sudo ufw status
   sudo ufw allow "$PORT"/tcp
   ```

   `ufw`가 없는 배포판은 `iptables`나 `firewalld`를 쓴다.

   ```bash
   # firewalld
   sudo firewall-cmd --permanent --add-port="$PORT"/tcp && sudo firewall-cmd --reload
   # iptables
   sudo iptables -I INPUT -p tcp --dport "$PORT" -j ACCEPT
   ```

4. 데모 서버를 그 포트로 띄우고 로컬에서 401이 오는지 확인한다.

   ```bash
   export EVALHOST_TOKEN=<발급받은 토큰>
   evalhost-demo --port "$PORT" --save-dir ./received
   # 다른 터미널에서
   curl -i http://127.0.0.1:"$PORT"
   ```

5. 외부 도달을 확인한다. 이 머신이 아닌 다른 망(휴대폰 테더링, 집)에서 실행한다. 같은 401이
   오면 열린 것이고, 타임아웃이면 OS 방화벽, 클라우드 보안 그룹, 기관 방화벽 순으로 확인한다.
   기관 방화벽은 머신에서 열 수 없으므로 관리자에게 인바운드 개방을 요청한다.

   ```bash
   curl -i http://<공인IP>:<PORT>
   ```

6. 제출 페이지에 `ws://<공인IP>:<PORT>`를 등록하고 접속 테스트를 실행한다. 통과하면
   [검증된 절차](#검증된-절차) 7번과 같이 `my_policy.py --port "$PORT"`로 교체한다.

### 증상별 확인

| 증상 | 확인할 것 |
| --- | --- |
| `listening` 줄이 안 뜨고 종료된다 | 포트 충돌. `ss -ltn`으로 같은 포트를 쓰는 프로세스를 찾아 다른 포트를 쓴다 |
| `listening`은 떴는데 아무 로그가 없다 | 정상. 평가 서버가 접속해야 로그가 생긴다. 제출 페이지에서 접속 테스트를 실행한다 |
| 로컬 `curl`은 401인데 외부 `curl`은 타임아웃 | 방화벽. `ufw status`, 클라우드 보안 그룹, 기관 방화벽 순으로 본다 |
| 외부 `curl`도 401인데 접속 테스트가 실패한다 | 토큰 불일치 또는 주소 오타. 제출 페이지의 토큰과 `EVALHOST_TOKEN`, `ws://IP:PORT`를 대조한다 |
| 셸을 새로 열었더니 접속 테스트가 실패한다 | `EVALHOST_TOKEN`은 셸마다 다시 넣어야 한다. 서버를 띄운 셸에서 `echo $EVALHOST_TOKEN`으로 확인한다 |
| 터미널을 닫았더니 평가가 실패했다 | 서버 프로세스가 같이 종료됐다. `tmux`나 `nohup`으로 띄운다 |

## 배포·운영 요구사항

- 정책 서버는 공인 IP 또는 도메인의 열린 포트에 있어야 한다. 사설 IP와 NAT 뒤 주소는
  평가 서버가 도달하지 못하므로 제출이 실패한다.
- 제출 페이지에서 발급받은 토큰을 환경 변수 `EVALHOST_TOKEN`으로 전달한다. 평가 서버는
  `Authorization: Bearer <토큰>` 헤더를 전송하고 정책 서버가 이를 검증한다. 토큰이 없으면
  주소를 아는 누구나 접속해 모델 출력을 얻을 수 있다. `--token <토큰>`으로도 전달할 수
  있으나 명령행 인자는 같은 머신의 다른 사용자가 프로세스 목록에서 볼 수 있으므로 공유
  머신에서는 환경 변수를 쓴다.
- **정책 서버는 평가가 끝날 때까지 실행 상태를 유지해야 한다.** 제출 후 큐 대기와 여러
  에피소드 평가에 수십 분이 소요된다. 도중에 종료되면 그 평가는 실패로 기록된다.
- 연결은 에피소드 1회 동안 유지된다. 정책 서버는 `done` 수신과 연결 종료를 모두 에피소드
  끝으로 처리한다. 강제 중단 시에는 `done` 없이 연결만 끊긴다.
- 공유 망에서는 `wss://`를 사용한다. `ws://`는 평문이므로 같은 망의 타인이 토큰을 볼 수
  있다. `scripts/wss_setup.sh`가 인증서 생성부터 제출값 출력까지 처리한다.

```bash
curl -fsSLO https://raw.githubusercontent.com/kairobahq/eval-host-server/v0.1.1/scripts/wss_setup.sh
bash wss_setup.sh 203.0.113.7
```

저장소를 복제해 설치한 경우에는 `./scripts/wss_setup.sh 203.0.113.7`로 실행한다. 스크립트는
`openssl`이 필요하다.

출력된 인증서 지문을 제출 페이지에 등록하면 평가 서버가 그 지문의 인증서만 신뢰한다.
인증서를 다시 발급하면 지문이 바뀌므로 제출 페이지도 갱신한다. 정식 인증서는 등록이 필요 없다.

```bash
export EVALHOST_TOKEN=<발급받은 토큰>
python my_policy.py --port 8000 --certfile cert.pem --keyfile key.pem
```

## Q&A와 트러블슈팅

참가자가 자주 묻는 질문과 증상별 조치는 [docs/faq.md](docs/faq.md)에 있다.

## 라이선스

MIT 라이선스로 배포한다. 저작권자는 Kairoba다. 전문은 [LICENSE](LICENSE)에 있다.
