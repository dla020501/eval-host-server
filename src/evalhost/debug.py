"""관측 확인용 도구. LogPolicy와 관측 출력·저장 헬퍼가 여기 있다.

실행: evalhost-log --port 8000 --token <제출토큰> --save-obs obs_dump
"""

import os

import numpy as np

from evalhost.policy import BasePolicy


def log_server_info(msg: dict) -> None:
    """reset/done에 포함된 server_info를 한 줄로 출력한다 (type은 Start와 Done 둘뿐이다).

    info는 `"Task Z-N"` 또는 `"Task Z-N | 추가 설명"`이다. 1회 분리 후 양쪽 trim이 규격이라
    설명 안에 `|`가 또 나와도 안전하다. 추가 설명은 자연어이므로 내용에 기대면 안 된다.

    **모르는 키와 모르는 type은 무시한다** — 평가 서버가 나중에 필드를 늘려도 이 서버의
    연결이 끊기지 않게 하는 규칙이다.

    Args:
        msg: reset 또는 done 메시지.
    """
    info = msg.get("server_info")
    if not isinstance(info, dict) or info.get("type") not in ("Start", "Done"):
        return
    task, _, note = str(info.get("info") or "").partition("|")
    print(f"[{info['type']}] {task.strip()}" + (f" | {note.strip()}" if note.strip() else ""),
          flush=True)


def save_images(images: dict, out_dir: str) -> list[str]:
    """관측 이미지를 카메라별 파일로 저장한다. **디코드 전에 호출하는 것이 전제다.**

    받은 그대로가 JPEG 바이트면 재인코딩 없이 그대로 쓰므로 비용이 사실상 0이고 파일이
    평가 서버가 전송한 원본과 바이트 단위로 같다. 무압축 배열로 수신했으면 PNG로 저장한다.

    Args:
        images: 관측의 images dict (LazyImages 또는 평범한 dict).
        out_dir: 저장 디렉터리. 없으면 만든다.

    Returns:
        저장한 파일 경로 목록.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for cam in images:
        raw = dict.__getitem__(images, cam)       # LazyImages 디코드를 우회한 원본
        if isinstance(raw, (bytes, bytearray)):
            path = os.path.join(out_dir, f"{cam}.jpg")
            with open(path, "wb") as f:
                f.write(raw)
        else:
            from PIL import Image

            path = os.path.join(out_dir, f"{cam}.png")
            Image.fromarray(np.asarray(raw, np.uint8)).save(path)
        paths.append(path)
    return paths


def describe(value) -> str:
    """관측 값 하나를 짧은 문자열로 만든다. 규격과 실제가 다를 수 있어 타입까지 보여준다.

    Args:
        value: 관측 dict의 값 하나.

    Returns:
        타입과 형태를 담은 한 줄 문자열.
    """
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape} {value.dtype}"
    if isinstance(value, (bytes, bytearray)):
        return f"bytes[{len(value)}]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {describe(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    return repr(value)


class LogPolicy(BasePolicy):
    """`evalhost-log`로 실행하는 관측 확인용 정책이다.

    에피소드 첫 틱은 관측 구조를 통째로 출력하고, 이후는 한 줄로 줄여 출력한다. save_dir을
    지정하면 이미지를 파일로도 저장한다. 액션은 항상 0이라 로봇은 리셋 포즈를 유지한다
    (관절 슬롯은 리셋 포즈 기준 오프셋이므로 0이 정지 액션이다). 관측이 제대로 오는지
    확인하는 용도이며 정책이 아니다.
    """

    def __init__(self, save_dir: str | None = None):
        self.save_dir = save_dir
        self.action_dim = None
        self.first = True

    def reset(self, msg: dict) -> None:
        self.action_dim = (msg.get("conf") or {}).get("action_dim")
        self.first = True

    def infer(self, obs: dict):
        images, state, scan = obs.get("images"), obs.get("state"), obs.get("scan")
        if self.save_dir and isinstance(images, dict):   # 디코드 전에 저장해야 원본이 남는다
            paths = save_images(images, self.save_dir)
            if self.first:
                print(f"[save] {len(paths)}개 이미지 -> {' '.join(paths)} (매 틱 덮어씀)",
                      flush=True)

        if self.first:                 # 첫 틱만 전체 구조를 출력한다 (규격 대조용)
            self.first = False
            for key, value in obs.items():
                print(f"[obs0] {key}: {describe(value)}", flush=True)

        sim_time = obs.get("sim_time")
        parts = [f"t={sim_time:.3f}" if isinstance(sim_time, float) else f"t={sim_time}"]
        if isinstance(images, dict):   # 꺼내는 순간 디코드되므로 shape는 디코드 결과다
            parts += [f"{cam}{np.shape(images[cam])}" for cam in images]
        if isinstance(state, dict):
            parts += [f"{key}[{np.size(value)}]" for key, value in state.items()]
        elif state is not None:
            parts.append(f"state[{np.size(state)}]")
        if scan is not None:
            parts.append(f"scan[{np.size(scan)}]")
        parts.append(f"instr={obs.get('instruction')!r}")
        print("[obs] " + " ".join(parts), flush=True)

        # 관절 슬롯은 리셋 포즈 기준 오프셋이라 0이 곧 현재 자세 유지다. 관측 joint_q는
        # 절대 관절각이므로 그대로 반환하면 로봇이 급격히 움직인다.
        return np.zeros((1, self.action_dim or 22), np.float32)
