"""데모 정책 서버. 통신 규약만 맞춰 제로 액션을 응답하고 받은 관측을 파일로 저장한다.

실행:
    evalhost-demo --port 8000 --token <제출토큰> --save-dir ./received

정책 로직은 없다. 프로토콜 연결을 확인하고 실제로 어떤 관측이 오는지 확인하는 용도다.
각 에피소드의 첫 관측을 --save-dir에 저장한다 (에피소드마다 하위 폴더):
    head_l.png / wrist_l.png / wrist_r.png   실기 해상도 RGB
    head_l_depth.png                         16-bit mm 원본
    head_l_depth_view.png                    미리보기(정규화)
    scan.npy / scan_view.png                 LiDAR 원본과 top-down 미리보기
    obs.json                                 sim_time·instruction·state·scan 요약

액션은 전부 0이라 로봇은 리셋 포즈를 유지한다. 채점은 0점이지만, 관측 확인과 자기 추론
스택을 붙이기 전 연결 검증에는 충분하다. depth는 시뮬레이터의 이상화 값이다
(evalhost.images.decode_depth 참조). 손목 카메라에는 depth가 없다.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from evalhost.cli import cli
from evalhost.images import LazyImages, decode_depth
from evalhost.policy import BasePolicy
from evalhost.server import serve_policy


class DemoPolicy(BasePolicy):
    """제로 액션을 반환하고 에피소드마다 첫 관측을 저장하는 예제 정책이다."""

    def __init__(self, save_dir: str = "./received"):
        self.save_dir = Path(save_dir)
        self.action_dim = 22
        self.control_hz = 20
        self.ep = -1
        self._saved_this_ep = False

    def reset(self, msg: dict) -> None:
        conf = msg.get("conf") or {}
        self.action_dim = int(conf.get("action_dim") or 22)
        self.control_hz = int(conf.get("control_hz") or 20)
        self.ep += 1
        self._saved_this_ep = False

    def infer(self, obs: dict) -> np.ndarray:
        # 저장은 진단용이다. 규격 밖 관측(연결 테스트의 더미 등)으로 저장이 실패해도
        # 액션 응답은 계속한다 -- 저장 실패가 에피소드 실패가 되면 안 된다.
        if not self._saved_this_ep:
            try:
                self._save(obs)
            except Exception as exc:  # noqa: BLE001
                print(f"[demo] ep{self.ep} 관측 저장 실패: {type(exc).__name__}: {exc}", flush=True)
            self._saved_this_ep = True
        return np.zeros((1, self.action_dim), np.float32)

    def _save(self, obs: dict) -> None:
        d = self.save_dir / f"ep{self.ep}"
        d.mkdir(parents=True, exist_ok=True)
        images = obs.get("images")
        if not isinstance(images, (dict, LazyImages)):
            images = {}
        for cam in ("head_l", "wrist_l", "wrist_r"):
            if cam in images:  # 꺼내는 순간 uint8 (H,W,3)로 디코드된다
                Image.fromarray(np.asarray(images[cam])).save(d / f"{cam}.png")
        if "head_l_depth" in obs:  # float32 미터, 0=무효. 원본(mm)과 미리보기를 저장
            depth_m = decode_depth(obs["head_l_depth"])
            mm = np.rint(np.clip(depth_m * 1000.0, 0, 65535)).astype(np.uint16)
            Image.fromarray(mm, mode="I;16").save(d / "head_l_depth.png")
            Image.fromarray(_depth_preview(depth_m)).save(d / "head_l_depth_view.png")
        if "scan" in obs:  # LiDAR 병합 스캔 float32[960], 원본과 top-down 미리보기를 저장
            scan = np.asarray(obs["scan"], np.float32)
            np.save(d / "scan.npy", scan)
            Image.fromarray(_scan_preview(scan)).save(d / "scan_view.png")
        state = obs.get("state")
        if state is None:
            state = {}
        summary = {
            "sim_time": obs.get("sim_time"),
            "instruction": obs.get("instruction"),
            "images": {c: list(np.asarray(images[c]).shape) for c in images},
            "state": {k: list(np.asarray(v).shape) for k, v in state.items()}
            if isinstance(state, dict) else f"flat{list(np.asarray(state).shape)}",
            "has_head_l_depth": "head_l_depth" in obs,
            "has_scan": "scan" in obs,
            "scan_len": int(np.asarray(obs["scan"]).size) if "scan" in obs else None,
        }
        (d / "obs.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[demo] ep{self.ep} 관측 저장: {d}  {summary}", flush=True)


def _scan_preview(scan: np.ndarray, size: int = 480, view_m: float = 5.0) -> np.ndarray:
    """LiDAR 병합 스캔을 top-down 2D 이미지로 그린다. 로봇은 중앙(위=전방, 왼=좌).

    빈 i의 각도는 -pi + i*(2*pi/960) (base_link atan2(y,x)), 값은 거리(m), 20은 무반사다.
    view_m 반경까지만 그린다.
    """
    from PIL import ImageDraw

    n = len(scan)
    img = Image.new("RGB", (size, size), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    c = size // 2
    scale = c / view_m
    for r in range(1, int(view_m) + 1):   # 거리 링 (1m 간격)
        rr = int(r * scale)
        dr.ellipse([c - rr, c - rr, c + rr, c + rr], outline=(45, 45, 55))
    dr.line([c, 0, c, size], fill=(45, 45, 55))
    dr.line([0, c, size, c], fill=(45, 45, 55))
    for i in range(n):   # 유효 반사만 점으로
        rng = float(scan[i])
        if not (0.0 < rng < view_m):   # 무반사(약 20m)나 범위 밖은 생략
            continue
        ang = -math.pi + i * (2.0 * math.pi / n)
        x = rng * math.cos(ang)   # 전방
        y = rng * math.sin(ang)   # 좌
        px = int(c - y * scale)
        py = int(c - x * scale)
        dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(120, 220, 160))
    dr.polygon([(c, c - 9), (c - 6, c + 7), (c + 6, c + 7)], fill=(230, 120, 90))  # 로봇(전방)
    return np.asarray(img)


def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
    """이상화 depth(미터, 0=무효)를 사람이 보기 위한 그레이 RGB로 바꾼다. 가까울수록 밝다."""
    valid = depth_m > 0
    if not valid.any():
        return np.zeros((*depth_m.shape, 3), np.uint8)
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    norm = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
    g = (norm * 255).astype(np.uint8)
    rgb = np.stack([255 - g, 255 - g, 255 - g], -1)   # 가까울수록 흰색
    rgb[~valid] = 0
    return rgb


def main(argv: list[str] | None = None) -> None:
    """데모 정책 서버를 실행한다 (`evalhost-demo`).

    Args:
        argv: 파싱할 인자 목록. None이면 sys.argv를 쓴다.
    """
    args = cli(argv)
    serve_policy(DemoPolicy(args.save_dir or "./received"), args.host, args.port,
                 args.token, args.certfile, args.keyfile)
