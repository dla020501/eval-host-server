"""serve_policy 스모크 테스트. 실제 WebSocket 클라이언트로 한 에피소드를 주고받는다."""

import json
import socket
import threading
import time

import numpy as np
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from evalhost.demo import DemoPolicy
from evalhost.protocol import pack, unpack
from evalhost.server import serve_policy

TOKEN = "test-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("received")
    port = _free_port()
    policy = DemoPolicy(str(save_dir))
    threading.Thread(target=serve_policy,
                     args=(policy, "127.0.0.1", port, TOKEN), daemon=True).start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:   # 바인딩될 때까지 기다린다
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("policy server did not start")
    return f"ws://127.0.0.1:{port}", save_dir


def test_reset_observation_done_exchange(server):
    uri, save_dir = server
    images = {cam: np.random.randint(0, 255, (8, 12, 3), np.uint8)
              for cam in ("head_l", "wrist_l", "wrist_r")}
    with connect(uri, additional_headers={"Authorization": f"Bearer {TOKEN}"},
                 max_size=None) as ws:
        ws.send(pack({"type": "reset", "episode_id": "job0-ep0", "task_id": "ConvStore-Task-B",
                      "instruction": "restock", "conf": {"action_dim": 22, "control_hz": 20},
                      "server_info": {"type": "Start", "info": "Task B-1"}}))
        assert unpack(ws.recv(timeout=10)) == {"type": "ready"}

        ws.send(pack({"type": "observation", "sim_time": 0.0, "images": images,
                      "state": {"joint_q": np.zeros(18, np.float32),
                                "lift": np.zeros(1, np.float32),
                                "mobile": np.zeros(3, np.float32)},
                      "scan": np.full(960, 20.0, np.float32), "instruction": "restock"}))
        reply = unpack(ws.recv(timeout=10))
        assert reply["type"] == "action"
        assert reply["actions"].shape == (1, 22)
        assert reply["actions"].dtype == np.float32
        assert not reply["actions"].any()

        ws.send(pack({"type": "ping-from-the-future"}))   # 모르는 type은 무시한다
        ws.send(pack({"type": "done", "reason": "max_ticks",
                      "server_info": {"type": "Done", "info": "Task B-1"}}))

    ep = save_dir / "ep0"
    for name in ("head_l.png", "wrist_l.png", "wrist_r.png", "scan.npy", "obs.json"):
        assert (ep / name).is_file()
    assert json.loads((ep / "obs.json").read_text())["instruction"] == "restock"


def test_wrong_token_is_rejected_with_401(server):
    uri, _ = server
    with pytest.raises(InvalidStatus) as exc:
        connect(uri, additional_headers={"Authorization": "Bearer wrong"})
    assert exc.value.response.status_code == 401
