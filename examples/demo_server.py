"""BasePolicy 상속 예제. 실행: python examples/demo_server.py --port 8000

관측 shape를 한 줄 출력하고 제로 액션을 응답한다. 제로 액션은 리셋 포즈 유지다.
관측 저장까지 하는 완전한 데모는 `evalhost-demo` 명령이다 (evalhost/demo.py).
"""

import numpy as np

from evalhost import BasePolicy, cli, serve_policy


class DemoPolicy(BasePolicy):
    def reset(self, msg: dict) -> None:
        self.action_dim = (msg.get("conf") or {}).get("action_dim") or 22

    def infer(self, obs: dict) -> np.ndarray:
        head = obs["images"]["head_l"]          # (376, 672, 3) uint8
        print(f"t={obs['sim_time']:.3f} head_l{head.shape} instr={obs['instruction']!r}",
              flush=True)
        return np.zeros((1, self.action_dim), np.float32)


if __name__ == "__main__":
    args = cli()
    serve_policy(DemoPolicy(), args.host, args.port, args.token, args.certfile, args.keyfile)
