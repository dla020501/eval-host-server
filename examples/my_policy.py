"""참가자 정책 서버 뼈대. 실행: python examples/my_policy.py --port 8000 --token <제출토큰>

load_my_model()은 참가자가 구현한다. 모델 로드는 __init__에서 한 번만 한다.
"""

import numpy as np

from evalhost import BasePolicy, cli, serve_policy


class MyPolicy(BasePolicy):
    def __init__(self):
        self.model = load_my_model()

    # msg 구조는 README의 reset 절을 참조한다.
    def reset(self, msg: dict) -> None:
        self.action_dim = msg["conf"]["action_dim"]

    # obs 구조는 README의 observation 절을 참조한다.
    def infer(self, obs: dict) -> np.ndarray:
        head = obs["images"]["head_l"]
        state = obs["state"]

        # 반환 형식은 README의 action 절을 참조한다.
        return self.model(head, state, obs["instruction"])


if __name__ == "__main__":
    args = cli()
    serve_policy(MyPolicy(), args.host, args.port, args.token, args.certfile, args.keyfile)
