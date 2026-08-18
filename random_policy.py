"""더미 정책 예시: 관측을 무시하고 작은 랜덤 액션 청크를 반환한다.

실행: python random_policy.py --port 8000 --chunk-len 10
"""

import numpy as np
from server import BasePolicy, cli, serve_policy


class RandomPolicy(BasePolicy):
    def __init__(self, chunk_len: int = 10, action_dim: int = 7):
        self.chunk_len = chunk_len
        self.action_dim = action_dim

    def reset(self, msg: dict) -> None:
        self.action_dim = msg["conf"]["action_dim"]

    def infer(self, obs: dict) -> np.ndarray:
        return np.random.uniform(-0.01, 0.01, (self.chunk_len, self.action_dim)).astype(np.float32)


if __name__ == "__main__":
    args = cli()
    serve_policy(RandomPolicy(args.chunk_len), args.host, args.port, args.token,
                 args.certfile, args.keyfile)
