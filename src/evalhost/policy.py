"""참가자가 상속하는 정책 인터페이스. 이 파일만 읽으면 제출 서버를 만들 수 있다.

BasePolicy를 상속해 infer()만 채우고 serve_policy()로 실행한다.
"""

import numpy as np


class BasePolicy:
    """평가 서버의 관측을 받아 액션 청크를 반환하는 정책의 기반 클래스다."""

    def reset(self, msg: dict) -> None:
        """에피소드 시작을 처리한다. 재정의는 선택이다.

        Args:
            msg: reset 메시지. `msg["conf"]`에 action_dim / control_hz /
                max_chunk_len(청크 크기 상한)이 들어온다. 실행 방식은 동기 소진 단일이다
                (chunk_mode·inference_hz 폐지).
        """

    def infer(self, obs: dict) -> np.ndarray:
        """관측 1건을 받아 액션 청크를 반환한다.

        obs = {"sim_time": float,
               "images": {"head_l": (376,672,3), "wrist_l"/"wrist_r": (240,424,3) uint8},
               "head_l_depth": (376,672) float32 미터,  # head depth. 0=무효. 손목 depth 없음
               "state": {"joint_q": [18], "lift": [1], "mobile": [3]} float32,
               "scan": float32[960],   # LiDAR가 있는 태스크만. 없으면 키 자체가 없다
               "instruction": str}

        이미지는 꺼내는 순간 JPEG 디코드가 끝난 numpy 배열이다 (해상도는 실기 SDK 기록값
        고정 — head_l 672x376, 손목 424x240. 크롭/리사이즈 등 전처리는 정책이 수행한다).

        Args:
            obs: observation 메시지.

        Returns:
            (T, action_dim) float32 액션 청크. T >= 1.
            평가 서버 conf의 max_chunk_len을 넘으면 앞에서부터 잘린다. 청크는 control_hz로
            동기 소진된 뒤에야 다시 관측·추론한다 (50개 청크 = 약 2.5초 blocking, 실시간
            아님). 일부 자유도만 쓰려면 dict나 짧은 벡터를 반환해도 된다 —
            fill_action()이 0을 채운다. v2 구조화 액션은
            action_groups(joint_q=..., lift=..., mobile=...)로 만든다.

        Raises:
            NotImplementedError: 하위 클래스가 구현하지 않았을 때.
        """
        raise NotImplementedError
