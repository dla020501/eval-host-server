"""액션 레이아웃과 부분 지정 액션 채우기. 참가자가 반환값을 만들 때 쓰는 헬퍼다."""

import numpy as np

# 통합 액션 벡터의 구간 이름. fill_action()으로 일부 구간만 채울 때 쓴다.
# 태스크별로 허용되지 않는 자유도는 **평가 서버가 0으로 마스킹**한다 (보내도 무시된다).
ACTION_SLICES = {
    "arm_r": slice(0, 7), "arm_l": slice(7, 14),
    "gripper_r": slice(14, 15), "gripper_l": slice(15, 16),
    "head": slice(16, 18), "lift": slice(18, 19), "base": slice(19, 22),
}

# 통합 IO v2 구조화 액션 그룹 (관측 state와 같은 이름·같은 순서). action_groups() 참조.
ACTION_GROUP_DIMS = {"joint_q": 18, "lift": 1, "mobile": 3}


def action_groups(joint_q=None, lift=None, mobile=None) -> dict:
    """v2 구조화 액션 `{"joint_q": (T,18), "lift": (T,1), "mobile": (T,3)}`을 만든다.

    쓰는 그룹만 지정하면 되고 **누락 그룹은 평가 서버가 0으로 채운다**. joint_q 순서는 관측
    `state["joint_q"]`와 같다 (arm_r7, arm_l7, grip_r, grip_l, head2 — 그리퍼는 연속 각도).

    Args:
        joint_q: (T, 18) 또는 (18,) 관절 타깃.
        lift: (T, 1), (T,) 또는 스칼라 리프트 타깃.
        mobile: (T, 3) 또는 (3,) 베이스 속도 명령.

    Returns:
        지정한 그룹만 담은 dict. 값은 float32 2차원 배열이다.
    """
    out = {}
    for key, value in (("joint_q", joint_q), ("lift", lift), ("mobile", mobile)):
        if value is None:
            continue
        a = np.asarray(value, np.float32)
        out[key] = (a.reshape(-1, 1) if ACTION_GROUP_DIMS[key] == 1 and a.ndim < 2
                    else np.atleast_2d(a))
    return out


def fill_action(action, action_dim: int | None = None, chunk_len: int = 1) -> np.ndarray:
    """부분 지정 액션을 (T, action_dim) float32 청크로 0 채워 완성한다.

    - v2 구조화 dict (`action_groups()`의 결과): 그대로 통과한다 — 22-dim 조립과 zero-fill은
      평가 서버가 수행한다.
    - dict: 쓰려는 구간만 지정한다 -> `{"arm_r": q7, "gripper_r": -1.0}`.
      키는 ACTION_SLICES 이름 또는 인덱스/슬라이스다. 나머지 자유도는 전부 0이다.
    - 짧은 벡터/청크: 뒤를 0으로 패딩한다. 1차원이면 (1, action_dim)으로 본다.
    - 이미 (T, action_dim)이면 그대로 통과한다.

    Args:
        action: (T, action_dim) 배열, 짧은 벡터, 구간 이름 dict, 또는 v2 구조화 dict.
        action_dim: 통합 액션 차원. reset의 conf로 전달된다. dict 액션에는 필수다.
        chunk_len: dict 액션으로 만들 청크 길이.

    Returns:
        (T, action_dim) float32 배열. v2 구조화 dict는 그대로 통과한다.

    Raises:
        ValueError: dict 액션인데 action_dim이 없을 때.
    """
    if isinstance(action, dict):
        if action and not action.keys() - ACTION_GROUP_DIMS.keys():
            return action
        if action_dim is None:
            raise ValueError("dict 액션에는 action_dim이 필요하다 (reset의 conf에 실려 온다)")
        out = np.zeros((chunk_len, action_dim), np.float32)
        for key, value in action.items():
            out[:, ACTION_SLICES[key] if isinstance(key, str) else key] = value
        return out
    a = np.atleast_2d(np.asarray(action, dtype=np.float32))
    if action_dim and a.shape[1] < action_dim:
        a = np.pad(a, ((0, 0), (0, action_dim - a.shape[1])))
    return a
