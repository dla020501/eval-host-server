# server.py — 구버전 호환 shim. v1.0.0에서 삭제한다.
"""`python server.py` / `from server import ...` 전달자.

새 경로: pip install -e .  ->  from evalhost import BasePolicy, serve_policy
직접 실행하면 관측 확인용 LogPolicy가 시작된다 (`evalhost-log`와 같다).
"""

import sys
import warnings

from evalhost import *  # noqa: F401,F403  (__all__ 만 넘어온다)
from evalhost.actions import ACTION_GROUP_DIMS  # noqa: F401
from evalhost.cli import main
from evalhost.debug import LogPolicy, describe, log_server_info, save_images  # noqa: F401
from evalhost.images import LazyImages, decode_jpeg  # noqa: F401
from evalhost.protocol import ARRAY_MAX_ELEMS, decode_ndarray, pack, unpack  # noqa: F401

_MSG = "`server` 모듈은 deprecated이다. `evalhost`를 임포트한다 (v1.0.0에서 제거)."
warnings.warn(_MSG, DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    # 직접 실행은 기본 경고 필터가 DeprecationWarning을 숨기므로 stderr에 직접 쓴다.
    print(f"DeprecationWarning: {_MSG} -> evalhost-log", file=sys.stderr)
    raise SystemExit(main())
