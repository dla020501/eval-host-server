# demo_server.py — 구버전 호환 shim. v1.0.0에서 삭제한다.
"""`python demo_server.py` / `from demo_server import ...` 전달자.

새 경로: pip install -e .  ->  evalhost-demo --port 8000 --save-dir ./received
"""

import sys
import warnings

from evalhost.demo import DemoPolicy, main  # noqa: F401

_MSG = "`demo_server` 모듈은 deprecated이다. `evalhost-demo`를 쓴다 (v1.0.0에서 제거)."
warnings.warn(_MSG, DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    # 직접 실행은 기본 경고 필터가 DeprecationWarning을 숨기므로 stderr에 직접 쓴다.
    print(f"DeprecationWarning: {_MSG}", file=sys.stderr)
    raise SystemExit(main())
