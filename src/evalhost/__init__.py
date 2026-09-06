"""참가자 정책 서버 템플릿.

BasePolicy를 상속해 infer()를 구현하고 serve_policy()로 실행한다.
"""

from importlib.metadata import PackageNotFoundError, version

from evalhost.actions import ACTION_SLICES, action_groups, fill_action
from evalhost.cli import cli
from evalhost.images import decode_depth
from evalhost.policy import BasePolicy
from evalhost.server import serve_policy

try:
    __version__ = version("eval-host-server")     # 단일 출처는 pyproject.toml
except PackageNotFoundError:                      # 설치되지 않은 소스 트리에서 실행된 경우
    __version__ = "0+unknown"

__all__ = ["ACTION_SLICES", "BasePolicy", "__version__", "action_groups", "cli",
           "decode_depth", "fill_action", "serve_policy"]
