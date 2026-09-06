"""명령행 진입점. evalhost-log·python -m evalhost·호환 shim이 모두 main()을 호출한다."""

import argparse
import os

from evalhost.debug import LogPolicy
from evalhost.server import serve_policy


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    """정책 서버 공통 명령행 인자를 파싱한다.

    설정 우선순위는 명령행 인자, 환경변수, 기본값 순이다.

    Args:
        argv: 파싱할 인자 목록. None이면 sys.argv를 쓴다.

    Returns:
        파싱 결과 Namespace.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token", default=os.environ.get("EVALHOST_TOKEN"),
                   help="홈페이지에서 발급받은 제출 토큰. 지정하면 접속 인증을 요구한다 "
                        "(권장). 환경변수 EVALHOST_TOKEN으로도 전달할 수 있다.")
    p.add_argument("--certfile", default=None,
                   help="TLS 인증서(PEM). 지정하면 wss://로 연다. self-signed면 지문을 "
                        "홈페이지에 등록한다.")
    p.add_argument("--keyfile", default=None, help="TLS 개인키(PEM). --certfile과 함께 쓴다.")
    p.add_argument("--save-obs", default=None, metavar="DIR",
                   help="evalhost-log 전용. 관측 이미지를 이 디렉터리에 카메라별 파일로 "
                        "저장한다 (매 틱 덮어쓰기).")
    p.add_argument("--save-dir", default=None,
                   help="evalhost-demo 전용. 받은 관측을 저장할 폴더 (다른 정책은 무시한다).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """관측 확인용 LogPolicy 서버를 실행한다 (`evalhost-log`).

    Args:
        argv: 파싱할 인자 목록. None이면 sys.argv를 쓴다.
    """
    args = cli(argv)
    serve_policy(LogPolicy(args.save_obs), args.host, args.port, args.token,
                 args.certfile, args.keyfile)
