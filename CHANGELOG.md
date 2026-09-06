# 변경 이력

형식은 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)를 따르고 버전은
[유의적 버전](https://semver.org/lang/ko/)을 따른다.

## [Unreleased]

## [0.1.0] - 2026-09-07

### Added

- 파이썬 패키지 `evalhost`로 재구성했다. 설치는 `pip install -e .`이다.
- 콘솔 명령 `evalhost-demo`(데모 정책)와 `evalhost-log`(관측 확인)를 추가했다.
- 예제를 `examples/`에, 운영 스크립트를 `scripts/`에 두었다.

### Changed

- `LogPolicy`가 관측 `joint_q` 대신 제로 액션을 반환한다. 액션 슬롯은 리셋 포즈 기준
  오프셋이라 절대 관절각을 그대로 반환하면 로봇이 급격히 움직인다.
- `wss_setup.sh`를 `scripts/wss_setup.sh`로 옮겼다.

### Deprecated

- 루트의 `server.py`와 `demo_server.py`는 호환 shim이며 v1.0.0에서 삭제한다.

### Removed

- `requirements.txt`를 삭제했다. 의존성은 `pyproject.toml`에 있다.
