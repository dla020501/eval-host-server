"""명령행 인자와 버전 단일 출처."""

from importlib.metadata import version

import evalhost
from evalhost import cli


def test_defaults():
    args = cli([])
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.token is None
    assert args.certfile is None and args.keyfile is None
    assert args.save_obs is None and args.save_dir is None


def test_parses_all_arguments():
    args = cli(["--host", "127.0.0.1", "--port", "9001", "--token", "t",
                "--certfile", "cert.pem", "--keyfile", "key.pem",
                "--save-obs", "dump", "--save-dir", "received"])
    assert args.host == "127.0.0.1"
    assert args.port == 9001
    assert args.token == "t"
    assert args.certfile == "cert.pem" and args.keyfile == "key.pem"
    assert args.save_obs == "dump" and args.save_dir == "received"


def test_token_env_fallback_and_cli_priority(monkeypatch):
    monkeypatch.setenv("EVALHOST_TOKEN", "from-env")
    assert cli([]).token == "from-env"
    assert cli(["--token", "from-cli"]).token == "from-cli"


def test_version_single_source():
    assert evalhost.__version__ == version("eval-host-server")
