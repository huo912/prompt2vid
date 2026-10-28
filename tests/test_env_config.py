"""env_config 模块测试：档位配置解析（路径/简名/环境变量）与加载。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prompt2vid import env_config


@pytest.fixture()
def clean_env(monkeypatch, tmp_path: Path):
    """隔离的解析环境：清空选择变量、重置进程内记忆、退出后还原 os.environ。"""
    monkeypatch.setattr(env_config, "_loaded_file", None)
    for var in (env_config.ENV_FILE_VAR, env_config.PROFILE_VAR):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    saved = dict(os.environ)
    yield tmp_path
    os.environ.clear()
    os.environ.update(saved)


def test_resolve_explicit_path(clean_env, tmp_path: Path):
    custom = tmp_path / "custom.env"
    custom.write_text("X=1\n", encoding="utf-8")
    assert env_config.resolve_env_file(custom) == custom


def test_resolve_profile_shorthand(clean_env, tmp_path: Path):
    """纯名字（如 4090）展开为 .env.4090；带分隔符或以 . 开头按路径。"""
    (tmp_path / ".env.4090").write_text("X=1\n", encoding="utf-8")
    assert env_config.resolve_env_file("4090") == tmp_path / ".env.4090"
    assert env_config.resolve_env_file(".env.4090") == tmp_path / ".env.4090"
    # 明确的路径写法（含分隔符）不再当简名展开
    assert env_config.resolve_env_file(str(tmp_path / ".env.4090")) == tmp_path / ".env.4090"


def test_resolve_missing_explicit_raises_with_profiles(clean_env, tmp_path: Path):
    (tmp_path / ".env.4060").write_text("X=1\n", encoding="utf-8")
    (tmp_path / ".env.5090").write_text("X=1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as exc:
        env_config.resolve_env_file("4090")
    # 报错要列出可用档位，便于自查
    assert ".env.4060" in str(exc.value)
    assert ".env.5090" in str(exc.value)


def test_resolve_env_vars(clean_env, tmp_path: Path):
    """PROMPT2VID_ENV_FILE 与 PROMPT2VID_PROFILE 都能指定；前者优先。"""
    (tmp_path / ".env.4090").write_text("X=1\n", encoding="utf-8")
    (tmp_path / ".env.5090").write_text("X=1\n", encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv(env_config.PROFILE_VAR, "4090")
        assert env_config.resolve_env_file() == tmp_path / ".env.4090"
        monkeypatch.setenv(env_config.ENV_FILE_VAR, ".env.5090")
        assert env_config.resolve_env_file() == tmp_path / ".env.5090"
    finally:
        monkeypatch.undo()


def test_resolve_default_dotenv(clean_env, tmp_path: Path):
    (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")
    assert env_config.resolve_env_file() == tmp_path / ".env"


def test_resolve_none_when_no_config(clean_env, tmp_path: Path):
    assert env_config.resolve_env_file() is None


def test_available_profiles_excludes_example(clean_env, tmp_path: Path):
    for name in (".env.4060", ".env.4090", ".env.example"):
        (tmp_path / name).write_text("X=1\n", encoding="utf-8")
    assert env_config.available_profiles() == [".env.4060", ".env.4090"]


def test_load_env_applies_values(clean_env, tmp_path: Path):
    (tmp_path / ".env.4090").write_text(
        "P2V_TEST_GPU=4090\nP2V_TEST_EXISTING=from-file\n",
        encoding="utf-8",
    )
    path = env_config.load_env(".env.4090")
    assert path == tmp_path / ".env.4090"
    assert os.environ["P2V_TEST_GPU"] == "4090"
    # 真实环境变量优先，不被文件覆盖
    os.environ["P2V_TEST_EXISTING"] = "from-shell"
    env_config.load_env(".env.4090")
    assert os.environ["P2V_TEST_EXISTING"] == "from-shell"
    assert env_config.loaded_env_file() == tmp_path / ".env.4090"


def test_load_env_idempotent_without_arg(clean_env, tmp_path: Path):
    """首次显式加载后，无参调用（如 build_crew 里）不得把配置换回默认 .env。"""
    (tmp_path / ".env.4090").write_text("P2V_TEST_GPU=4090\n", encoding="utf-8")
    (tmp_path / ".env").write_text("P2V_TEST_GPU=4060\n", encoding="utf-8")

    env_config.load_env(".env.4090")
    assert env_config.load_env() == tmp_path / ".env.4090"
    assert os.environ["P2V_TEST_GPU"] == "4090"


def test_load_env_missing_raises(clean_env, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        env_config.load_env("nonexistent.env")
