"""workspace 模块测试：统一目录解析与覆盖式设置。"""

from __future__ import annotations

import os
from pathlib import Path

from prompt2vid import workspace


def test_workspace_env_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(tmp_path))
    assert workspace.workspace_root() == tmp_path.resolve()


def test_workspace_default_when_unset(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(workspace.WORKSPACE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    root = workspace.workspace_root()
    assert root == (tmp_path / "output" / "run_default").resolve()
    assert root.is_dir()


def test_set_workspace_overwrites(tmp_path: Path, monkeypatch):
    first = tmp_path / "run_1"
    second = tmp_path / "run_2"
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(first))
    workspace.set_workspace(second)
    # 覆盖而非 setdefault：第二次设置必须生效（server 多请求场景）
    assert os.environ[workspace.WORKSPACE_ENV] == str(second.resolve())
    assert workspace.workspace_root() == second.resolve()
    assert second.is_dir()


def test_resolve_relative_to_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(tmp_path))
    resolved = workspace.resolve_in_workspace("frames/shot_1.png")
    assert resolved == tmp_path / "frames" / "shot_1.png"

    absolute = tmp_path / "elsewhere.png"
    assert workspace.resolve_in_workspace(absolute) == absolute
