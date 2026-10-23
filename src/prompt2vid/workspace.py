"""统一的工作目录解析。

之前 5 个工具模块各复制了一份 `_workspace()`，且未设置环境变量时默认
`output/run_default`，与 main.py 的 `output/run_<时间戳>` 不一致；本模块
收敛为唯一定义，供 main / server / tools / stages 共用。

约定：
- `PROMPT2VID_WORKSPACE` 环境变量由入口（CLI / A2A 服务器）在构建 Crew 前
  显式设置（覆盖式），工具与阶段引擎运行期只读。
- 相对路径一律相对工作目录解析。
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ENV = "PROMPT2VID_WORKSPACE"
DEFAULT_WORKSPACE = Path("output") / "run_default"


def set_workspace(path: str | Path) -> Path:
    """设置本次运行的工作目录（覆盖式写入环境变量）。

    main / server 每次运行调用；覆盖而非 setdefault，避免上一次运行的
    目录泄漏到下一次（server 多请求共享进程时曾因此互相串目录）。
    """
    run_dir = Path(path).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ[WORKSPACE_ENV] = str(run_dir)
    return run_dir


def workspace_root(explicit: str | Path | None = None) -> Path:
    """获取当前工作目录（不存在则创建）。

    Args:
        explicit: 显式指定目录；缺省读 PROMPT2VID_WORKSPACE，
            再缺省回落到 output/run_default。
    """
    if explicit is not None:
        path = Path(explicit)
    else:
        value = os.getenv(WORKSPACE_ENV, "").strip()
        path = Path(value) if value else DEFAULT_WORKSPACE
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_in_workspace(
    path_str: str | Path, workspace: str | Path | None = None
) -> Path:
    """把路径解析为绝对路径：绝对路径原样，相对路径挂到工作目录下。"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return workspace_root(workspace) / path
